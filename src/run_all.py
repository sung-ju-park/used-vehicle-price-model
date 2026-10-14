"""전체 실행: python src/run_all.py
이미 학습한 결과로 분석·리포트만 다시 만들려면: python src/run_all.py --reuse
"""
import argparse
import json
import os
import pickle
import time
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd

import analyze
import intervals
import train
from clean import clean, duplicate_stats
from config import EXPERIMENT, MODELS, OUT, PROCESSED, RAW, REF_YEAR
from features import CATEGORICAL, FEATURES
from split import split_indices


def save_serving(data, qmodel, qhat):
    MODELS.mkdir(exist_ok=True)
    data["model"].get_booster().save_model(MODELS / "price_model.ubj")
    qmodel.get_booster().save_model(MODELS / "quantile_model.ubj")
    meta = {
        "features": FEATURES, "categorical": CATEGORICAL, "categories": data["cats"],
        "model_vocab": sorted(data["vocab"]), "conformal_q": qhat, "ref_year": REF_YEAR,
        "selected_model": data["selected"], "best_iteration": int(data["model"].best_iteration),
        "quantile_best_iteration": int(qmodel.best_iteration),
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def register(data, metrics: dict):
    """선택한 모델을 MLflow 모델 레지스트리에 등록한다."""
    train.setup_mlflow()
    with mlflow.start_run(run_name="최종 모델 등록"):
        mlflow.set_tag("purpose", "final_model")
        mlflow.log_params({k: v for k, v in data["best_params"].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.set_tag("selected_model", data["selected"])
        mlflow.xgboost.log_model(data["model"], name="model", registered_model_name="used-vehicle-price")
        for f in ("quantile_model.ubj", "meta.json"):
            mlflow.log_artifact(str(MODELS / f), artifact_path="serving")


def main(reuse: bool):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print("1/5 정제")
    raw = pd.read_csv(RAW, low_memory=False)
    with_dups, df, funnel = clean(raw)
    idx = split_indices(len(df))

    cache = PROCESSED / "train.pkl"
    if reuse and cache.exists():
        print("2/5 학습 결과 불러오기")
        data, leak = pickle.loads(cache.read_bytes())
    else:
        print("2/5 모델 비교 실험 (MLflow 기록)")
        data = train.run(df, idx)
        print("    평가 방식 점검")
        leak = train.leakage_check(with_dups, df, set(data["parts"]["test"]["group"]))
        cache.write_bytes(pickle.dumps((data, leak)))

    print("3/5 예측 구간")
    xc, ys = data["xc"], data["ys"]
    qmodel = intervals.fit_quantile(xc, ys, data["best_params"])
    raw_lo, raw_hi = intervals.interval(qmodel, xc["test"])
    qhat = intervals.conformal_q(qmodel, xc["calib"], ys["calib"])
    lo, hi = intervals.interval(qmodel, xc["test"], qhat)
    point = data["test_point"]
    seg = analyze.segment(data["parts"]["test"]["type"]).to_numpy()
    iv = {"raw": intervals.summarize(ys["test"], raw_lo, raw_hi, point),
          "conformal": intervals.summarize(ys["test"], lo, hi, point), "qhat": qhat,
          "by_segment": {s: intervals.summarize(ys["test"][seg == s], lo[seg == s], hi[seg == s], point[seg == s])
                         for s in ["truck", "suv", "sedan", "other", "unknown"]}}

    print("4/5 오차 분석")
    test_df = data["parts"]["test"]
    best = data["model"]
    seg_res = analyze.segments(test_df, data["xs"]["test"], ys["test"], point, data["test_table"])
    truck = analyze.truck_only_model(data["parts"], xc, ys, data["best_params"], data["valid_point"])
    contrib = analyze.contributions(best, xc["test"])
    from features import build
    x_all, _ = build(df, data["vocab"])
    retention = analyze.retention_curves(df, x_all)
    anom = analyze.anomalies(test_df, data["xs"]["test"], ys["test"], point)
    ratio_all = anom.pop("ratio_all")

    save_serving(data, qmodel, qhat)
    register(data, data["test"]["selected"])

    res = {
        "overview": {"raw_rows": int(len(raw)), "clean_rows_with_dups": int(len(with_dups)), "final_rows": int(len(df)),
                     "split": {k: int(len(v)) for k, v in idx.items()}, "n_models": int(len(data["vocab"])),
                     "price_median": float(df["price"].median()), "truck_share": float(df["type"].isin({"truck", "pickup"}).mean())},
        "funnel": funnel.to_dict("records"), "duplicates": duplicate_stats(raw), "leakage": leak,
        "models": data["results"], "selected": data["selected"], "test": data["test"], "best_params": data["best_params"], "tune_seconds": data["tune_seconds"],
        "trials": data["trials"], "intervals": iv, "segments": seg_res, "truck_only": truck,
        "contributions": contrib, "retention": retention, "anomalies": anom,
    }
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    np.save(PROCESSED / "test_arrays.npy", {"y": ys["test"], "pred": point, "lo": lo, "hi": hi, "seg": seg,
                                            "ratio": ratio_all, "table": data["test_table"]}, allow_pickle=True)

    print("    SQLite 저장·SQL 품질 검사")
    import build_db
    build_db.main()

    print("5/5 그래프·리포트")
    import figures, make_readme, report
    figures.all_figs(res)
    report.build(res)
    make_readme.build(res)
    print(f"완료 ({time.time() - t0:.0f}초): {OUT / 'report.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="저장된 학습 결과를 다시 쓴다")
    main(ap.parse_args().reuse)
