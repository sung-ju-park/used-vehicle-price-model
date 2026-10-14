"""모델 비교 실험. 모든 실행을 MLflow에 기록한다.

1) 평가 방식 점검: 중복을 지우지 않고 무작위로 나누면 점수가 얼마나 부풀려지는가
2) 기준선 2개(전체 중앙값, 시세표)와 Ridge, Random Forest, XGBoost 비교
3) Optuna로 XGBoost 하이퍼파라미터 탐색 (검증 데이터 기준)
4) 검증 데이터로 최종 모델을 고른 뒤, 평가 데이터로 그 모델의 성능을 한 번 잰다
"""
import os
import time
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from baselines import GlobalMedian, PriceTable
from config import EXPERIMENT, MLFLOW_ARTIFACTS, MLFLOW_URI, SEED
from features import build, target, to_category
from metrics import evaluate
from models import XGB_DEFAULT, random_forest, random_forest_te, ridge, xgboost_model
from split import naive_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS)
    mlflow.set_experiment(EXPERIMENT)


def _log_run(name: str, params: dict, metrics: dict, tags: dict | None = None, nested: bool = False):
    with mlflow.start_run(run_name=name, nested=nested) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.set_tags(tags or {})
        return run.info.run_id


def _holdout_score(d: pd.DataFrame, seed: int = SEED):
    """d를 행 단위로 무작위 8:2로 나눠 기본 XGBoost를 학습·평가한다.
    반환: (평가 점수, 평가 행 중 학습에 있던 차의 비율, 학습 행 수, 평가 행 수)"""
    tr, te = naive_split(len(d), seed)
    d_tr, d_te = d.iloc[tr], d.iloc[te]
    x_tr, vocab = build(d_tr)
    x_te, _ = build(d_te, vocab)
    x_tr, cats = to_category(x_tr)
    x_te, _ = to_category(x_te, cats)
    y_tr = target(d_tr)
    n_es = int(len(x_tr) * 0.1)   # 조기 종료용도 같은 방식(행 단위 무작위)으로 뗀다
    m = xgboost_model()
    m.fit(x_tr.iloc[n_es:], y_tr[n_es:], eval_set=[(x_tr.iloc[:n_es], y_tr[:n_es])], verbose=False)
    seen = float(d_te["group"].isin(set(d_tr["group"])).mean())
    return evaluate(target(d_te), m.predict(x_te)), seen, len(d_tr), len(d_te)


def leakage_check(with_dups: pd.DataFrame, df: pd.DataFrame, test_groups: set) -> dict:
    """중복을 남기고 행 단위로 나눈 평가와, 중복을 지우고 나눈 평가를 같은 모델·같은 절차로 비교한다.

    최종 평가 데이터(test)에 속한 차는 두 실험 모두에서 빼서, 이 점검이 평가 데이터를 쓰지 않게 한다.
    두 실험의 차이는 '중복을 지웠는가' 하나뿐이다.
    """
    dev_dups = with_dups[~with_dups["group"].isin(test_groups)].reset_index(drop=True)
    dev = df[~df["group"].isin(test_groups)].reset_index(drop=True)
    naive, seen, n_tr, n_te = _holdout_score(dev_dups)
    honest, seen_h, _, _ = _holdout_score(dev)
    _log_run("평가 방식 점검: 중복 포함 무작위 분할", {"split": "row_random_with_duplicates", **XGB_DEFAULT},
             naive, {"purpose": "leakage_check"})
    _log_run("평가 방식 점검: 중복 제거 후 분할", {"split": "vehicle_random_deduplicated", **XGB_DEFAULT},
             honest, {"purpose": "leakage_check"})
    return {"naive": naive, "honest": honest, "test_rows_seen_in_train": seen, "honest_rows_seen_in_train": seen_h,
            "naive_train_rows": int(n_tr), "naive_test_rows": int(n_te)}


def run(df: pd.DataFrame, idx: dict) -> dict:
    """df: 중복 제거된 데이터, idx: split_indices 결과.

    1) 모든 후보를 학습 데이터(fit, 조기 종료가 필요 없는 모델은 fit+es)로 학습하고 검증 데이터(valid)로 비교한다.
    2) 검증 오차율 중앙값이 가장 낮은 모델을 최종 모델로 고른다.
    3) 평가 데이터(test)는 고른 모델의 최종 성능을 재는 데 한 번 쓴다. 시세표 기준선의 평가 점수도 함께 재지만,
       '시세표보다 얼마나 나은가'를 보여 주는 참고값일 뿐 선택에는 쓰지 않는다.
    """
    setup_mlflow()
    parts = {k: df.iloc[v] for k, v in idx.items()}
    x_fit_raw, vocab = build(parts["fit"])
    xs = {"fit": x_fit_raw}
    for k in ("es", "valid", "calib", "test"):
        xs[k], _ = build(parts[k], vocab)
    ys = {k: target(parts[k]) for k in parts}
    xc = {}
    xc["fit"], cats = to_category(xs["fit"])
    for k in ("es", "valid", "calib", "test"):
        xc[k], _ = to_category(xs[k], cats)

    results, valid_preds, fitted = {}, {}, {}

    def record(name, model, model_params, pred_valid, seconds, tags=None):
        m = evaluate(ys["valid"], pred_valid)
        m["train_seconds"] = seconds
        results[name] = m
        valid_preds[name] = pred_valid
        fitted[name] = (model, model_params)
        _log_run(name, model_params, {f"valid_{k}": v for k, v in m.items()}, {"purpose": "model_comparison", **(tags or {})})
        print(f"  {name:22s} 검증 MdAPE {m['mdape']:.3f}  ±20% {m['within_20']:.3f}  MAE ${m['mae_usd']:,.0f}  ({seconds:.0f}s)")

    # 조기 종료가 없는 모델은 fit+es를 모두 학습에 쓴다 (XGBoost와 같은 데이터 범위)
    x_fe = pd.concat([xs["fit"], xs["es"]])
    y_fe = np.concatenate([ys["fit"], ys["es"]])

    for name, model, params in [
        ("기준선: 전체 중앙값", GlobalMedian(), {"model": "global_median"}),
        ("기준선: 시세표", PriceTable(), {"model": "price_table", "levels": "model+age > model > manufacturer"}),
        ("Ridge 회귀", ridge(), {"model": "ridge", "alpha": 3.0}),
        ("Random Forest", random_forest(), {"model": "random_forest", "n_estimators": 200, "min_samples_leaf": 3,
                                            "max_features": 0.4, "categorical": "ordinal_code"}),
        ("Random Forest (평균가 인코딩)", random_forest_te(),
         {"model": "random_forest", "n_estimators": 200, "min_samples_leaf": 3, "max_features": 0.4,
          "categorical": "out_of_fold_target_encoding(model,manufacturer,state)+ordinal"}),
    ]:
        t = time.time()
        model.fit(x_fe, y_fe)
        record(name, model, params, model.predict(xs["valid"]), time.time() - t)

    es_set = [(xc["es"], ys["es"])]
    t = time.time()
    xgb_default = xgboost_model()
    xgb_default.fit(xc["fit"], ys["fit"], eval_set=es_set, verbose=False)
    record("XGBoost (기본 설정)", xgb_default, {"model": "xgboost", **XGB_DEFAULT, "best_iteration": xgb_default.best_iteration},
           xgb_default.predict(xc["valid"]), time.time() - t)

    # Optuna 탐색: 시간 절약을 위해 학습 데이터 일부(6만 건)와 높은 학습률로 후보를 고르고,
    # 고른 설정을 전체 학습 데이터와 낮은 학습률로 다시 학습한다.
    # 나무 수는 조기 종료용 데이터(es)로 정하고, 후보 간 비교는 검증 데이터(valid) RMSE(log)로 한다.
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(xc["fit"]), size=min(60_000, len(xc["fit"])), replace=False)
    xs_sub, ys_sub = xc["fit"].iloc[sub], ys["fit"][sub]

    with mlflow.start_run(run_name="Optuna 탐색 (XGBoost)") as parent:
        mlflow.set_tag("purpose", "tuning")

        def objective(trial):
            params = {
                "learning_rate": 0.1,
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
                "max_cat_to_onehot": 1,
                "max_cat_threshold": trial.suggest_int("max_cat_threshold", 16, 128, log=True),
            }
            m = xgboost_model(params, n_estimators=1500)
            m.fit(xs_sub, ys_sub, eval_set=es_set, verbose=False)
            score = evaluate(ys["valid"], m.predict(xc["valid"]))
            with mlflow.start_run(run_name=f"trial {trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({"valid_rmse_log": score["rmse_log"], "valid_mdape": score["mdape"]})
            return score["rmse_log"]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        t = time.time()
        study.optimize(objective, n_trials=int(os.environ.get("N_TRIALS", 25)))
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_valid_rmse_log", study.best_value)
        tune_seconds = time.time() - t

    tuned_params = {**study.best_params, "learning_rate": 0.03, "max_cat_to_onehot": 1}
    t = time.time()
    xgb_tuned = xgboost_model(tuned_params, n_estimators=8000)
    xgb_tuned.fit(xc["fit"], ys["fit"], eval_set=es_set, verbose=False)
    record("XGBoost (튜닝)", xgb_tuned, {"model": "xgboost_tuned", **tuned_params, "best_iteration": xgb_tuned.best_iteration},
           xgb_tuned.predict(xc["valid"]), time.time() - t)

    # 2) 검증 데이터로 최종 모델 선택
    selected = min(results, key=lambda k: results[k]["mdape"])
    model, _ = fitted[selected]
    if not isinstance(model, xgb.XGBRegressor):
        raise RuntimeError(f"선택된 모델({selected})이 XGBoost가 아니다. 서빙·해석 코드는 XGBoost 기준이다.")
    best_params = {**XGB_DEFAULT} if selected == "XGBoost (기본 설정)" else tuned_params
    print(f"  → 검증 기준 선택: {selected}")

    # 3) 평가 데이터: 선택한 모델 한 번 + 시세표(참고용)
    table_model, _ = fitted["기준선: 시세표"]
    test_point = model.predict(xc["test"])
    test_table = table_model.predict(xs["test"])
    test = {"selected": evaluate(ys["test"], test_point), "price_table": evaluate(ys["test"], test_table)}
    _log_run(f"최종 평가: {selected}", {"model": selected}, {f"test_{k}": v for k, v in test["selected"].items()},
             {"purpose": "final_test", "selected": "true"})
    print(f"  평가 데이터  {selected}  MdAPE {test['selected']['mdape']:.3f}  ±20% {test['selected']['within_20']:.3f}"
          f"  (시세표 {test['price_table']['mdape']:.3f})")

    trials = study.trials_dataframe()[["number", "value"]].rename(columns={"value": "valid_rmse_log"})
    return {
        "results": results, "selected": selected, "test": test, "test_point": test_point, "test_table": test_table,
        "valid_point": valid_preds[selected], "vocab": vocab, "cats": cats, "xs": xs, "xc": xc, "ys": ys,
        "parts": parts, "model": model, "best_params": best_params, "tune_seconds": tune_seconds,
        "trials": trials.to_dict("records"),
    }
