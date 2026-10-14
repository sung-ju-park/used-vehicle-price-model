"""모델 비교 실험. 모든 실행을 MLflow에 기록한다.

1) 평가 방식 점검: 중복을 지우지 않고 무작위로 나누면 점수가 얼마나 부풀려지는가
2) 기준선 2개(전체 중앙값, 시세표)와 Ridge, Random Forest, XGBoost 비교
3) Optuna로 XGBoost 하이퍼파라미터 탐색 (검증 데이터 기준, 평가 데이터는 보지 않음)
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


def leakage_check(with_dups: pd.DataFrame, data: dict) -> dict:
    """같은 차가 학습·평가에 함께 들어갔을 때의 점수와 정직한 평가 점수를 같은 모델 설정으로 비교한다."""
    tr, te = naive_split(len(with_dups))
    d_tr, d_te = with_dups.iloc[tr], with_dups.iloc[te]
    x_tr, vocab = build(d_tr)
    x_te, _ = build(d_te, vocab)
    x_tr, cats = to_category(x_tr)
    x_te, _ = to_category(x_te, cats)
    # 조기 종료용 검증 데이터도 같은 방식(행 단위 무작위)으로 뗀다
    n_val = int(len(x_tr) * 0.1)
    m = xgboost_model()
    m.fit(x_tr.iloc[n_val:], target(d_tr)[n_val:], eval_set=[(x_tr.iloc[:n_val], target(d_tr)[:n_val])], verbose=False)
    naive = evaluate(target(d_te), m.predict(x_te))
    seen = float(d_te["group"].isin(set(d_tr["group"])).mean())

    honest = data["results"]["XGBoost (기본 설정)"]
    _log_run("평가 방식 점검: 중복 포함 무작위 분할", {"split": "row_random_with_duplicates", **XGB_DEFAULT},
             naive, {"purpose": "leakage_check"})
    return {"naive": naive, "honest": honest, "test_rows_seen_in_train": seen,
            "naive_train_rows": int(len(d_tr)), "naive_test_rows": int(len(d_te))}


def run(df: pd.DataFrame, idx: dict) -> dict:
    """df: 중복 제거된 데이터, idx: split_indices 결과."""
    setup_mlflow()
    parts = {k: df.iloc[v] for k, v in idx.items()}
    x_fit_raw, vocab = build(parts["fit"])
    xs = {"fit": x_fit_raw}
    for k in ("valid", "calib", "test"):
        xs[k], _ = build(parts[k], vocab)
    ys = {k: target(parts[k]) for k in parts}
    xc = {}
    xc["fit"], cats = to_category(xs["fit"])
    for k in ("valid", "calib", "test"):
        xc[k], _ = to_category(xs[k], cats)

    results, preds, timing = {}, {}, {}

    def record(name, model_params, pred_test, seconds, tags=None):
        m = evaluate(ys["test"], pred_test)
        m["train_seconds"] = seconds
        results[name] = m
        preds[name] = pred_test
        timing[name] = seconds
        _log_run(name, model_params, m, {"purpose": "model_comparison", **(tags or {})})
        print(f"  {name:22s} MdAPE {m['mdape']:.3f}  ±20% {m['within_20']:.3f}  MAE ${m['mae_usd']:,.0f}  ({seconds:.0f}s)")

    # 기준선과 선형·RF는 fit+valid를 합쳐 학습 (조기 종료가 없으므로)
    x_fv = pd.concat([xs["fit"], xs["valid"]])
    y_fv = np.concatenate([ys["fit"], ys["valid"]])

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
        model.fit(x_fv, y_fv)
        record(name, params, model.predict(xs["test"]), time.time() - t)

    t = time.time()
    xgb_default = xgboost_model()
    xgb_default.fit(xc["fit"], ys["fit"], eval_set=[(xc["valid"], ys["valid"])], verbose=False)
    record("XGBoost (기본 설정)", {"model": "xgboost", **XGB_DEFAULT, "best_iteration": xgb_default.best_iteration},
           xgb_default.predict(xc["test"]), time.time() - t)

    # Optuna 탐색: 시간 절약을 위해 학습 데이터 일부(6만 건)와 높은 학습률로 후보를 고르고,
    # 고른 설정을 전체 학습 데이터와 낮은 학습률로 다시 학습한다. 기준은 검증 데이터 RMSE(log).
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
            m.fit(xs_sub, ys_sub, eval_set=[(xc["valid"], ys["valid"])], verbose=False)
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

    best = {**study.best_params, "learning_rate": 0.03, "max_cat_to_onehot": 1}
    t = time.time()
    xgb_tuned = xgboost_model(best, n_estimators=8000)
    xgb_tuned.fit(xc["fit"], ys["fit"], eval_set=[(xc["valid"], ys["valid"])], verbose=False)
    record("XGBoost (튜닝)", {"model": "xgboost_tuned", **best, "best_iteration": xgb_tuned.best_iteration},
           xgb_tuned.predict(xc["test"]), time.time() - t, {"selected": "true"})

    trials = study.trials_dataframe()[["number", "value"]].rename(columns={"value": "valid_rmse_log"})
    return {
        "results": results, "preds": preds, "vocab": vocab, "cats": cats, "xs": xs, "xc": xc, "ys": ys,
        "parts": parts, "models": {"xgb_default": xgb_default, "xgb_tuned": xgb_tuned},
        "best_params": best, "tune_seconds": tune_seconds, "trials": trials.to_dict("records"),
    }
