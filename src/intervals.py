"""예측 구간: '이 차는 대략 얼마에서 얼마 사이'를 알려 주기 위한 80% 구간.

1) XGBoost 분위수 회귀로 10%·50%·90% 분위수를 함께 학습한다.
2) 분위수 회귀만으로는 실제 포함률이 80%에서 벗어날 수 있어, 보정용 데이터(calib)로
   Conformalized Quantile Regression(CQR) 보정을 한다. 보정용 데이터에서 구간을 벗어난 정도를 재고,
   그만큼 구간을 넓히거나 좁혀 평가 데이터에서도 약 80%를 맞추도록 한다.
"""
import numpy as np
import xgboost as xgb

from config import SEED

ALPHA = 0.2  # 80% 구간


def fit_quantile(xc: dict, ys: dict, params: dict):
    # 분위수 3개를 각각 나무로 학습하므로 점 예측보다 3배 무겁다. 학습률을 높이고 나무 수 상한을 둔다.
    p = {k: v for k, v in params.items() if k not in ("learning_rate",)}
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array([ALPHA / 2, 0.5, 1 - ALPHA / 2]),
                         n_estimators=1500, learning_rate=0.1, tree_method="hist", enable_categorical=True,
                         early_stopping_rounds=50, random_state=SEED, n_jobs=-1, **p)
    m.fit(xc["fit"], ys["fit"], eval_set=[(xc["valid"], ys["valid"])], verbose=False)
    return m


def conformal_q(model, x_calib, y_calib) -> float:
    q = model.predict(x_calib)
    scores = np.maximum(q[:, 0] - y_calib, y_calib - q[:, 2])
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(scores, level, method="higher"))


def interval(model, x, qhat: float = 0.0):
    q = model.predict(x)
    return q[:, 0] - qhat, q[:, 2] + qhat


def summarize(y_log, lo, hi, point_log) -> dict:
    inside = (y_log >= lo) & (y_log <= hi)
    width = (np.exp(hi) - np.exp(lo)) / np.exp(point_log)
    return {"coverage": float(inside.mean()), "median_width_pct": float(np.median(width))}
