"""평가 지표. 모델은 log(가격)을 학습하지만, 평가는 달러와 비율로 한다."""
import numpy as np


def evaluate(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    y, p = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(p - y) / y
    resid = y_log - pred_log
    return {
        "mae_usd": float(np.mean(np.abs(p - y))),
        "median_ae_usd": float(np.median(np.abs(p - y))),
        "mdape": float(np.median(ape)),                    # 오차율 중앙값
        "within_10": float(np.mean(ape <= 0.10)),          # 실제 가격의 ±10% 안에 든 비율
        "within_20": float(np.mean(ape <= 0.20)),
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "r2_log": float(1 - np.sum(resid ** 2) / np.sum((y_log - y_log.mean()) ** 2)),
    }
