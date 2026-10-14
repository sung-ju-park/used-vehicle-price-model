"""데이터 분리.

정제·중복 제거를 마친 데이터에서는 한 행이 한 대의 차다. 차 단위로 다섯 묶음으로 나눈다.

- fit   (55%) 학습
- es    ( 5%) XGBoost 조기 종료 (나무 수 결정)
- valid (10%) 모델 비교·선택, 하이퍼파라미터 튜닝
- calib (10%) 예측 구간(CQR) 보정
- test  (20%) 평가: 모델 선택이 끝난 뒤 최종 모델 성능을 재는 데만 쓴다
"""
import numpy as np

from config import CALIB_SIZE, ES_SIZE, SEED, TEST_SIZE, VALID_SIZE


def split_indices(n: int, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    sizes = {"test": TEST_SIZE, "valid": VALID_SIZE, "calib": CALIB_SIZE, "es": ES_SIZE}
    out, start = {}, 0
    for k, frac in sizes.items():
        out[k] = np.sort(idx[start:start + int(n * frac)])
        start += int(n * frac)
    out["fit"] = np.sort(idx[start:])
    return {k: out[k] for k in ("fit", "es", "valid", "calib", "test")}


def naive_split(n: int, seed: int = SEED):
    """중복을 지우지 않고 행 단위로 무작위 분할 (흔히 하는 방식, 비교용)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * TEST_SIZE)
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])
