"""평가 데이터 분리.

정제·중복 제거를 마친 데이터에서는 한 행이 한 대의 차다. 차 단위로 평가용(20%)을 먼저 떼어 두고,
나머지를 학습용(fit), 검증용(valid, 조기 종료·튜닝), 보정용(calib, 예측 구간 보정)으로 나눈다.
평가용은 모델 선택이 끝날 때까지 한 번도 보지 않는다.
"""
import numpy as np
import pandas as pd

from config import CALIB_SIZE, SEED, TEST_SIZE, VALID_SIZE


def split_indices(n: int, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * TEST_SIZE)
    test, rest = idx[:n_test], idx[n_test:]
    n_valid, n_calib = int(n * VALID_SIZE), int(n * CALIB_SIZE)
    valid, calib, fit = rest[:n_valid], rest[n_valid:n_valid + n_calib], rest[n_valid + n_calib:]
    return {"fit": np.sort(fit), "valid": np.sort(valid), "calib": np.sort(calib), "test": np.sort(test)}


def naive_split(n: int, seed: int = SEED):
    """중복을 지우지 않고 행 단위로 무작위 분할 (흔히 하는 방식, 비교용)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * TEST_SIZE)
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])
