"""머신러닝 없이 만들 수 있는 기준 모델. 모델이 이것보다 나아야 의미가 있다.

시세표 기준선: 같은 모델·연식 매물 가격의 중앙값. 표본이 적으면 모델 → 제조사 → 전체 순으로 넓힌다.
사람이 손으로 만드는 '모델·연식별 시세표'와 같은 방식이다.
"""
import numpy as np
import pandas as pd

MIN_N = 5


class GlobalMedian:
    def fit(self, x, y):
        self.m = float(np.median(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.m)


class PriceTable:
    def fit(self, x: pd.DataFrame, y: np.ndarray):
        d = x[["manufacturer", "model_key", "age"]].copy()
        d["y"] = y
        self.levels = []
        for keys in (["model_key", "age"], ["model_key"], ["manufacturer"]):
            g = d.groupby(keys, observed=True)["y"].agg(["median", "size"])
            self.levels.append((keys, g[g["size"] >= MIN_N]["median"]))
        self.global_ = float(np.median(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        out = pd.Series(np.nan, index=x.index)
        for keys, table in self.levels:
            idx = pd.MultiIndex.from_frame(x[keys].astype(object)) if len(keys) > 1 else x[keys[0]].astype(object)
            vals = pd.Series(table.reindex(idx).to_numpy(), index=x.index)
            out = out.fillna(vals)
        return out.fillna(self.global_).to_numpy()
