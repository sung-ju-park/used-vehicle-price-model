"""비교할 모델들. 모두 log(가격)을 예측한다."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import SEED
from features import CATEGORICAL, NUMERIC


def ridge(alpha: float = 3.0):
    """선형 모델. 범주는 원-핫, 수치는 표준화. 차령·주행거리의 비선형성을 위해 제곱항을 더한다."""
    class _Ridge:
        def __init__(self):
            pre = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC + ["age2", "log_odo"]),
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
            ])
            self.pipe = make_pipeline(pre, Ridge(alpha=alpha))

        @staticmethod
        def _prep(x):
            x = x.copy()
            for c in CATEGORICAL:
                x[c] = x[c].astype(str)
            x["age2"] = x["age"] ** 2
            x["log_odo"] = np.log1p(x["odometer"])
            return x

        def fit(self, x, y):
            self.pipe.fit(self._prep(x), y)
            return self

        def predict(self, x):
            return self.pipe.predict(self._prep(x))

    return _Ridge()


def random_forest(n_estimators: int = 200, min_samples_leaf: int = 3, max_features: float = 0.4):
    """랜덤 포레스트. 범주는 정수 코드로, 결측은 -1로 둔다."""
    class _RF:
        def __init__(self):
            self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                            max_features=max_features, n_jobs=-1, random_state=SEED)

        def _prep(self, x, fit=False):
            cat = x[CATEGORICAL].astype(str)
            cat = self.enc.fit_transform(cat) if fit else self.enc.transform(cat)
            return np.hstack([x[NUMERIC].fillna(-1).to_numpy(), cat])

        def fit(self, x, y):
            self.rf.fit(self._prep(x, fit=True), y)
            return self

        def predict(self, x):
            return self.rf.predict(self._prep(x))

    return _RF()


def random_forest_te(n_estimators: int = 200, min_samples_leaf: int = 3, max_features: float = 0.4,
                     smoothing: float = 20.0, folds: int = 5):
    """모델·제조사·지역을 '평균 가격(log)'으로 바꿔 넣은 랜덤 포레스트.

    랜덤 포레스트는 수백 개 모델을 정수 코드로 받으면 '코드 번호가 큰 쪽/작은 쪽'으로만 나눌 수 있어
    모델 정보를 거의 쓰지 못한다. 그래서 각 모델의 평균 가격을 숫자 변수로 넣는다(타깃 인코딩).
    학습 데이터에서는 자기 가격이 섞이지 않도록 5겹 교차 방식(out-of-fold)으로 계산하고,
    표본이 적은 모델은 전체 평균 쪽으로 당긴다(smoothing).
    """
    te_cols = ["model_key", "manufacturer", "state"]

    class _RFTE:
        def __init__(self):
            self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                            max_features=max_features, n_jobs=-1, random_state=SEED)

        @staticmethod
        def _table(keys, y, glob):
            g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
            return (g["sum"] + smoothing * glob) / (g["count"] + smoothing)

        def _te_fit(self, x, y):
            self.glob = float(np.mean(y))
            out = pd.DataFrame(index=x.index)
            self.tables = {}
            fold = np.random.default_rng(SEED).integers(0, folds, len(x))
            for c in te_cols:
                keys = x[c].astype(str).to_numpy()
                col = np.empty(len(x))
                for f in range(folds):
                    tr, va = fold != f, fold == f
                    t = self._table(keys[tr], y[tr], self.glob)
                    col[va] = pd.Series(keys[va]).map(t).fillna(self.glob).to_numpy()
                out[c + "_te"] = col
                self.tables[c] = self._table(keys, y, self.glob)
            return out

        def _te_apply(self, x):
            return pd.DataFrame({c + "_te": x[c].astype(str).map(self.tables[c]).fillna(self.glob).to_numpy()
                                 for c in te_cols}, index=x.index)

        def _prep(self, x, te, fit=False):
            rest = [c for c in CATEGORICAL if c not in te_cols]
            cat = x[rest].astype(str)
            cat = self.enc.fit_transform(cat) if fit else self.enc.transform(cat)
            return np.hstack([x[NUMERIC].fillna(-1).to_numpy(), te.to_numpy(), cat])

        def fit(self, x, y):
            te = self._te_fit(x, y)
            self.rf.fit(self._prep(x, te, fit=True), y)
            return self

        def predict(self, x):
            return self.rf.predict(self._prep(x, self._te_apply(x)))

    return _RFTE()


XGB_DEFAULT = dict(learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                   reg_lambda=1.0, max_cat_to_onehot=1)


def xgboost_model(params: dict | None = None, n_estimators: int = 4000):
    """XGBoost. 범주형을 그대로 다루고(enable_categorical), 검증 데이터로 조기 종료한다."""
    p = {**XGB_DEFAULT, **(params or {})}
    return xgb.XGBRegressor(n_estimators=n_estimators, tree_method="hist", enable_categorical=True,
                            early_stopping_rounds=100, eval_metric="rmse", random_state=SEED, n_jobs=-1, **p)
