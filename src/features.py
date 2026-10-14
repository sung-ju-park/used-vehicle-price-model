"""모델 입력 변수를 만든다."""
import numpy as np
import pandas as pd

from clean import load_class
from config import MODEL_MIN_COUNT, REF_YEAR, TRUCK_TYPES

CATEGORICAL = ["manufacturer", "model_key", "type", "condition", "fuel", "title_status", "transmission",
               "drive", "size", "paint_color", "state", "load_class"]
NUMERIC = ["age", "odometer", "miles_per_year", "cylinders", "info_fields"]
FEATURES = NUMERIC + CATEGORICAL

FEATURE_KO = {
    "age": "차령", "odometer": "주행거리", "miles_per_year": "연평균 주행거리", "cylinders": "기통 수",
    "info_fields": "매물 정보 기재 수", "manufacturer": "제조사", "model_key": "모델", "type": "차종",
    "condition": "상태", "fuel": "연료", "title_status": "소유권 상태", "transmission": "변속기",
    "drive": "구동 방식", "size": "크기", "paint_color": "색상", "state": "주(지역)", "load_class": "적재 등급",
}


def segment(type_: pd.Series) -> pd.Series:
    t = type_.fillna("unknown")
    out = pd.Series("other", index=t.index)
    out[t.isin(TRUCK_TYPES)] = "truck"
    out[t == "SUV"] = "suv"
    out[t == "sedan"] = "sedan"
    out[t == "unknown"] = "unknown"
    return out


def build(df: pd.DataFrame, model_vocab: set | None = None):
    """정제된 매물로 입력 변수를 만든다. model_vocab은 학습 데이터에서 정한 모델 목록(평가 데이터에 그대로 적용)."""
    x = pd.DataFrame(index=df.index)
    x["age"] = (REF_YEAR - df["year"]).clip(lower=0).astype(float)
    x["odometer"] = df["odometer"].astype(float)
    x["miles_per_year"] = x["odometer"] / x["age"].clip(lower=0.5)
    x["cylinders"] = pd.to_numeric(df["cylinders"].str.extract(r"(\d+)")[0], errors="coerce")
    optional = ["condition", "cylinders", "drive", "size", "type", "paint_color", "VIN"]
    x["info_fields"] = df[optional].notna().sum(axis=1).astype(float)

    key = df["manufacturer"] + " " + df["model_base"]
    if model_vocab is None:
        vc = key.value_counts()
        model_vocab = set(vc[vc >= MODEL_MIN_COUNT].index)
    x["model_key"] = key.where(key.isin(model_vocab), df["manufacturer"] + " (기타)")
    for c in ["manufacturer", "type", "condition", "fuel", "title_status", "transmission", "drive", "size",
              "paint_color", "state"]:
        x[c] = df[c].fillna("unknown").astype(str)
    x["load_class"] = load_class(df["model"]).where(df["type"].isin(TRUCK_TYPES) | df["model_base"].str.contains(
        r"silverado|sierra|ram|f-2|f-3|f-4|1500|2500|3500", regex=True), "해당 없음")
    return x[FEATURES], model_vocab


def to_category(x: pd.DataFrame, categories: dict | None = None):
    """XGBoost가 범주형으로 쓰도록 dtype을 고정한다. 범주 목록은 학습 데이터 기준."""
    x = x.copy()
    if categories is None:
        categories = {c: sorted(x[c].unique().tolist()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        known = x[c].where(x[c].isin(categories[c]))   # 학습 때 없던 값은 결측으로 둔다
        x[c] = pd.Categorical(known, categories=categories[c])
    return x, categories


def target(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["price"].to_numpy(dtype=float))
