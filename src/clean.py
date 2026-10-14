"""원본 매물 데이터를 정제하고, 단계별로 몇 건이 왜 빠졌는지 기록한다.

정제는 두 종류로 나눈다.
1. 값 자체가 시세가 아닌 행 제거: 가격 미기재(0, 1달러), 입력 오류, 주행거리 자리표시 등
2. 같은 차가 여러 번 올라온 중복 매물 제거: 딜러가 같은 차를 여러 지역에 반복 게시한 경우
"""
import re

import numpy as np
import pandas as pd

from config import (ODO_MAX, ODO_PLACEHOLDER, PRICE_MAX, PRICE_MIN, REF_YEAR, YEAR_MAX, YEAR_MIN)

VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")  # VIN은 17자리, I·O·Q 문자를 쓰지 않는다


def normalize_model(model: pd.Series) -> pd.Series:
    """자유 입력인 모델명을 비교 가능한 기본 모델명으로 바꾼다.

    예) 'F150 XLT' → 'f-150', 'super duty f-250 srw' → 'f-250',
        'silverado 2500hd crew' → 'silverado 2500', 'grand cherokee laredo' → 'grand cherokee'
    """
    m = model.fillna("").str.lower()
    m = m.str.replace(r"[^a-z0-9\- ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    m = m.str.replace(r"^(new |used )", "", regex=True)
    m = m.str.replace(r"^super duty ", "", regex=True)
    m = m.str.replace(r"\bf ?-?(\d{3})\b", r"f-\1", regex=True)          # f150, f 150 → f-150
    toks = m.str.split(" ")

    two_word_heads = {"grand", "santa", "town", "land", "range", "model", "e", "c", "s", "rav", "monte", "el", "pt"}
    load_class_heads = {"silverado", "sierra", "ram"}

    def base(t):
        if not t or t == [""]:
            return ""
        head = t[0]
        if head in two_word_heads and len(t) > 1:
            return f"{head} {t[1]}"
        if head in load_class_heads and len(t) > 1:
            cls = re.match(r"^(1500|2500|3500)", t[1])
            if cls:
                return f"{head} {cls.group(1)}"
        return head

    return toks.map(base)


def load_class(model: pd.Series) -> pd.Series:
    """트럭의 적재 등급을 모델명에서 뽑는다 (1500급은 경형, 2500·3500급은 대형·중대형)."""
    m = model.fillna("").str.lower()
    out = pd.Series("일반", index=model.index)
    out[m.str.contains(r"\b(?:2500|250|f-?250)\b|2500hd", regex=True)] = "2500급"
    out[m.str.contains(r"\b(?:3500|350|f-?350|f-?450|4500|5500)\b|3500hd", regex=True)] = "3500급 이상"
    return out


def group_key(df: pd.DataFrame) -> pd.Series:
    """같은 차를 묶는 키. 올바른 VIN이 있으면 VIN, 없으면 제조사·모델·연식·주행거리 조합.

    주행거리가 100,000처럼 1,000 단위로 딱 떨어지면 서로 다른 차가 같은 값을 가질 수 있다.
    실제로 VIN 없는 반복 게시 묶음 중 가격까지 같은 비율이 끝자리가 있는 주행거리는 89%였지만
    1,000 단위 주행거리는 53%에 그쳤다. 그래서 이 경우에는 가격까지 같아야 같은 차로 본다.
    """
    vin = df["VIN"].fillna("").str.upper().str.strip()
    valid = vin.str.match(VIN_RE)
    odo = df["odometer"].astype("Int64")
    fp = ("FP|" + df["manufacturer"].fillna("") + "|" + df["model_base"] + "|"
          + df["year"].astype("Int64").astype(str) + "|" + odo.astype(str))
    round_odo = (odo % 1000 == 0).fillna(False).astype(bool)
    fp = fp.where(~round_odo, fp + "|" + df["price"].astype("Int64").astype(str))
    return vin.where(valid, fp)


def clean(raw: pd.DataFrame):
    """정제한 데이터(중복 포함, 중복 제거)와 단계별 기록을 돌려준다."""
    df = raw.copy()
    steps = [("원본 매물", len(df), "")]

    def keep(mask, label, reason):
        nonlocal df
        df = df[mask].copy()
        steps.append((label, len(df), reason))

    keep(df["price"].between(PRICE_MIN, PRICE_MAX),
         "가격 범위", f"{PRICE_MIN:,}달러 미만(0·1달러 등 미기재)과 {PRICE_MAX:,}달러 초과(입력 오류·수집차) 제외")
    keep(df["year"].between(YEAR_MIN, YEAR_MAX),
         "연식 범위", f"연식 미기재와 {YEAR_MIN}년 이전 클래식카 제외")
    age = REF_YEAR - df["year"]
    odo_ok = df["odometer"].between(0, ODO_MAX) & ~((df["odometer"] < ODO_PLACEHOLDER) & (age >= 2))
    keep(odo_ok, "주행거리", f"미기재, {ODO_MAX:,}마일 초과, 2년 넘은 차의 {ODO_PLACEHOLDER}마일 미만(자리표시) 제외")
    keep(df["manufacturer"].notna() & df["model"].notna(), "제조사·모델", "제조사나 모델명이 없는 매물 제외")

    df["model_base"] = normalize_model(df["model"])
    keep(df["model_base"].str.len() > 0, "모델명 해석", "모델명이 기호뿐이라 해석할 수 없는 매물 제외")

    df["group"] = group_key(df)
    with_dups = df.copy()

    # 같은 차는 가장 최근 게시물(ID가 가장 큰 것) 하나만 남긴다
    df = df.sort_values("id").drop_duplicates("group", keep="last")
    steps.append(("중복 매물", len(df), "같은 차(VIN, 없으면 제조사·모델·연식·주행거리)를 반복 게시한 매물은 최신 1건만"))

    funnel = pd.DataFrame(steps, columns=["step", "rows", "reason"])
    funnel["removed"] = (-funnel["rows"].diff()).fillna(0).astype(int)
    return with_dups.reset_index(drop=True), df.reset_index(drop=True), funnel


def duplicate_stats(raw: pd.DataFrame) -> dict:
    """원본에서 VIN 기준 반복 게시가 얼마나 되는지."""
    vin = raw["VIN"].dropna().str.upper()
    vc = vin.value_counts()
    rep = raw.dropna(subset=["VIN"]).assign(VIN=lambda d: d["VIN"].str.upper())
    multi = rep[rep["VIN"].isin(vc[vc > 1].index)].groupby("VIN")
    return {
        "vin_rows": int(len(vin)),
        "vin_unique": int(vc.size),
        "reposted_vins": int((vc > 1).sum()),
        "reposted_rows": int(vc[vc > 1].sum()),
        "max_reposts": int(vc.max()),
        "same_price_share": float((multi["price"].nunique() == 1).mean()),
        "multi_region_share": float((multi["region"].nunique() > 1).mean()),
    }
