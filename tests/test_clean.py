import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baselines import PriceTable  # noqa: E402
from clean import clean, group_key, load_class, normalize_model  # noqa: E402


def test_normalize_model_unifies_spellings():
    s = pd.Series(["F150 XLT", "f-150 lariat", "super duty f-250 srw", "Silverado 2500HD crew", "silverado 1500",
                   "grand cherokee laredo", "1500 crew cab big horn", "Tacoma Access Cab", None])
    assert normalize_model(s).tolist() == ["f-150", "f-150", "f-250", "silverado 2500", "silverado 1500",
                                           "grand cherokee", "1500", "tacoma", ""]


def test_load_class():
    s = pd.Series(["f-150 xlt", "silverado 2500hd", "f-350 super duty", "ram 3500", "tacoma"])
    assert load_class(s).tolist() == ["일반", "2500급", "3500급 이상", "3500급 이상", "일반"]


def _rows(**cols):
    base = {"id": [1], "VIN": [None], "manufacturer": ["ford"], "model_base": ["f-150"], "year": [2015],
            "odometer": [100000.0], "price": [20000]}
    base.update(cols)
    return pd.DataFrame(base)


def test_group_key_uses_vin_when_valid():
    d = pd.concat([_rows(VIN=["1FTFW1EF5FFA12345"]), _rows(VIN=["1ftfw1ef5ffa12345"])])
    k = group_key(d)
    assert k.iloc[0] == k.iloc[1] == "1FTFW1EF5FFA12345"


def test_round_odometer_needs_same_price():
    # 1,000 단위로 딱 떨어지는 주행거리는 가격까지 같아야 같은 차로 본다
    d = pd.concat([_rows(price=[20000]), _rows(price=[18000])])
    k = group_key(d)
    assert k.iloc[0] != k.iloc[1]
    d2 = pd.concat([_rows(odometer=[100123.0], price=[20000]), _rows(odometer=[100123.0], price=[18000])])
    k2 = group_key(d2)
    assert k2.iloc[0] == k2.iloc[1]


def test_clean_removes_placeholders_and_reposts():
    raw = pd.DataFrame({
        "id": [1, 2, 3, 4, 5], "region": ["a", "b", "a", "a", "a"], "price": [15000, 15000, 0, 9000, 8000],
        "year": [2015, 2015, 2015, 2015, 1980], "manufacturer": ["ford"] * 5, "model": ["f-150"] * 5,
        "condition": [None] * 5, "cylinders": [None] * 5, "fuel": ["gas"] * 5,
        "odometer": [80321.0, 80321.0, 50000.0, 1.0, 90000.0], "title_status": ["clean"] * 5,
        "transmission": ["automatic"] * 5, "VIN": ["1FTFW1EF5FFA12345", "1FTFW1EF5FFA12345", None, None, None],
        "drive": [None] * 5, "size": [None] * 5, "type": ["pickup"] * 5, "paint_color": [None] * 5, "state": ["tx"] * 5,
    })
    with_dups, df, funnel = clean(raw)
    assert len(with_dups) == 2           # 0달러, 주행거리 1마일(7년 된 차), 1980년식 제외
    assert len(df) == 1 and df["id"].iloc[0] == 2   # 같은 VIN은 최신 게시물 하나만
    assert funnel["rows"].iloc[-1] == 1


def test_price_table_falls_back():
    x = pd.DataFrame({"manufacturer": ["ford"] * 6 + ["kia"], "model_key": ["ford f-150"] * 6 + ["kia rio"],
                      "age": [5] * 6 + [3]})
    y = np.log(np.array([20000] * 6 + [8000]))
    t = PriceTable().fit(x, y)
    q = pd.DataFrame({"manufacturer": ["ford", "ford", "kia"], "model_key": ["ford f-150", "ford f-150", "kia rio"],
                      "age": [5, 9, 3]})
    p = np.exp(t.predict(q))
    assert np.isclose(p[0], 20000) and np.isclose(p[1], 20000)   # 같은 연식이 없으면 모델 전체 중앙값
    assert np.isclose(p[2], np.exp(np.median(y)))                 # 표본 5건 미만이면 전체 중앙값
