"""SQL 파이프라인: 작은 가짜 데이터로 DB를 만들고, 품질 검사가 정상 데이터는 통과시키고 문제는 잡아내는지 본다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_db import build, run_checks  # noqa: E402

COLS = ["id", "region", "price", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer",
        "title_status", "transmission", "VIN", "drive", "size", "type", "paint_color", "state"]


def fake_raw(n_cars=200, seed=0):
    rng = np.random.default_rng(seed)
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    rows, next_id = [], 7_300_000_000
    for i in range(n_cars):
        vin = "".join(rng.choice(list(letters), 17))
        base = {"region": "austin", "price": int(rng.integers(3_000, 40_000)), "year": float(rng.integers(2000, 2021)),
                "manufacturer": "ford", "model": "f-150 xlt", "condition": "good", "cylinders": "8 cylinders",
                "fuel": "gas", "odometer": float(rng.integers(10_000, 200_000)) + 7, "title_status": "clean",
                "transmission": "automatic", "VIN": vin, "drive": "4wd", "size": None, "type": "pickup",
                "paint_color": "white", "state": "tx"}
        for _ in range(2 if i % 5 == 0 else 1):      # 다섯 대 중 한 대는 두 번 올라온 차
            rows.append({"id": next_id, **base})
            next_id += 1
    rows.append({**rows[0], "id": next_id, "price": 0, "VIN": None})          # 가격 미기재 → 정제에서 빠져야 함
    rows.append({**rows[0], "id": next_id + 1, "year": 1980.0, "VIN": None})  # 클래식카 → 정제에서 빠져야 함
    return pd.DataFrame(rows)[COLS]


@pytest.fixture()
def db(tmp_path):
    con, checks, df, _ = build(fake_raw(), tmp_path / "t.db")
    yield con, checks, df
    con.close()


def test_clean_data_passes_all_error_checks(db):
    con, checks, df = db
    assert len(df) == 200                                        # 중복·규칙 위반을 걷어 내고 차 200대
    assert con.execute("SELECT COUNT(*) FROM clean_listings").fetchone()[0] == 240
    failed = [c["name"] for c in checks if c["level"] == "error" and c["count"]]
    assert failed == []


def test_checks_catch_injected_problems(db):
    con, _, _ = db
    # 1) 학습 데이터에 가격 0달러 매물이 섞임
    con.execute("UPDATE raw_listings SET price = 0 WHERE id = (SELECT listing_id FROM vehicles WHERE n_postings = 1 LIMIT 1)")
    # 2) 두 번 올라온 차에서 최신이 아닌 게시물을 남김
    con.execute("""UPDATE vehicles SET listing_id = (SELECT MIN(id) FROM clean_listings c WHERE c.vehicle_key = vehicles.vehicle_key)
                   WHERE n_postings = 2 AND rowid = (SELECT MIN(rowid) FROM vehicles WHERE n_postings = 2)""")
    caught = {c["name"]: c["count"] for c in run_checks(con)}
    assert caught["price_range"] == 1
    assert caught["latest_posting_kept"] == 1
