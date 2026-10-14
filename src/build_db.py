"""정제 파이프라인 결과를 SQLite 데이터베이스로 저장하고, SQL로 품질 검사와 재계산을 한다.

실행: python src/build_db.py
결과: data/processed/vehicles.db, outputs/data_quality.json

1) 원본(raw_listings) → 정제(clean_listings) → 차량 단위·데이터 분리(vehicles) 순서로 테이블에 저장한다.
   모델명 정규화는 정규식이 많아 Python(clean.py)에서 하고, 그 결과를 테이블에 넣는다.
2) sql/quality_checks.sql의 검사를 돌린다. error 단계 검사가 하나라도 걸리면 실패로 끝난다.
3) sql/reports.sql로 정제 단계별 건수, 중복 통계, 트럭 비율을 SQL로 다시 계산해 pandas 결과(results.json)와 맞춰 본다.
4) training_data 뷰의 차량과 데이터 분리가 모델 학습에 쓴 것과 같은지 확인한다.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from clean import clean
from split import split_indices

SQL = config.ROOT / "sql"
DB = config.PROCESSED / "vehicles.db"
PARAMS = {
    "price_min": config.PRICE_MIN, "price_max": config.PRICE_MAX, "year_min": config.YEAR_MIN,
    "year_max": config.YEAR_MAX, "odo_max": config.ODO_MAX, "odo_placeholder": config.ODO_PLACEHOLDER,
    "ref_year": config.REF_YEAR, "test_size": config.TEST_SIZE, "valid_size": config.VALID_SIZE,
    "calib_size": config.CALIB_SIZE, "es_size": config.ES_SIZE,
}


def named_queries(path: Path) -> list[dict]:
    """'-- name:', '-- level:', '-- desc:' 머리말로 나뉜 SQL 파일을 쿼리 목록으로 읽는다."""
    out = []
    for block in re.split(r"(?m)^-- name:\s*", path.read_text(encoding="utf-8"))[1:]:
        name, rest = block.split("\n", 1)
        meta = dict(re.findall(r"(?m)^-- (level|desc):\s*(.+)$", rest))
        sql = "\n".join(l for l in rest.splitlines() if not l.startswith("--")).strip()
        out.append({"name": name.strip(), "level": meta.get("level", "info"), "desc": meta.get("desc", ""), "sql": sql})
    return out


def load(con: sqlite3.Connection, raw: pd.DataFrame, with_dups: pd.DataFrame, df: pd.DataFrame, idx: dict):
    con.executescript((SQL / "schema.sql").read_text(encoding="utf-8"))
    r = raw.rename(columns={"VIN": "vin"}).copy()
    r["year"] = r["year"].astype("Int64")
    r.to_sql("raw_listings", con, if_exists="append", index=False, chunksize=50_000)

    clean_rows = with_dups[["id", "model_base", "group"]].rename(columns={"group": "vehicle_key"})
    clean_rows.to_sql("clean_listings", con, if_exists="append", index=False, chunksize=50_000)

    split = np.empty(len(df), dtype=object)
    for name, rows in idx.items():
        split[rows] = name
    postings = with_dups.groupby("group").size()
    vehicles = pd.DataFrame({"vehicle_key": df["group"], "listing_id": df["id"],
                             "n_postings": df["group"].map(postings).astype(int), "split": split})
    vehicles.to_sql("vehicles", con, if_exists="append", index=False, chunksize=50_000)
    con.commit()


def run_checks(con: sqlite3.Connection) -> list[dict]:
    results = []
    for q in named_queries(SQL / "quality_checks.sql"):
        n = con.execute(q["sql"], PARAMS).fetchone()[0]
        ok = n == 0 or q["level"] != "error"
        results.append({"name": q["name"], "level": q["level"], "desc": q["desc"], "count": int(n), "passed": bool(ok)})
    return results


def recompute(con: sqlite3.Connection) -> dict:
    return {q["name"]: con.execute(q["sql"], PARAMS).fetchone()[0] for q in named_queries(SQL / "reports.sql")}


def compare(sql: dict, res: dict) -> list[dict]:
    """SQL로 다시 계산한 값과 pandas 결과(results.json)를 나란히 놓는다."""
    funnel = {f["step"]: f["rows"] for f in res["funnel"]}
    pairs = [
        ("정제: 가격 범위", sql["funnel_price"], funnel["가격 범위"]),
        ("정제: 연식 범위", sql["funnel_year"], funnel["연식 범위"]),
        ("정제: 주행거리", sql["funnel_odometer"], funnel["주행거리"]),
        ("정제: 제조사·모델", sql["funnel_model"], funnel["제조사·모델"]),
        ("정제: 모델명 해석", sql["funnel_clean"], funnel["모델명 해석"]),
        ("중복 제거 후 차량 수", sql["funnel_vehicles"], funnel["중복 매물"]),
        ("VIN 있는 매물", sql["vin_rows"], res["duplicates"]["vin_rows"]),
        ("서로 다른 VIN", sql["vin_unique"], res["duplicates"]["vin_unique"]),
        ("반복 게시된 VIN", sql["reposted_vins"], res["duplicates"]["reposted_vins"]),
        ("반복 게시물 수", sql["reposted_rows"], res["duplicates"]["reposted_rows"]),
        ("최대 반복 횟수", sql["max_reposts"], res["duplicates"]["max_reposts"]),
        ("같은 가격 비율", sql["same_price_share"], res["duplicates"]["same_price_share"]),
        ("여러 지역 비율", sql["multi_region_share"], res["duplicates"]["multi_region_share"]),
        ("트럭·픽업 비율", sql["truck_share"], res["overview"]["truck_share"]),
        ("평가 데이터 차량 수", sql["test_rows"], res["overview"]["split"]["test"]),
    ]
    return [{"item": k, "sql": float(a), "pandas": float(b), "match": bool(abs(a - b) <= 1e-9 * max(1, abs(b)))}
            for k, a, b in pairs]


def same_as_training(con: sqlite3.Connection, df: pd.DataFrame, idx: dict) -> bool:
    """training_data 뷰의 (게시물 ID, 묶음)이 모델 학습에 쓴 데이터와 같은가."""
    db = pd.read_sql("SELECT id, split FROM training_data", con).sort_values("id").reset_index(drop=True)
    mine = pd.concat([pd.DataFrame({"id": df["id"].iloc[rows].to_numpy(), "split": name}) for name, rows in idx.items()])
    mine = mine.sort_values("id").reset_index(drop=True)
    return bool(len(db) == len(mine) and (db["id"].to_numpy() == mine["id"].to_numpy()).all()
                and (db["split"].to_numpy() == mine["split"].to_numpy()).all())


def connect(path: Path) -> sqlite3.Connection:
    """SQLite에는 정규식이 없어서, VIN 형식 검사용 REGEXP 함수를 등록한다."""
    con = sqlite3.connect(path)
    con.create_function("REGEXP", 2, lambda pattern, value: value is not None and re.search(pattern, value) is not None,
                        deterministic=True)
    return con


def build(raw: pd.DataFrame, path: Path = DB):
    """원본 데이터로 정제부터 DB 저장, 품질 검사까지. (연결, 검사 결과, 정제 결과)를 돌려준다."""
    with_dups, df, _ = clean(raw)
    idx = split_indices(len(df))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = connect(path)
    load(con, raw, with_dups, df, idx)
    return con, run_checks(con), df, idx


def main():
    raw = pd.read_csv(config.RAW, low_memory=False)
    con, checks, df, idx = build(raw)
    res = json.loads((config.OUT / "results.json").read_text(encoding="utf-8"))
    sql = recompute(con)
    cmp = compare(sql, res)
    same = same_as_training(con, df, idx)
    sizes = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("raw_listings", "clean_listings", "vehicles")}

    print("테이블:", sizes)
    print("\n품질 검사")
    for c in checks:
        mark = "실패" if not c["passed"] else ("경고" if c["level"] == "warn" and c["count"] else "통과")
        print(f"  [{mark}] {c['name']:28s} {c['count']:>7,}  {c['desc']}")
    print("\nSQL 재계산 vs pandas")
    for c in cmp:
        print(f"  {'일치' if c['match'] else '불일치'}  {c['item']:16s} SQL {c['sql']:>12,.4f}  pandas {c['pandas']:>12,.4f}")
    print("\ntraining_data 뷰 = 학습에 쓴 데이터:", "일치" if same else "불일치")

    out = {"tables": sizes, "checks": checks, "recompute": cmp, "training_view_matches": same}
    (config.OUT / "data_quality.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = [c["name"] for c in checks if not c["passed"]] + [c["item"] for c in cmp if not c["match"]]
    if failed or not same:
        sys.exit(f"검사 실패: {failed}")
    print(f"\n저장: {DB}, {config.OUT / 'data_quality.json'}")


if __name__ == "__main__":
    main()
