from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parent.parent / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "meta.json").exists(), reason="모델 파일이 없음 (run_all.py 먼저 실행)")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from serve.app import app
    return TestClient(app)


def test_predict_returns_estimate_inside_range(client):
    r = client.post("/predict", json={"year": 2016, "manufacturer": "ford", "model": "F-150 XLT",
                                      "odometer": 98000, "type": "pickup", "drive": "4wd", "state": "tx"})
    assert r.status_code == 200
    body = r.json()
    lo, hi = body["range_80_usd"]
    assert lo <= body["estimate_usd"] <= hi
    assert body["model_key"] == "ford f-150" and body["known_model"]
    assert len(body["top_factors"]) == 4


def test_older_and_higher_mileage_is_cheaper(client):
    base = {"manufacturer": "toyota", "model": "camry", "type": "sedan", "state": "ca"}
    newer = client.post("/predict", json={**base, "year": 2018, "odometer": 30000}).json()["estimate_usd"]
    older = client.post("/predict", json={**base, "year": 2010, "odometer": 150000}).json()["estimate_usd"]
    assert older < newer


def test_unknown_model_still_predicts(client):
    r = client.post("/predict", json={"year": 2012, "manufacturer": "ford", "model": "unknown-xyz", "odometer": 120000,
                                      "state": "oh"})
    assert r.status_code == 200
    assert r.json()["known_model"] is False


def test_invalid_input_rejected(client):
    r = client.post("/predict", json={"year": 1970, "manufacturer": "ford", "model": "f-150", "odometer": 1000, "state": "tx"})
    assert r.status_code == 422
    r = client.post("/predict", json={"year": 2015, "manufacturer": "ford", "model": "f-150", "odometer": 1000})
    assert r.status_code == 422   # 주(state)는 필수
