"""시세 예측 API.

실행: uvicorn serve.app:app --reload   (저장소 최상위 폴더에서, 먼저 python src/run_all.py로 모델 생성)
예시: POST /predict  {"year": 2016, "manufacturer": "ford", "model": "f-150 xlt", "odometer": 98000, "type": "pickup", "state": "tx"}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from clean import normalize_model  # noqa: E402
from features import FEATURE_KO, build, to_category  # noqa: E402

MODELS = ROOT / "models"


class Listing(BaseModel):
    year: int = Field(..., ge=1995, le=2022, description="연식")
    manufacturer: str = Field(..., description="제조사 (예: ford)")
    model: str = Field(..., description="모델명 (예: f-150 xlt)")
    odometer: float = Field(..., ge=0, le=400_000, description="주행거리 (마일)")
    type: str | None = Field(None, description="차종 (pickup, truck, SUV, sedan …)")
    condition: str | None = None
    cylinders: str | None = Field(None, description="예: 8 cylinders")
    fuel: str | None = None
    title_status: str | None = None
    transmission: str | None = None
    drive: str | None = None
    size: str | None = None
    paint_color: str | None = None
    # 학습 데이터에는 주(state)가 빠진 매물이 없어서, 비워 두면 모델이 본 적 없는 입력이 된다. 그래서 필수로 받는다.
    state: str = Field(..., min_length=2, max_length=2, description="미국 주 약어 (예: tx)")


class Predictor:
    def __init__(self, models_dir: Path = MODELS):
        self.meta = json.loads((models_dir / "meta.json").read_text(encoding="utf-8"))
        self.price = xgb.Booster()
        self.price.load_model(models_dir / "price_model.ubj")
        self.quant = xgb.Booster()
        self.quant.load_model(models_dir / "quantile_model.ubj")
        self.vocab = set(self.meta["model_vocab"])

    def features(self, item: Listing) -> pd.DataFrame:
        row = item.model_dump()
        row["VIN"] = None
        df = pd.DataFrame([row])
        df["manufacturer"] = df["manufacturer"].str.lower().str.strip()
        df["state"] = df["state"].str.lower().str.strip()
        df["model_base"] = normalize_model(df["model"])
        x, _ = build(df, self.vocab)
        x, _ = to_category(x, self.meta["categories"])
        return x

    def predict(self, item: Listing) -> dict:
        x = self.features(item)
        d = xgb.DMatrix(x, enable_categorical=True)
        it = (0, self.meta["best_iteration"] + 1)
        point = float(self.price.predict(d, iteration_range=it)[0])
        q = self.quant.predict(d, iteration_range=(0, self.meta["quantile_best_iteration"] + 1))[0]
        qhat = self.meta["conformal_q"]
        lo, hi = float(q[0] - qhat), float(q[2] + qhat)
        contrib = self.price.predict(d, pred_contribs=True, iteration_range=it)[0][:-1]
        top = np.argsort(-np.abs(contrib))[:4]
        return {
            "estimate_usd": round(float(np.exp(point)), -1),
            "range_80_usd": [round(float(np.exp(min(lo, point))), -1), round(float(np.exp(max(hi, point))), -1)],
            "model_key": str(x["model_key"].iloc[0]),
            "known_model": not str(x["model_key"].iloc[0]).endswith("(기타)"),
            "top_factors": [{"feature": FEATURE_KO[self.meta["features"][i]],
                             "effect_pct": round(float(np.expm1(contrib[i]) * 100), 1)} for i in top],
        }


app = FastAPI(title="중고차 시세 예측 API", version="1.0")
_predictor: Predictor | None = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        if not (MODELS / "meta.json").exists():
            raise HTTPException(503, "모델 파일이 없습니다. 먼저 python src/run_all.py를 실행하세요.")
        _predictor = Predictor()
    return _predictor


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (MODELS / "meta.json").exists()}


@app.post("/predict")
def predict(item: Listing):
    return predictor().predict(item)
