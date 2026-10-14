"""선택한 모델의 오차 분석과 해석."""
import numpy as np
import pandas as pd
import xgboost as xgb

from config import TRUCK_TYPES
from features import FEATURES, FEATURE_KO, segment
from metrics import evaluate
from models import xgboost_model


def by_group(y_log, pred_log, groups: pd.Series, order=None) -> list:
    rows = []
    for g in (order or sorted(groups.unique())):
        mask = (groups == g).to_numpy()
        if mask.sum() < 30:
            continue
        m = evaluate(y_log[mask], pred_log[mask])
        rows.append({"group": g, "n": int(mask.sum()), "mdape": m["mdape"], "within_20": m["within_20"],
                     "mae_usd": m["mae_usd"]})
    return rows


def price_band(price: pd.Series) -> pd.Series:
    bins = [0, 5_000, 10_000, 20_000, 40_000, np.inf]
    labels = ["5천 달러 미만", "5천~1만", "1만~2만", "2만~4만", "4만 달러 이상"]
    return pd.cut(price, bins=bins, labels=labels, right=False).astype(str)


def age_band(age: pd.Series) -> pd.Series:
    bins = [-1, 3, 7, 12, 18, 100]
    labels = ["0~3년", "4~7년", "8~12년", "13~18년", "19년 이상"]
    return pd.cut(age, bins=bins, labels=labels).astype(str)


def segments(test_df, x_test, y_log, pred_log, table_pred) -> dict:
    seg = segment(test_df["type"]).reset_index(drop=True)
    order = ["truck", "suv", "sedan", "other", "unknown"]
    return {
        "segment_model": by_group(y_log, pred_log, seg, order),
        "segment_table": by_group(y_log, table_pred, seg, order),
        "price_band": by_group(y_log, pred_log, price_band(test_df["price"]).reset_index(drop=True),
                               ["5천 달러 미만", "5천~1만", "1만~2만", "2만~4만", "4만 달러 이상"]),
        "age_band": by_group(y_log, pred_log, age_band(x_test["age"]).reset_index(drop=True),
                             ["0~3년", "4~7년", "8~12년", "13~18년", "19년 이상"]),
    }


def truck_only_model(parts, xc, ys, params, global_pred_valid) -> dict:
    """트럭·픽업만으로 학습한 모델이 전체 데이터로 학습한 모델보다 트럭을 더 잘 맞히는가.
    운영 방식을 정하는 판단이므로 검증 데이터(valid)로 비교한다."""
    def mask(k):
        return parts[k]["type"].isin(TRUCK_TYPES).to_numpy()

    mf, me, mv = mask("fit"), mask("es"), mask("valid")
    m = xgboost_model({**params, "learning_rate": 0.03}, n_estimators=8000)
    m.fit(xc["fit"][mf], ys["fit"][mf], eval_set=[(xc["es"][me], ys["es"][me])], verbose=False)
    own = evaluate(ys["valid"][mv], m.predict(xc["valid"][mv]))
    glob = evaluate(ys["valid"][mv], global_pred_valid[mv])
    return {"n_train": int(mf.sum()), "n_eval": int(mv.sum()), "eval_set": "valid", "truck_only": own, "global": glob}


def contributions(model, x_test: pd.DataFrame, n: int = 6000, seed: int = 0) -> dict:
    """XGBoost가 계산하는 SHAP 값(pred_contribs)으로 변수별 영향을 본다. 단위는 log(가격)."""
    s = x_test.sample(min(n, len(x_test)), random_state=seed)
    booster = model.get_booster()
    c = booster.predict(xgb.DMatrix(s, enable_categorical=True), pred_contribs=True,
                        iteration_range=(0, model.best_iteration + 1))
    contrib = pd.DataFrame(c[:, :-1], columns=FEATURES, index=s.index)
    imp = contrib.abs().mean().sort_values(ascending=False)

    def dependence(col, bins):
        d = pd.DataFrame({"v": s[col], "c": contrib[col]}).dropna()
        d["b"] = pd.cut(d["v"], bins=bins)
        g = d.groupby("b", observed=True).agg(x=("v", "median"), c=("c", "mean"), n=("c", "size"))
        g = g[g["n"] >= 30]
        return {"x": g["x"].round(1).tolist(), "effect_pct": (np.expm1(g["c"]) * 100).round(1).tolist()}

    return {
        "importance": [{"feature": f, "label": FEATURE_KO[f], "mean_abs": float(v)} for f, v in imp.items()],
        "age": dependence("age", np.arange(-0.5, 28.5, 1)),
        "odometer": dependence("odometer", np.arange(0, 320_001, 20_000)),
    }


def retention_curves(df: pd.DataFrame, x: pd.DataFrame, max_age: int = 15) -> dict:
    """같은 모델 안에서 비교한 연식별 가격 유지율. 차종 구성 차이를 피하려고 모델별로 먼저 비율을 낸다.

    모델별로 '차령 1~3년 가격 중앙값' 대비 '차령 a년 가격 중앙값'의 비율을 구하고,
    차종 묶음(트럭·픽업, SUV, 세단)마다 모델들의 비율 중앙값을 낸다.
    """
    d = pd.DataFrame({"model": x["model_key"], "age": x["age"], "price": df["price"].to_numpy(),
                      "seg": segment(df["type"]).to_numpy()})
    d = d[~d["model"].str.contains("(기타)", regex=False)]
    seg_of_model = d[d["seg"] != "unknown"].groupby("model")["seg"].agg(lambda s: s.value_counts().index[0])
    med = d.groupby(["model", "age"])["price"].agg(["median", "size"])
    med = med[med["size"] >= 5]["median"].unstack()
    ref = d[d["age"].between(1, 3)].groupby("model")["price"].median()
    ref_n = d[d["age"].between(1, 3)].groupby("model").size()
    ref = ref[ref_n >= 10]
    ratio = med.loc[med.index.intersection(ref.index)].div(ref, axis=0)
    out = {}
    for s in ["truck", "suv", "sedan"]:
        models = seg_of_model[seg_of_model == s].index.intersection(ratio.index)
        r = ratio.loc[models]
        ages = [a for a in range(1, max_age + 1) if a in r.columns and r[a].notna().sum() >= 8]
        out[s] = {"age": ages, "retention": [float(r[a].median()) for a in ages], "n_models": int(len(models))}
    return out


def anomalies(test_df: pd.DataFrame, x_test: pd.DataFrame, y_log, pred_log, low=0.4, high=2.5) -> dict:
    """예측 시세와 크게 다른 매물. 대부분 모델 오류라기보다 매물 정보 자체의 문제일 가능성이 있다."""
    ratio = np.exp(y_log - pred_log)
    d = test_df.reset_index(drop=True).assign(pred=np.exp(pred_log), ratio=ratio, age=x_test["age"].to_numpy())
    lo, hi = d[d["ratio"] < low], d[d["ratio"] > high]
    cols = ["year", "manufacturer", "model", "odometer", "condition", "title_status", "price", "pred", "ratio"]
    lo_ex = lo.sort_values("ratio").head(400)
    # 낮은 쪽 중 '최근 5년 차인데 5천 달러 미만'은 할부 선납금·월 납입금을 가격 칸에 쓴 광고일 가능성이 크다
    deposit_like = lo[(lo["age"] <= 5) & (lo["price"] < 5000)]
    return {
        "n_test": int(len(d)), "low_n": int(len(lo)), "high_n": int(len(hi)),
        "low_share": float(len(lo) / len(d)), "high_share": float(len(hi) / len(d)),
        "low_deposit_like_n": int(len(deposit_like)),
        "low_examples": lo_ex[cols].head(6).round({"pred": 0, "ratio": 2}).to_dict("records"),
        "high_examples": hi.sort_values("ratio", ascending=False)[cols].head(4).round({"pred": 0, "ratio": 2}).to_dict("records"),
        "ratio_all": ratio,
    }
