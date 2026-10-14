"""브라우저에서 돌아가는 시세 체험 페이지용 경량 모델을 만들어 JSON으로 내보낸다.

실행: python src/export_demo.py   (먼저 python src/run_all.py로 학습 결과를 만들어 둘 것)
결과: demo/model.json  → demo/index.html이 읽어서 브라우저 안에서 예측한다 (서버 없음)

본 모델(약 60MB)은 웹페이지에 올리기에 너무 커서, 같은 데이터·같은 분리로 작은 모델을 따로 학습한다.
- 깊이 6, 나무 300개 이하. 나무 수는 조기 종료용 데이터(es)로 정한다.
- 차령·주행거리에 단조 제약을 건다: 다른 조건이 같으면 더 오래되거나 더 많이 탄 차가 더 비싸게 나오지 않게 한다.
  제약 없이 학습하면 주행거리가 정확히 100,000마일인 매물(어림값으로 적은 경우가 많다) 같은 데이터의 굴곡까지 따라가서,
  체험 화면에서 '더 많이 탔는데 더 비싸다'는 결과가 나온다. 검증 오차율은 12.98% → 13.61%로 0.6%p 나빠지지만 결과가 직관적이다.
  연평균 주행거리는 차령·주행거리와 반대 방향으로 얽혀 있어 제약을 깨뜨리므로 뺐다.
- '매물 정보 기재 수'는 뺐다. 판매자가 정보를 덜 적으면 시세가 낮게 나오는 것은 체험 페이지에서 오해를 부르기 때문이다.
- 80% 시세 구간은 본 모델과 같은 방식(분위수 회귀 + 보정용 데이터로 CQR)으로 만든다.
- 평가 데이터(test)는 경량 모델의 최종 성능을 재는 데 한 번 쓴다.
- 가격 요인은 SHAP 대신 브라우저에서 계산하기 쉬운 근사값(경로 기반 기여도, Saabas 방식)을 쓴다.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from config import PROCESSED, REF_YEAR, ROOT, SEED, TRUCK_TYPES
from features import CATEGORICAL, FEATURE_KO, NUMERIC
from intervals import ALPHA, conformal_q
from metrics import evaluate

OUT = ROOT / "demo" / "model.json"
DEMO_NUMERIC = [c for c in NUMERIC if c not in ("info_fields", "miles_per_year")]
MONOTONE = {"age": -1, "odometer": -1}   # 늘어날수록 가격이 내려가거나 그대로
DEMO_FEATURES = DEMO_NUMERIC + CATEGORICAL
LIGHT = dict(max_depth=6, learning_rate=0.1, n_estimators=300)


def train_light(xc, ys, params, quantile=False):
    p = {k: v for k, v in params.items() if k not in ("max_depth", "learning_rate")}
    extra = dict(objective="reg:quantileerror", quantile_alpha=np.array([ALPHA / 2, 0.5, 1 - ALPHA / 2])) if quantile else {}
    m = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, early_stopping_rounds=50, random_state=SEED,
                         n_jobs=-1, max_depth=LIGHT["max_depth"], learning_rate=LIGHT["learning_rate"],
                         n_estimators=LIGHT["n_estimators"], monotone_constraints=MONOTONE, **extra, **p)
    m.fit(xc["fit"], ys["fit"], eval_set=[(xc["es"], ys["es"])], verbose=False)
    return m


def compact_trees(booster: xgb.Booster, n_trees: int, n_groups: int = 1):
    """XGBoost JSON의 나무들을 브라우저에서 읽기 좋은 작은 형태로 바꾼다.

    노드마다 [왼쪽 자식, 오른쪽 자식, 변수 번호, 기준값, 결측이면 왼쪽(1/0)]을 두고,
    범주형 분기는 '오른쪽으로 가는 범주'를 비트 묶음(16진수 문자열)으로 저장한다.
    v는 잎 노드의 값, 안쪽 노드는 그 아래 잎들의 평균(가중치: hessian 합)으로 기여도 계산에 쓴다.
    """
    j = json.loads(booster.save_raw("json"))
    model = j["learner"]["gradient_booster"]["model"]
    info = model["tree_info"]
    out = []
    for t_idx, t in enumerate(model["trees"]):
        if t_idx >= n_trees * n_groups:
            break
        left, right = t["left_children"], t["right_children"]
        hess, cond = t["sum_hessian"], t["split_conditions"]
        n = len(left)
        cat_nodes = dict(zip(t["categories_nodes"], zip(t["categories_segments"], t["categories_sizes"])))

        mean = [0.0] * n
        def fill(i):
            if left[i] == -1:
                mean[i] = cond[i]
                return cond[i] * hess[i], hess[i]
            ls, lh = fill(left[i])
            rs, rh = fill(right[i])
            mean[i] = (ls + rs) / (lh + rh) if lh + rh > 0 else 0.0
            return ls + rs, lh + rh
        fill(0)

        nodes, cats = [], {}
        for i in range(n):
            if left[i] == -1:
                nodes.append([-1, -1, -1, 0, 0])
                continue
            f = t["split_indices"][i]
            if t["split_type"][i] == 1:
                seg, size = cat_nodes[i]
                members = t["categories"][seg:seg + size]
                bits = 0
                for c in members:
                    bits |= 1 << c
                cats[i] = format(bits, "x")
                thr = 0
            else:
                thr = float(np.float32(cond[i]))
            nodes.append([left[i], right[i], f, thr, int(t["default_left"][i])])
        out.append({"g": info[t_idx], "n": nodes, "c": cats, "v": [float(f"{m:.6g}") for m in mean]})
    base = j["learner"]["learner_model_param"]["base_score"].strip("[]").split(",")
    return out, [float(b) for b in base]


def predict_python(trees, base, x_codes: np.ndarray, n_groups=1, contrib=False):
    """내보낸 JSON과 같은 규칙으로 예측한다 (브라우저 예측기와 결과를 맞춰 보기 위한 검산용)."""
    x_codes = x_codes.astype(np.float32).astype(float)
    n, nf = x_codes.shape
    pred = np.tile(np.array(base, dtype=float), (n, 1))
    con = np.zeros((n, nf + 1)) if contrib else None
    for t in trees:
        nodes, cats, v, g = t["n"], {int(k): int(b, 16) for k, b in t["c"].items()}, t["v"], t["g"]
        for r in range(n):
            i = 0
            while nodes[i][0] != -1:
                l, rr, f, thr, dl = nodes[i]
                x = x_codes[r, f]   # XGBoost는 입력을 float32로 바꿔 비교한다
                if np.isnan(x):
                    nxt = l if dl else rr
                elif i in cats:
                    nxt = rr if (cats[i] >> int(x)) & 1 else l
                else:
                    nxt = l if x < thr else rr
                if contrib:
                    con[r, f] += v[nxt] - v[i]
                i = nxt
            pred[r, g] += v[i]
            if contrib:
                con[r, -1] += v[0]
    if n_groups > 1:
        pred = np.sort(pred, axis=1)   # XGBoost는 분위수 예측이 서로 엇갈리면 크기순으로 정렬해 돌려준다
    return pred, con


def codes(x: pd.DataFrame) -> np.ndarray:
    """범주형은 범주 번호로, 결측은 NaN으로 바꾼 숫자 행렬."""
    out = np.empty((len(x), len(DEMO_FEATURES)))
    for j, c in enumerate(DEMO_FEATURES):
        if c in CATEGORICAL:
            k = x[c].cat.codes.to_numpy().astype(float)
            k[k < 0] = np.nan
            out[:, j] = k
        else:
            out[:, j] = x[c].to_numpy(dtype=float)
    return out


def main():
    data, _ = pickle.loads((PROCESSED / "train.pkl").read_bytes())
    xc = {k: v[DEMO_FEATURES] for k, v in data["xc"].items()}
    ys = data["ys"]

    point = train_light(xc, ys, data["best_params"])
    quant = train_light(xc, ys, data["best_params"], quantile=True)
    n_point, n_quant = point.best_iteration + 1, quant.best_iteration + 1
    qhat = conformal_q(_Trimmed(quant, n_quant), xc["calib"], ys["calib"])

    # 검증 데이터 점수(참고)와 평가 데이터 최종 점수
    valid = evaluate(ys["valid"], point.predict(xc["valid"], iteration_range=(0, n_point)))
    p_test = point.predict(xc["test"], iteration_range=(0, n_point))
    q_test = quant.predict(xc["test"], iteration_range=(0, n_quant))
    test = evaluate(ys["test"], p_test)
    lo, hi = q_test[:, 0] - qhat, q_test[:, 2] + qhat
    coverage = float(((ys["test"] >= lo) & (ys["test"] <= hi)).mean())
    print(f"경량 모델  나무 {n_point}개 / 분위수 {n_quant}개  검증 MdAPE {valid['mdape']:.4f}  "
          f"평가 MdAPE {test['mdape']:.4f}  ±20% {test['within_20']:.4f}  80% 구간 포함률 {coverage:.4f}")

    judge = price_judgement(ys["test"], p_test, lo, hi)
    print("매물 가격 판단 (평가 데이터):", {k: round(v, 4) for k, v in judge.items() if isinstance(v, float)})

    trees_p, base_p = compact_trees(point.get_booster(), n_point)
    trees_q, base_q = compact_trees(quant.get_booster(), n_quant, n_groups=3)

    # 검산: 내보낸 나무로 계산한 값이 XGBoost 예측과 같은가
    sample = xc["test"].sample(1500, random_state=SEED)
    xs = codes(sample)
    mine, con = predict_python(trees_p, base_p, xs, contrib=True)
    ref = point.predict(sample, iteration_range=(0, n_point))
    ref_con = point.get_booster().predict(xgb.DMatrix(sample, enable_categorical=True), pred_contribs=True,
                                          approx_contribs=True, iteration_range=(0, n_point))
    mine_q, _ = predict_python(trees_q, base_q, xs, n_groups=3)
    ref_q = quant.predict(sample, iteration_range=(0, n_quant))
    err = dict(point=float(np.abs(mine[:, 0] - ref).max()), quantile=float(np.abs(mine_q - ref_q).max()),
               contrib=float(np.abs(con[:, :-1] - ref_con[:, :-1]).max()))
    print("검산 (최대 차이, log 가격):", err)
    if max(err.values()) > 1e-3:
        sys.exit("내보낸 모델이 XGBoost 예측과 다르다")

    cats = data["cats"]
    vocab = sorted(data["vocab"])
    payload = {
        "about": {"ref_year": REF_YEAR, "truck_types": sorted(TRUCK_TYPES), "n_trees": n_point, "n_quantile_trees": n_quant,
                  "max_depth": LIGHT["max_depth"], "monotone": MONOTONE, "valid": valid, "test": test, "test_coverage_80": coverage,
                  "full_model_test": data["test"]["selected"], "n_test": int(len(ys["test"])), "judge": judge},
        "features": DEMO_FEATURES, "feature_ko": {c: FEATURE_KO[c] for c in DEMO_FEATURES},
        "categories": {c: cats[c] for c in CATEGORICAL}, "model_vocab": vocab,
        "point": {"base": base_p, "trees": trees_p},
        "quantile": {"base": base_q, "trees": trees_q, "qhat": qhat},
        "check": {"x": [[None if np.isnan(v) else v for v in row] for row in xs[:200].tolist()],
                  "pred": ref[:200].tolist(), "contrib": ref_con[:200, :-1].round(6).tolist(),
                  "quantile": ref_q[:200].tolist()},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size / 1e6:.2f}MB)")


def price_judgement(y, point, lo, hi) -> dict:
    """체험 페이지의 '매물 가격 판단' 기준을 평가 데이터로 정하고 확인한다.

    - 80% 시세 구간 아래 / 안 / 위로 나눈다 (구간이 맞다면 대략 10% / 80% / 10%가 나와야 한다)
    - 예측의 0.4배 미만, 2.5배 초과는 정보 오류 의심 (리포트 7절 이상 매물 기준과 같음)
    - 구간 폭(예측 대비)이 평가 데이터 상위 20%에 들면 '판단이 불확실'로 표시한다.
      실제로 그런 매물의 오차가 더 큰지 함께 확인한다.
    """
    lo_c, hi_c = np.minimum(lo, point), np.maximum(hi, point)
    width = (np.exp(hi_c) - np.exp(lo_c)) / np.exp(point)
    wide_cut = float(np.quantile(width, 0.8))
    wide = width > wide_cut
    ape = np.abs(np.exp(point) - np.exp(y)) / np.exp(y)
    return {
        "below": float((y < lo_c).mean()), "inside": float(((y >= lo_c) & (y <= hi_c)).mean()),
        "above": float((y > hi_c).mean()),
        "suspect_low": float((y < point + np.log(0.4)).mean()), "suspect_high": float((y > point + np.log(2.5)).mean()),
        "wide_cut": wide_cut, "median_width": float(np.median(width)),
        "mdape_wide": float(np.median(ape[wide])), "mdape_narrow": float(np.median(ape[~wide])),
    }


class _Trimmed:
    """conformal_q가 model.predict(x)를 부르므로, 조기 종료 지점까지만 쓰도록 감싼다."""
    def __init__(self, m, n):
        self.m, self.n = m, n

    def predict(self, x):
        return self.m.predict(x, iteration_range=(0, self.n))


if __name__ == "__main__":
    main()
