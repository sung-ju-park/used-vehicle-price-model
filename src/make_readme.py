"""results.json의 숫자로 README.md를 만든다. 숫자를 손으로 옮겨 적다 틀리는 일을 막기 위함."""
import json

from config import OUT, ROOT, SEGMENT_KO

SITE = "https://sung-ju-park.github.io/projects"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


def build(r: dict):
    ov, m, lk, iv, sg, tr, ct, an, dup = (r[k] for k in ["overview", "models", "leakage", "intervals", "segments",
                                                         "truck_only", "contributions", "anomalies", "duplicates"])
    best, table, xgb0 = m["XGBoost (튜닝)"], m["기준선: 시세표"], m["XGBoost (기본 설정)"]
    rf, rfte, ridge = m["Random Forest"], m["Random Forest (평균가 인코딩)"], m["Ridge 회귀"]
    seg_m = {d["group"]: d for d in sg["segment_model"]}
    seg_t = {d["group"]: d for d in sg["segment_table"]}
    ret = r["retention"]

    def ret_at(s, a):
        d = ret[s]
        return d["retention"][d["age"].index(a)]

    order = sorted(m.items(), key=lambda kv: kv[1]["mdape"])
    model_rows = "\n".join(
        f"| {'**' + k + '**' if k == 'XGBoost (튜닝)' else k} | {pct(v['mdape'])} | {pct(v['within_20'])} | {usd(v['mae_usd'])} | {v['train_seconds']:.0f}초 |"
        for k, v in order)
    funnel_rows = "\n".join(f"| {f['step']} | {f['rows']:,} | {('-' + format(f['removed'], ',')) if f['removed'] else ''} | {f['reason']} |"
                            for f in r["funnel"])
    seg_rows = "\n".join(f"| {SEGMENT_KO[g]} | {seg_m[g]['n']:,} | {pct(seg_t[g]['mdape'])} | {pct(seg_m[g]['mdape'])} | {pct(iv['by_segment'][g]['coverage'])} |"
                         for g in ["truck", "suv", "sedan", "other", "unknown"] if g in seg_m)
    imp = ct["importance"]

    md = f"""# 중고차 시세 예측 모델

실제 중고차 매물 {ov['raw_rows']:,}건을 정제해 **{ov['final_rows']:,}대의 차로 시세 예측 모델**을 만들고, 점수가 부풀려지지 않는 평가 방식으로 시세표·선형 모델·Random Forest·XGBoost를 비교했습니다. 모든 실험은 **MLflow**에 기록했고, 예측 시세와 80% 시세 구간, 가격을 움직인 요인을 돌려주는 **FastAPI** 서비스로 마무리했습니다.

> **[분석 리포트 보기]({SITE}/used-vehicle-price-report.html)** · [프로젝트 소개]({SITE}/used-vehicle-price.html)

![모델 비교](outputs/figures/models.png)

## 한눈에 보기

| | |
|---|---|
| **문제** | 판매자가 입력한 매물 정보만으로 중고차 시세를 예측하고, 시세 범위와 근거까지 보여주기 |
| **데이터** | Craigslist 미국 중고차 매물 {ov['raw_rows']:,}건 (Kaggle 공개 데이터, 2021년 수집) → 정제 후 {ov['final_rows']:,}대, 트럭·픽업 {pct(ov['truck_share'], 0)} |
| **방법** | 정제 파이프라인, 중복 매물 제거와 누수 없는 평가 설계, 시세표 기준선, Ridge·Random Forest·XGBoost 비교, Optuna 튜닝, SHAP 해석, 분위수 회귀 + CQR 예측 구간 |
| **도구** | Python, pandas, scikit-learn, XGBoost, Optuna, **MLflow**(실험 기록·모델 레지스트리), FastAPI, pytest |
| **결과** | 오차율 중앙값 **{pct(table['mdape'])} → {pct(best['mdape'])}** (시세표 대비), 실제 가격 ±20% 적중 **{pct(best['within_20'])}**, 80% 시세 구간 포함률 **{pct(iv['conformal']['coverage'])}** |

## 핵심 결과

**1. 원본 매물의 {pct(dup['reposted_rows'] / ov['raw_rows'], 0)}는 같은 차를 반복 게시한 것이었다**
VIN이 있는 매물 {dup['vin_rows']:,}건 중 실제 차는 {dup['vin_unique']:,}대였고, 한 대가 최대 {dup['max_reposts']}번 올라왔습니다. 반복 게시된 차의 {pct(dup['multi_region_share'], 0)}는 여러 지역에, {pct(dup['same_price_share'], 0)}는 같은 가격으로 올라왔습니다. 딜러가 같은 차를 여러 지역에 올린 것입니다.

![정제 단계](outputs/figures/funnel.png)

**2. 중복을 그대로 두고 평가하면 점수가 부풀려진다**
같은 XGBoost 설정으로, 중복을 남긴 채 매물 단위로 무작위로 나누면 평가 매물의 **{pct(lk['test_rows_seen_in_train'], 0)}가 학습 데이터에 이미 있던 차**였고 ±20% 적중률이 {pct(lk['naive']['within_20'])}로 나왔습니다. 중복을 지우고 차 단위로 나눈 정직한 평가에서는 {pct(lk['honest']['within_20'])}입니다. 이 프로젝트의 모든 수치는 뒤의 방식으로 쟀습니다.

![평가 방식 비교](outputs/figures/leakage.png)

**3. XGBoost가 시세표보다 오차를 3분의 1 이상 줄였다**

| 모델 | 오차율 중앙값 | ±20% 적중 | 평균 절대 오차 | 학습 시간 |
|---|---|---|---|---|
{model_rows}

Optuna 튜닝으로 줄어든 오차율은 {pct(xgb0['mdape'] - best['mdape'])}p로 작았습니다. 큰 차이는 튜닝보다 데이터 정제와 모델 종류에서 나왔습니다.

**4. 가격을 가장 크게 움직인 것은 {imp[0]['label']}, {imp[1]['label']}, {imp[2]['label']}**

![변수 영향](outputs/figures/importance.png)

**5. 트럭·픽업은 시세표로는 오차가 컸지만, 모델이 차종 가운데 가장 큰 폭으로 줄였다**

| 차종 | 평가 매물 | 시세표 오차율 | 모델 오차율 | 80% 구간 포함률 |
|---|---|---|---|---|
{seg_rows}

트럭만으로 학습한 모델(오차율 {pct(tr['truck_only']['mdape'], 2)})과 전체로 학습한 모델({pct(tr['global']['mdape'], 2)})의 차이는 거의 없어, 관리가 쉬운 하나의 모델로 운영하는 편이 낫다고 판단했습니다. 같은 모델 안에서 비교하면 트럭·픽업은 10년차에 1~3년차 가격의 {pct(ret_at('truck', 10), 0)}를 유지해 세단({pct(ret_at('sedan', 10), 0)}), SUV({pct(ret_at('suv', 10), 0)})보다 값이 천천히 떨어졌습니다.

![연식별 가격 유지율](outputs/figures/retention.png)

**6. 80% 시세 구간을 보정해 실제로 80%를 담게 했다**
분위수 회귀만으로는 실제 가격이 구간에 든 비율이 {pct(iv['raw']['coverage'])}였습니다. 보정용 데이터로 CQR(Conformalized Quantile Regression)을 적용해 {pct(iv['conformal']['coverage'])}로 맞췄고, 차종별로도 79~81%를 유지했습니다.

**7. 시세와 크게 다른 매물은 대부분 매물 정보의 문제로 보인다**
평가 매물 중 실제 가격이 예측의 0.4배 미만인 매물이 {an['low_n']:,}건, 2.5배를 넘는 매물이 {an['high_n']:,}건이었습니다. 1,000달러를 살짝 넘긴 2020년식 에스컬레이드(선납금이나 계약금으로 보임), 111,111달러·123,456달러 같은 자리표시 가격이 대표적입니다. 규칙만으로 걸러지지 않는 이상 가격을 모델이 검수 대상으로 짚어낼 수 있습니다.

## 분석 과정에서 내린 판단

| 상황 | 판단 |
|---|---|
| 원본 매물의 {pct(dup['reposted_rows'] / ov['raw_rows'], 0)}가 같은 차의 반복 게시였다 | VIN으로 같은 차를 묶고 최신 게시물 하나만 남겼다. VIN이 없으면 제조사·모델·연식·주행거리가 같으면 같은 차로 봤다 |
| VIN 없는 반복 게시 묶음 중, 1,000 단위로 떨어지는 주행거리는 가격까지 같은 비율이 53%뿐이었다 (끝자리 있는 주행거리는 89%) | 100,000마일처럼 둥근 주행거리는 서로 다른 차가 겹친 것으로 보고, 이때는 가격까지 같아야 같은 차로 판정했다 |
| 중복을 남기고 무작위로 나누면 ±20% 적중률이 {pct(lk['naive']['within_20'])}로 나왔다 | 평가 매물의 {pct(lk['test_rows_seen_in_train'], 0)}가 학습 데이터에 있던 차였다. 중복 제거 후 차 단위로 나눠 처음 보는 차에 대한 성능만 보고했다 |
| Random Forest가 선형 모델보다 못했다 ({pct(rf['mdape'])} 대 {pct(ridge['mdape'])}) | {ov['n_models']}개 모델을 정수 코드로 넣어 모델 정보를 거의 못 쓰고 있었다. 모델·제조사·주의 평균 가격으로 바꿔 넣는 타깃 인코딩(5겹 교차 방식)으로 {pct(rfte['mdape'])}까지 개선했다 |
| 튜닝 효과가 작았다 ({pct(xgb0['mdape'])} → {pct(best['mdape'])}) | 튜닝보다 정제와 평가 설계가 결과를 더 크게 좌우했다고 판단했다. 튜닝은 검증 데이터로만 했고 평가 데이터는 마지막에 한 번만 썼다 |
| 80% 구간이 실제로는 {pct(iv['raw']['coverage'])}만 담았다 | 분위수 회귀를 그대로 쓰지 않고 CQR로 보정해 {pct(iv['conformal']['coverage'])}로 맞췄다 |
| 트럭 전용 모델이 필요한가 | 트럭만으로 학습한 모델과 전체 모델의 트럭 오차율이 거의 같아, 하나의 모델로 운영하기로 했다 |

## 시세 예측 API

```bash
uvicorn serve.app:app --reload
```

```
POST /predict
{{"year": 2016, "manufacturer": "ford", "model": "F-150 XLT", "odometer": 98000, "type": "pickup", "drive": "4wd", "state": "tx"}}

→ {{"estimate_usd": 31490, "range_80_usd": [22880, 37900], "model_key": "ford f-150",
    "top_factors": [{{"feature": "차령", "effect_pct": 47.5}}, {{"feature": "모델", "effect_pct": 36.0}}, …]}}
```

예측 시세(`estimate_usd`), 80% 시세 구간(`range_80_usd`), 가격을 가장 크게 움직인 요인 4개(`top_factors`, SHAP 값 기준)를 돌려줍니다. 처음 보는 모델명은 제조사별 '기타'로 처리하고, 범위를 벗어난 입력은 거부합니다. 학습 데이터에 주(state)가 빠진 매물이 없어 주는 필수로 받습니다.

## 구조

```
used-vehicle-price-model/
├─ src/
│  ├─ run_all.py          # 전체 실행
│  ├─ clean.py            # 정제 규칙, 모델명 정규화, 중복 판정
│  ├─ features.py         # 입력 변수 (차령, 주행거리, 연평균 주행거리, 적재 등급 등)
│  ├─ split.py            # 학습·검증·보정·평가 분리
│  ├─ baselines.py        # 전체 중앙값, 시세표 기준선
│  ├─ models.py           # Ridge, Random Forest(정수 코드·타깃 인코딩), XGBoost
│  ├─ train.py            # 모델 비교, Optuna 튜닝, 평가 방식 점검 (모두 MLflow 기록)
│  ├─ intervals.py        # 분위수 회귀 + CQR 보정
│  ├─ analyze.py          # 차종·가격대·연식별 오차, SHAP, 가격 유지율, 이상 매물
│  ├─ figures.py, report.py, make_readme.py
│  └─ mlflow_screenshot.py
├─ serve/app.py           # FastAPI 시세 예측 API
├─ tests/                 # 정제 규칙·API 테스트 (pytest)
├─ outputs/               # 리포트, 그래프, results.json
├─ Dockerfile             # API 컨테이너
└─ data/raw/              # 원본 데이터 (저장소에는 포함하지 않음)
```

## 재현 방법

```bash
pip install -r requirements.txt
# data/raw/vehicles.csv 준비 (data/raw/README.md 참고)
python src/run_all.py          # 정제 → 실험(MLflow 기록) → 분석 → 리포트, 2코어 기준 약 30분
mlflow ui --backend-store-uri sqlite:///mlflow.db   # 실험 기록 보기
pytest -q                      # 테스트
```

## 한계

- 판매자가 부른 **매물 가격**이라 실제 거래 가격과 다를 수 있습니다.
- 데이터가 2021년 한두 달치라 **시간에 따른 시세 변화**는 보지 못했습니다. 실제 서비스라면 과거로 학습해 이후 기간으로 평가하고, 성능을 주기적으로 감시해 재학습 시점을 정해야 합니다.
- 화물차 시세에 중요한 **적재 중량, 차축, 적재함 형태, 정비 이력**은 데이터에 없어 모델명에서 적재 등급만 뽑았습니다.
- **미국 데이터**라 국내 시장에 쓰려면 같은 정제·평가 방식으로 국내 매물 데이터에 다시 학습해야 합니다.

## 데이터

Austin Reese, *Used Cars Dataset* (Craigslist 미국 중고차 매물, Kaggle). 원본 데이터는 저장소에 포함하지 않았습니다.
"""
    (ROOT / "README.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build(json.loads((OUT / "results.json").read_text(encoding="utf-8")))
