"""results.json의 숫자로 README.md를 만든다. 숫자를 손으로 옮겨 적다 틀리는 일을 막기 위함."""
import json

from config import OUT, ROOT, SEGMENT_KO

SITE = "https://sung-ju-park.github.io/projects"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


def sql_section(dq) -> str:
    if not dq:
        return ""
    errors = [c for c in dq["checks"] if c["level"] == "error"]
    warns = [c for c in dq["checks"] if c["level"] == "warn"]
    rows = "\n".join(f"| `{c['name']}` | {'오류' if c['level'] == 'error' else '경고'} | {c['desc']} | {c['count']:,} |"
                     for c in errors + warns)
    t = dq["tables"]
    n_match = sum(c["match"] for c in dq["recompute"])
    n_test = next(c["sql"] for c in dq["recompute"] if c["item"] == "평가 데이터 차량 수")
    dup = next(c["count"] for c in dq["checks"] if c["name"] == "possible_residual_duplicate")
    dup_share = f"{dup / n_test * 100:.1f}%({dup:,}대)"
    return f"""## 데이터 파이프라인 (SQL)

정제 결과를 SQLite 데이터베이스에 단계별 테이블로 저장하고, 품질 검사와 주요 숫자 재계산을 SQL로 합니다 (`python src/build_db.py`).

| 테이블 | 행 수 | 내용 |
|---|---|---|
| `raw_listings` | {t['raw_listings']:,} | 원본 매물 그대로 |
| `clean_listings` | {t['clean_listings']:,} | 정제 규칙을 통과한 매물 + 같은 차를 묶는 키 (중복 포함) |
| `vehicles` | {t['vehicles']:,} | 같은 차는 최신 게시물 하나만, 데이터 묶음(split) 표시 |
| `training_data` (뷰) | {t['vehicles']:,} | 학습·평가에 쓰는 데이터 |

- 스키마는 `sql/schema.sql`, 검사는 `sql/quality_checks.sql`, 재계산은 `sql/reports.sql`에 있습니다. 정제 기준값은 `src/config.py`에서 SQL 파라미터로 넘겨, 기준이 두 군데서 따로 놀지 않게 했습니다.
- 정제 단계별 건수, VIN 반복 게시 통계, 트럭 비율 등 **{len(dq['recompute'])}개 숫자를 SQL로 다시 계산해 pandas 결과와 {n_match}개 모두 일치**하는 것을 확인했고, `training_data` 뷰가 실제 학습에 쓴 데이터·분리와 같은지도 확인합니다.
- 오류 단계 검사가 하나라도 걸리면 실행이 실패합니다. 경고 단계는 건수만 기록합니다.

| 검사 | 단계 | 내용 | 건수 |
|---|---|---|---|
{rows}

처음 만든 VIN 누수 검사는 2건을 잡았는데, 확인해 보니 'AFFORDABLEQUALITY'처럼 광고 문구를 VIN 칸에 적은 서로 다른 차들과, 숫자 0을 알파벳 O로 잘못 적은 VIN이었습니다. 둘 다 정제 단계에서 '올바르지 않은 VIN'으로 보고 VIN 대신 제조사·모델·연식·주행거리로 묶었기 때문에 생긴 일이라, 오류 검사는 올바른 VIN 기준으로 하고 이런 경우는 경고로 따로 셉니다. 제조사·모델·연식·주행거리·가격이 모두 같은 '중복 의심' 차는 평가 데이터의 {dup_share}라 결과에 미치는 영향은 작지만, 실제 서비스라면 VIN 오타 보정(O→0 등)을 정제 규칙에 넣을 만합니다.

"""


def build(r: dict):
    dq_path = OUT / "data_quality.json"
    dq = json.loads(dq_path.read_text(encoding="utf-8")) if dq_path.exists() else None
    demo_path = ROOT / "demo" / "model.json"
    if demo_path.exists():
        r = {**r, "demo": json.loads(demo_path.read_text(encoding="utf-8"))["about"]}
    ov, m, lk, iv, sg, tr, ct, an, dup = (r[k] for k in ["overview", "models", "leakage", "intervals", "segments",
                                                         "truck_only", "contributions", "anomalies", "duplicates"])
    sel = r["selected"]
    best, table = r["test"]["selected"], r["test"]["price_table"]   # 평가 데이터 (최종 모델 + 시세표 참고)
    xgb0, xgb1 = m["XGBoost (기본 설정)"], m["XGBoost (튜닝)"]       # 검증 데이터
    rf, rfte, ridge = m["Random Forest"], m["Random Forest (평균가 인코딩)"], m["Ridge 회귀"]
    seg_m = {d["group"]: d for d in sg["segment_model"]}
    seg_t = {d["group"]: d for d in sg["segment_table"]}
    ret = r["retention"]

    def ret_at(s, a):
        d = ret[s]
        return d["retention"][d["age"].index(a)]

    order = sorted(m.items(), key=lambda kv: kv[1]["mdape"])
    model_rows = "\n".join(
        f"| {'**' + k + '**' if k == sel else k} | {pct(v['mdape'])} | {pct(v['within_20'])} | {usd(v['mae_usd'])} | {v['train_seconds']:.0f}초 |"
        for k, v in order)
    funnel_rows = "\n".join(f"| {f['step']} | {f['rows']:,} | {('-' + format(f['removed'], ',')) if f['removed'] else ''} | {f['reason']} |"
                            for f in r["funnel"])
    seg_rows = "\n".join(f"| {SEGMENT_KO[g]} | {seg_m[g]['n']:,} | {pct(seg_t[g]['mdape'])} | {pct(seg_m[g]['mdape'])} | {pct(iv['by_segment'][g]['coverage'])} |"
                         for g in ["truck", "suv", "sedan", "other", "unknown"] if g in seg_m)
    imp = ct["importance"]
    cov = [iv["by_segment"][g]["coverage"] for g in iv["by_segment"]]
    truck_gain = tr["global"]["mdape"] - tr["truck_only"]["mdape"]
    if truck_gain > 0.002:
        truck_text = f"트럭 전용 모델이 {pct(truck_gain)}p 낮아, 트럭은 따로 모델을 두는 것도 검토할 만했습니다."
        truck_decision = "트럭 전용 모델이 조금 나았다. 관리 비용과 비교해 판단할 문제로 남겼다"
    elif truck_gain < -0.002:
        truck_text = "전체 데이터로 학습한 모델이 더 나았습니다. 다른 차종에서 배운 감가 패턴이 트럭에도 도움이 되는 것으로 보여, 하나의 모델로 운영하기로 했습니다."
        truck_decision = "전체 모델이 트럭도 더 잘 맞혀, 하나의 모델로 운영하기로 했다 (검증 데이터 기준)"
    else:
        truck_text = "차이가 거의 없어, 관리가 쉬운 하나의 모델로 운영하는 편이 낫다고 판단했습니다."
        truck_decision = "트럭만으로 학습한 모델과 전체 모델의 트럭 오차율이 거의 같아, 하나의 모델로 운영하기로 했다 (검증 데이터 기준)"

    md = f"""# 중고차 시세 예측 모델

실제 중고차 매물 {ov['raw_rows']:,}건을 정제해 **{ov['final_rows']:,}대의 차로 시세 예측 모델**을 만들고, 점수가 부풀려지지 않는 평가 방식으로 시세표·선형 모델·Random Forest·XGBoost를 비교했습니다. 모든 실험은 **MLflow**에 기록했고, 예측 시세와 80% 시세 구간, 가격을 움직인 요인을 돌려주는 **FastAPI** 서비스로 마무리했습니다.

> **[시세 직접 계산해 보기]({SITE}/used-vehicle-price-demo/)** · **[분석 리포트 보기]({SITE}/used-vehicle-price-report.html)** · [프로젝트 소개]({SITE}/used-vehicle-price.html)

![모델 비교](outputs/figures/models.png)

## 한눈에 보기

| | |
|---|---|
| **문제** | 판매자가 입력한 매물 정보만으로 중고차 시세를 예측하고, 시세 범위와 근거까지 보여주기 |
| **데이터** | Craigslist 미국 중고차 매물 {ov['raw_rows']:,}건 (Kaggle 공개 데이터, 2021년 수집) → 정제 후 {ov['final_rows']:,}대, 트럭·픽업 {pct(ov['truck_share'], 0)} |
| **방법** | 정제 파이프라인, 중복 매물 제거와 누수 없는 평가 설계, 시세표 기준선, Ridge·Random Forest·XGBoost 비교, Optuna 튜닝, SHAP 해석, 분위수 회귀 + CQR 예측 구간 |
| **도구** | Python, pandas, scikit-learn, XGBoost, Optuna, **MLflow**(실험 기록·모델 레지스트리), FastAPI, pytest |
| **결과** (평가 데이터) | 오차율 중앙값 **{pct(table['mdape'])} → {pct(best['mdape'])}** (시세표 대비), 실제 가격 ±20% 적중 **{pct(best['within_20'])}**, 80% 시세 구간 포함률 **{pct(iv['conformal']['coverage'])}** |

## 평가 설계

중복을 지운 {ov['final_rows']:,}대를 차 단위로 다섯 묶음으로 나눴습니다.

| 묶음 | 대수 | 쓰임 |
|---|---|---|
| 학습 | {ov['split']['fit']:,} | 모델 학습 |
| 조기 종료 | {ov['split']['es']:,} | XGBoost 나무 수 결정 |
| 검증 | {ov['split']['valid']:,} | 모델 비교·선택, Optuna 튜닝, 트럭 전용 모델 필요 여부 판단 |
| 보정 | {ov['split']['calib']:,} | 80% 시세 구간 보정(CQR) |
| 평가 | {ov['split']['test']:,} | 최종 모델 성능 측정 (모델 선택이 끝난 뒤) |

- 모든 후보 모델은 검증 데이터로 비교했고, 검증 오차율이 가장 낮은 모델을 코드에서 자동으로 최종 모델로 고릅니다.
- 평가 데이터로는 최종 모델을 채점하고, 비교 참고용으로 시세표 기준선을 함께 쟀습니다. 시세표 점수는 선택에 쓰지 않았습니다.
- 조기 종료용 데이터를 검증용과 따로 둔 것은, 나무 수를 정한 데이터로 다시 채점하면 XGBoost만 검증 점수가 유리해지기 때문입니다.

## 핵심 결과

**1. 원본 매물의 {pct(dup['reposted_rows'] / ov['raw_rows'], 0)}는 같은 차를 반복 게시한 것이었다**
VIN이 있는 매물 {dup['vin_rows']:,}건 중 실제 차는 {dup['vin_unique']:,}대였고, 한 대가 최대 {dup['max_reposts']}번 올라왔습니다. 반복 게시된 차의 {pct(dup['multi_region_share'], 0)}는 여러 지역에, {pct(dup['same_price_share'], 0)}는 같은 가격으로 올라왔습니다. 딜러가 같은 차를 여러 지역에 올린 것입니다.

![정제 단계](outputs/figures/funnel.png)

**2. 중복을 그대로 두고 평가하면 점수가 부풀려진다**
같은 XGBoost 설정과 절차로, 중복을 남긴 채 매물 단위로 무작위로 나누면 평가 매물의 **{pct(lk['test_rows_seen_in_train'], 0)}가 학습 데이터에 이미 있던 차**였고 ±20% 적중률이 {pct(lk['naive']['within_20'])}로 나왔습니다. 중복을 지우고 차 단위로 나눈 정직한 평가에서는 {pct(lk['honest']['within_20'])}입니다. 이 프로젝트의 모든 수치는 뒤의 방식으로 쟀습니다. 이 점검은 최종 평가 데이터에 속한 차를 빼고 나머지 데이터로만 했습니다.

![평가 방식 비교](outputs/figures/leakage.png)

**3. XGBoost가 시세표보다 오차를 {pct(1 - best['mdape'] / table['mdape'], 0)} 줄였다**

검증 데이터로 비교한 결과입니다. 이 점수로 최종 모델을 골랐습니다.

| 모델 (검증 데이터) | 오차율 중앙값 | ±20% 적중 | 평균 절대 오차 | 학습 시간 |
|---|---|---|---|---|
{model_rows}

Optuna 튜닝으로 줄어든 검증 오차율은 {pct(xgb0['mdape'] - xgb1['mdape'])}p로 작았습니다. 큰 차이는 튜닝보다 데이터 정제와 모델 종류에서 나왔습니다.

최종 모델로 고른 **{sel}**을 평가 데이터 {ov['split']['test']:,}대로 한 번 채점한 결과는 오차율 중앙값 **{pct(best['mdape'])}**, ±20% 적중 **{pct(best['within_20'])}**, 평균 절대 오차 **{usd(best['mae_usd'])}**입니다. 검증 점수({pct(m[sel]['mdape'])})와 비슷해 검증 데이터에 과하게 맞춘 흔적은 보이지 않았습니다. 같은 평가 데이터에서 시세표 기준선은 {pct(table['mdape'])}였습니다.

**4. 가격을 가장 크게 움직인 것은 {imp[0]['label']}, {imp[1]['label']}, {imp[2]['label']}**

![변수 영향](outputs/figures/importance.png)

**5. 트럭·픽업은 시세표로는 오차가 컸지만, 모델이 차종 가운데 가장 큰 폭으로 줄였다**

| 차종 | 평가 매물 | 시세표 오차율 | 모델 오차율 | 80% 구간 포함률 |
|---|---|---|---|---|
{seg_rows}

위 표는 평가 데이터 기준입니다. 트럭 전용 모델이 필요한지는 검증 데이터의 트럭 {tr['n_eval']:,}대로 판단했습니다. 트럭만으로 학습한 모델의 오차율은 {pct(tr['truck_only']['mdape'], 2)}, 전체로 학습한 모델은 {pct(tr['global']['mdape'], 2)}였습니다. {truck_text} 같은 모델 안에서 비교하면 트럭·픽업은 10년차에 1~3년차 가격의 {pct(ret_at('truck', 10), 0)}를 유지해 세단({pct(ret_at('sedan', 10), 0)}), SUV({pct(ret_at('suv', 10), 0)})보다 값이 천천히 떨어졌습니다.

![연식별 가격 유지율](outputs/figures/retention.png)

**6. 80% 시세 구간을 보정해 실제로 80%를 담게 했다**
분위수 회귀만으로는 실제 가격이 구간에 든 비율이 {pct(iv['raw']['coverage'])}였습니다. 보정용 데이터로 CQR(Conformalized Quantile Regression)을 적용해 {pct(iv['conformal']['coverage'])}로 맞췄고, 차종별로도 {min(cov) * 100:.1f}~{pct(max(cov))}를 유지했습니다.

**7. 시세와 크게 다른 매물은 대부분 매물 정보의 문제로 보인다**
평가 매물 중 실제 가격이 예측의 0.4배 미만인 매물이 {an['low_n']:,}건, 2.5배를 넘는 매물이 {an['high_n']:,}건이었습니다. 1,000달러를 살짝 넘긴 2020년식 에스컬레이드(선납금이나 계약금으로 보임), 111,111달러·123,456달러 같은 자리표시 가격이 대표적입니다. 규칙만으로 걸러지지 않는 이상 가격을 모델이 검수 대상으로 짚어낼 수 있습니다.

## 분석 과정에서 내린 판단

| 상황 | 판단 |
|---|---|
| 원본 매물의 {pct(dup['reposted_rows'] / ov['raw_rows'], 0)}가 같은 차의 반복 게시였다 | VIN으로 같은 차를 묶고 최신 게시물 하나만 남겼다. VIN이 없으면 제조사·모델·연식·주행거리가 같으면 같은 차로 봤다 |
| VIN 없는 반복 게시 묶음 중, 1,000 단위로 떨어지는 주행거리는 가격까지 같은 비율이 53%뿐이었다 (끝자리 있는 주행거리는 89%) | 100,000마일처럼 둥근 주행거리는 서로 다른 차가 겹친 것으로 보고, 이때는 가격까지 같아야 같은 차로 판정했다 |
| 중복을 남기고 무작위로 나누면 ±20% 적중률이 {pct(lk['naive']['within_20'])}로 나왔다 | 평가 매물의 {pct(lk['test_rows_seen_in_train'], 0)}가 학습 데이터에 있던 차였다. 중복 제거 후 차 단위로 나눠 처음 보는 차에 대한 성능만 보고했다 |
| Random Forest가 선형 모델보다 못했다 ({pct(rf['mdape'])} 대 {pct(ridge['mdape'])}) | {ov['n_models']}개 모델을 정수 코드로 넣어 모델 정보를 거의 못 쓰고 있었다. 모델·제조사·주의 평균 가격으로 바꿔 넣는 타깃 인코딩(5겹 교차 방식)으로 {pct(rfte['mdape'])}까지 개선했다 |
| 튜닝 효과가 작았다 (검증 {pct(xgb0['mdape'])} → {pct(xgb1['mdape'])}) | 튜닝보다 정제와 평가 설계가 결과를 더 크게 좌우했다고 판단했다 |
| 모델 비교에 평가 데이터를 쓰면 선택 자체가 평가 데이터에 맞춰진다 | 비교·튜닝·운영 판단은 검증 데이터로 하고, 평가 데이터는 최종 모델을 고른 뒤 성능을 재는 데만 썼다 |
| 80% 구간이 실제로는 {pct(iv['raw']['coverage'])}만 담았다 | 분위수 회귀를 그대로 쓰지 않고 CQR로 보정해 {pct(iv['conformal']['coverage'])}로 맞췄다 |
| 트럭 전용 모델이 필요한가 | {truck_decision} |

## 시세 예측 API

모델 파일(`models/*.ubj`, `models/meta.json`)은 용량 때문에 저장소에 올리지 않았습니다. 아래 재현 방법대로 `run_all.py`를 한 번 실행하면 만들어지고, 그 뒤에 API를 띄울 수 있습니다.

```bash
uvicorn serve.app:app --reload
```

```
POST /predict
{{"year": 2016, "manufacturer": "ford", "model": "F-150 XLT", "odometer": 98000, "type": "pickup", "drive": "4wd", "state": "tx"}}

→ {{"estimate_usd": 30030, "range_80_usd": [24570, 38260], "model_key": "ford f-150",
    "top_factors": [{{"feature": "차령", "effect_pct": 50.1}}, {{"feature": "모델", "effect_pct": 39.3}}, …]}}
```

예측 시세(`estimate_usd`), 80% 시세 구간(`range_80_usd`), 가격을 가장 크게 움직인 요인 4개(`top_factors`, SHAP 값 기준)를 돌려줍니다. 처음 보는 모델명은 제조사별 '기타'로 처리하고, 범위를 벗어난 입력은 거부합니다. 학습 데이터에 주(state)가 빠진 매물이 없어 주는 필수로 받습니다.

{sql_section(dq)}## 시세 체험 페이지

**[{SITE}/used-vehicle-price-demo/]({SITE}/used-vehicle-price-demo/)**

차 정보를 넣으면 브라우저 안에서 예상 시세, 80% 시세 구간, 가격을 움직인 요인, 주행거리·연식에 따른 시세 곡선을 바로 계산합니다. 서버 없이 `demo/model.json`(gzip 약 1MB)만 읽어서 나무를 따라 내려가는 방식이라 GitHub Pages에서 그대로 돌아갑니다.

- 본 모델(약 60MB)은 웹에 올리기 무거워서, 같은 데이터·같은 분리로 깊이 6, 나무 300개 이하의 경량 모델을 따로 학습했습니다(`src/export_demo.py`).
- 경량 모델에는 **차령·주행거리 단조 제약**을 걸었습니다. 제약 없이 학습하면 '정확히 100,000마일'처럼 어림값으로 적은 매물의 굴곡까지 따라가서 "더 많이 탔는데 더 비싸다"는 결과가 나왔습니다. 검증 오차율이 0.6%p 나빠지는 대신 결과가 직관적입니다.
- 판매자에게 오해를 줄 수 있는 '매물 정보 기재 수'와, 단조 제약과 충돌하는 '연평균 주행거리'는 경량 모델에서 뺐습니다.
- 평가 데이터 기준 경량 모델 오차율 중앙값 {pct(r['demo']['test']['mdape'])}, ±20% 적중 {pct(r['demo']['test']['within_20'])}, 80% 구간 포함률 {pct(r['demo']['test_coverage_80'])} (본 모델 {pct(best['mdape'])}).
- 내보낸 나무로 계산한 값이 XGBoost 예측과 같은지 내보낼 때 한 번, 브라우저 계산기(`demo/predict.js`)가 같은 값을 내는지 `pytest`(node 사용)에서 한 번 검산합니다 (최대 차이 0.0001 미만, log 가격).

## 구조

```
used-vehicle-price-model/
├─ src/
│  ├─ run_all.py          # 전체 실행
│  ├─ clean.py            # 정제 규칙, 모델명 정규화, 중복 판정
│  ├─ features.py         # 입력 변수 (차령, 주행거리, 연평균 주행거리, 적재 등급 등)
│  ├─ split.py            # 학습·조기 종료·검증·보정·평가 분리
│  ├─ baselines.py        # 전체 중앙값, 시세표 기준선
│  ├─ models.py           # Ridge, Random Forest(정수 코드·타깃 인코딩), XGBoost
│  ├─ train.py            # 검증 데이터로 모델 비교·튜닝·선택, 최종 평가, 평가 방식 점검 (모두 MLflow 기록)
│  ├─ intervals.py        # 분위수 회귀 + CQR 보정
│  ├─ analyze.py          # 차종·가격대·연식별 오차, SHAP, 가격 유지율, 이상 매물
│  ├─ build_db.py         # SQLite 저장, SQL 품질 검사·재계산
│  ├─ export_demo.py      # 체험 페이지용 경량 모델 학습·검산·JSON 내보내기
│  ├─ figures.py, report.py, make_readme.py
│  └─ mlflow_screenshot.py
├─ serve/app.py           # FastAPI 시세 예측 API
├─ sql/                   # 스키마, 품질 검사, 재계산 쿼리
├─ demo/                  # 시세 체험 페이지 (index.html, predict.js, model.json)
├─ tests/                 # 정제 규칙·API·SQL 품질 검사·체험 페이지 계산기 테스트 (pytest)
├─ outputs/               # 리포트, 그래프, results.json
├─ Dockerfile             # API 컨테이너
└─ data/raw/              # 원본 데이터 (저장소에는 포함하지 않음)
```

## 재현 방법

```bash
pip install -r requirements.txt
# data/raw/vehicles.csv 준비 (data/raw/README.md 참고)
python src/run_all.py          # 정제 → 실험(MLflow 기록) → 분석 → 리포트 → 모델 파일 저장, 2코어 기준 약 30분
mlflow ui --backend-store-uri sqlite:///mlflow.db   # 실험 기록 보기
python src/build_db.py         # SQLite 저장과 SQL 품질 검사만 다시 (run_all.py에 포함)
python src/export_demo.py      # 체험 페이지용 경량 모델 (demo/model.json)
pytest -q                      # 테스트
```

## 한계

- 판매자가 부른 **매물 가격**이라 실제 거래 가격과 다를 수 있습니다.
- 승용차 중심 데이터이고 트럭·픽업은 {pct(ov['truck_share'], 0)}입니다. **상용 트럭 전용 데이터가 아니므로**, 트럭 분석은 픽업 위주의 하위 집단 분석입니다.
- 데이터가 2021년 한두 달치라 **시간에 따른 시세 변화**는 보지 못했습니다. 실제 서비스라면 과거로 학습해 이후 기간으로 평가하고, 성능을 주기적으로 감시해 재학습 시점을 정해야 합니다.
- 화물차 시세에 중요한 **적재 중량, 차축, 적재함 형태, 정비 이력**은 데이터에 없어 모델명에서 적재 등급만 뽑았습니다.
- **미국 데이터**라 국내 시장에 쓰려면 같은 정제·평가 방식으로 국내 매물 데이터에 다시 학습해야 합니다.
- MLflow는 실험 기록과 모델 등록에 썼습니다. API는 로컬에서 동작을 확인했으며, 클라우드 배포나 운영 중 성능 모니터링은 하지 않았습니다.

## 데이터

Austin Reese, *Used Cars Dataset* (Craigslist 미국 중고차 매물, Kaggle). 원본 데이터는 저장소에 포함하지 않았습니다.
"""
    (ROOT / "README.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build(json.loads((OUT / "results.json").read_text(encoding="utf-8")))
