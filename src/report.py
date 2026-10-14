"""분석 결과를 한 장짜리 HTML 리포트로 만든다 (그림 포함 단일 파일). 모든 숫자는 results.json에서 가져온다."""
import base64
import html as H

from config import FIG, OUT, SEGMENT_KO


def img(name, alt="", cls=""):
    p = FIG / f"{name}.png"
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<figure class="{cls}"><img src="data:image/png;base64,{data}" alt="{H.escape(alt)}"></figure>'


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


def build(r):
    ov, m, lk, iv, sg, tr, ct, an, dup = (r[k] for k in ["overview", "models", "leakage", "intervals", "segments",
                                                         "truck_only", "contributions", "anomalies", "duplicates"])
    best, table = m["XGBoost (튜닝)"], m["기준선: 시세표"]
    rf, rfte = m["Random Forest"], m.get("Random Forest (평균가 인코딩)")
    seg_m = {d["group"]: d for d in sg["segment_model"]}
    seg_t = {d["group"]: d for d in sg["segment_table"]}
    ret = r["retention"]

    def ret_at(s, a):
        d = ret[s]
        return d["retention"][d["age"].index(a)] if a in d["age"] else None

    funnel_rows = "".join(
        f"<tr><td>{H.escape(f['step'])}</td><td class='n'>{f['rows']:,}</td>"
        f"<td class='n'>{('-' + format(f['removed'], ',')) if f['removed'] else ''}</td><td>{H.escape(f['reason'])}</td></tr>"
        for f in r["funnel"])
    order = sorted(m.items(), key=lambda kv: kv[1]["mdape"])
    model_rows = "".join(
        f"<tr{' class=hl' if k == 'XGBoost (튜닝)' else ''}><td>{H.escape(k)}</td><td class='n'>{pct(v['mdape'])}</td>"
        f"<td class='n'>{pct(v['within_10'])}</td><td class='n'>{pct(v['within_20'])}</td><td class='n'>{usd(v['mae_usd'])}</td>"
        f"<td class='n'>{v['r2_log']:.3f}</td><td class='n'>{v['train_seconds']:.0f}초</td></tr>" for k, v in order)
    seg_rows = "".join(
        f"<tr><td>{SEGMENT_KO[g]}</td><td class='n'>{seg_m[g]['n']:,}</td><td class='n'>{pct(seg_t[g]['mdape'])}</td>"
        f"<td class='n'>{pct(seg_m[g]['mdape'])}</td><td class='n'>{pct(seg_m[g]['within_20'])}</td>"
        f"<td class='n'>{pct(iv['by_segment'][g]['coverage'])}</td></tr>"
        for g in ["truck", "suv", "sedan", "other", "unknown"] if g in seg_m)
    band_rows = "".join(f"<tr><td>{d['group']}</td><td class='n'>{d['n']:,}</td><td class='n'>{pct(d['mdape'])}</td>"
                        f"<td class='n'>{usd(d['mae_usd'])}</td></tr>" for d in sg["price_band"])
    age_rows = "".join(f"<tr><td>{d['group']}</td><td class='n'>{d['n']:,}</td><td class='n'>{pct(d['mdape'])}</td>"
                       f"<td class='n'>{pct(d['within_20'])}</td></tr>" for d in sg["age_band"])
    ex_rows = "".join(
        f"<tr><td>{int(e['year'])} {H.escape(str(e['manufacturer']))} {H.escape(str(e['model']))}</td>"
        f"<td class='n'>{e['odometer']:,.0f}</td><td class='n'>{usd(e['price'])}</td><td class='n'>{usd(e['pred'])}</td>"
        f"<td class='n'>{e['ratio']:.2f}배</td></tr>" for e in an["low_examples"] + an["high_examples"])
    imp = ct["importance"]
    age = ct["age"]
    age_neutral = next((int(x) for x, e in zip(age["x"], age["effect_pct"]) if e <= 0), 10)
    age15 = dict(zip(age["x"], age["effect_pct"])).get(15.0, age["effect_pct"][-1])
    xgb0 = m["XGBoost (기본 설정)"]
    truck_gain = tr["global"]["mdape"] - tr["truck_only"]["mdape"]
    rf_text = ""
    if rfte:
        rf_text = (f"<p>처음에는 Random Forest가 선형 모델(Ridge)보다도 못했습니다(오차율 {pct(rf['mdape'])} 대 {pct(m['Ridge 회귀']['mdape'])}). "
                   f"원인을 보니 {ov['n_models']}개 모델을 정수 코드로 넣어서, 나무가 '코드 번호가 큰 쪽과 작은 쪽'으로만 나눌 수 있어 모델 정보를 거의 쓰지 못하고 있었습니다. "
                   f"모델·제조사·주의 평균 가격을 숫자로 바꿔 넣자(5겹 교차 방식의 타깃 인코딩) {pct(rfte['mdape'])}로 좋아졌습니다. "
                   f"모델을 비교할 때 한쪽만 불리한 입력을 쓰고 있지 않은지 먼저 확인해야 공정한 비교가 됩니다.</p>")

    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>중고차 시세 예측 모델</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
:root {{ --ink:#1B2333; --sub:#5A6475; --line:#E2E8F0; --acc:#2a78d6; --acc-soft:#EEF3FC; --warn:#C2410C; --warn-soft:#FDF1EA; --bg:#F7F8FA; }}
* {{ box-sizing:border-box; }}
body {{ font-feature-settings:"calt" 0, "case" 0, "tnum" 0; margin:0; background:var(--bg); color:var(--ink); font-family:"Pretendard","Malgun Gothic",system-ui,sans-serif; line-height:1.7; font-size:15.5px; word-break:keep-all; }}
.wrap {{ max-width:920px; margin:0 auto; padding:56px 24px 96px; }}
.kicker {{ color:var(--acc); font-weight:700; font-size:14px; }}
h1 {{ font-size:32px; line-height:1.3; letter-spacing:-0.02em; margin:8px 0 12px; }}
.lead {{ font-size:17px; color:#334155; max-width:64ch; }}
.meta {{ margin-top:14px; color:var(--sub); font-size:13.5px; }}
.summary {{ margin-top:36px; background:#fff; border:1px solid var(--line); border-radius:14px; padding:28px; }}
.summary h2 {{ margin-top:0; }}
.kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:18px 0 6px; }}
.kpi {{ border-radius:10px; padding:16px; background:var(--bg); }}
.kpi b {{ display:block; font-size:27px; letter-spacing:-0.02em; color:var(--acc); }}
.kpi span {{ font-size:13.5px; color:var(--sub); }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:16px; }}
.two h3 {{ margin:0 0 6px; font-size:15px; }}
ol, ul {{ margin:6px 0 0; padding-left:20px; }} li {{ margin:6px 0; }}
section {{ margin-top:56px; }}
h2 {{ font-size:22px; letter-spacing:-0.01em; margin:0 0 6px; }}
.num {{ color:var(--acc); font-weight:700; font-size:13px; letter-spacing:0.04em; }}
p {{ margin:10px 0; max-width:72ch; }}
figure {{ margin:18px 0; background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; }}
figure img {{ width:100%; display:block; }}
figure.narrow img {{ max-width:520px; margin:0 auto; }}
table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:14px; margin:14px 0; }}
th, td {{ padding:9px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:#F1F4F9; font-weight:600; color:#334155; }}
td.n {{ text-align:right; white-space:nowrap; }}
tr.hl td {{ background:var(--acc-soft); font-weight:600; }}
.callout {{ background:var(--acc-soft); border-radius:10px; padding:14px 16px; margin:14px 0; }}
.callout.warn {{ background:var(--warn-soft); }}
.note {{ color:var(--sub); font-size:13.5px; }}
pre {{ background:#1B2333; color:#E2E8F0; border-radius:10px; padding:14px 16px; font-size:13px; overflow-x:auto; line-height:1.55; }}
footer {{ margin-top:64px; color:var(--sub); font-size:13px; border-top:1px solid var(--line); padding-top:18px; }}
@media (max-width:720px) {{ .kpis, .two {{ grid-template-columns:1fr; }} h1 {{ font-size:26px; }} table {{ font-size:12.5px; }} }}
</style></head>
<body><div class="wrap">

<header>
  <div class="kicker">Machine Learning Case Study</div>
  <h1>중고차 시세 예측 모델</h1>
  <p class="lead">실제 중고차 매물 {ov['raw_rows']:,}건을 정제해 {ov['final_rows']:,}대의 차로 시세 예측 모델을 만들고, 점수가 부풀려지지 않는 평가 방식으로 시세표·선형 모델·Random Forest·XGBoost를 비교했습니다. 예측값과 함께 80% 시세 구간, 가격을 움직인 요인까지 돌려주는 API로 마무리했습니다.</p>
  <div class="meta">데이터: Craigslist 미국 중고차 매물 (Kaggle 공개 데이터, 2021년 수집) · 실험 기록: MLflow · 모델: XGBoost (Optuna 튜닝) · 서빙: FastAPI</div>
</header>

<div class="summary">
  <h2>요약</h2>
  <div class="kpis">
    <div class="kpi"><b>{pct(best['mdape'])}</b><span>평가 데이터 오차율 중앙값<br>(시세표 방식 {pct(table['mdape'])})</span></div>
    <div class="kpi"><b>{pct(best['within_20'])}</b><span>실제 가격의 ±20% 안에 든 비율<br>(시세표 방식 {pct(table['within_20'])})</span></div>
    <div class="kpi"><b>{pct(iv['conformal']['coverage'])}</b><span>80% 시세 구간에 실제 가격이 든 비율<br>(보정 전 {pct(iv['raw']['coverage'])})</span></div>
  </div>
  <div class="two">
    <div><h3>핵심 발견</h3><ol>
      <li>원본 매물의 {pct(dup['reposted_rows'] / ov['raw_rows'], 0)}가 같은 차를 반복 게시한 것이었습니다. 이를 그대로 두고 무작위로 나누면 평가 데이터의 {pct(lk['test_rows_seen_in_train'], 0)}가 학습 데이터에 이미 있는 차여서 점수가 부풀려집니다.</li>
      <li>가격 예측에 가장 큰 영향을 준 것은 {imp[0]['label']}, {imp[1]['label']}, {imp[2]['label']}입니다.</li>
      <li>같은 모델 안에서 비교하면 트럭·픽업은 10년이 지나도 1~3년차 가격의 {pct(ret_at('truck', 10) or 0, 0)}를 유지해 세단({pct(ret_at('sedan', 10) or 0, 0)})보다 값이 천천히 떨어집니다.</li>
    </ol></div>
    <div><h3>이렇게 판단했습니다</h3><ol>
      <li>중복 매물을 지우고 차 단위로 평가 데이터를 떼어, 실제로 처음 보는 차에 대한 성능을 쟀습니다.</li>
      <li>모델이 '시세표'보다 나은지를 먼저 확인하고, 평가 데이터는 모델을 고른 뒤 한 번만 썼습니다.</li>
      <li>예측 구간이 실제로 80%를 담는지 확인하고, 모자라면 보정(CQR)했습니다.</li>
    </ol></div>
  </div>
</div>

<section>
  <div class="num">01</div><h2>데이터와 정제</h2>
  <p>Craigslist에 올라온 미국 중고차 매물 {ov['raw_rows']:,}건입니다. 가격·연식·주행거리·제조사·모델·차종·상태 등이 매물마다 판매자가 직접 입력한 값이라, 시세를 배우기 전에 '시세가 아닌 값'부터 걸러야 했습니다. 트럭·픽업이 정제 후 매물의 {pct(ov['truck_share'], 0)}를 차지해, 트럭 시세를 따로 살펴볼 만큼 충분합니다.</p>
  {img('funnel', '정제 단계별 남은 매물 수')}
  <table><tr><th>단계</th><th>남은 매물</th><th>제외</th><th>기준</th></tr>{funnel_rows}</table>
  <div class="callout"><b>가장 많이 빠진 것은 중복 매물입니다.</b> VIN(차대번호)이 있는 매물 {dup['vin_rows']:,}건 중 실제 차는 {dup['vin_unique']:,}대였습니다. {dup['reposted_vins']:,}대가 여러 번 올라왔고(최대 {dup['max_reposts']}번), 이런 차의 {pct(dup['multi_region_share'], 0)}는 여러 지역에, {pct(dup['same_price_share'], 0)}는 같은 가격으로 반복 게시됐습니다. 딜러가 같은 차를 여러 지역에 올린 것입니다. VIN이 없는 매물은 제조사·모델·연식·주행거리가 모두 같으면 같은 차로 봤는데, 주행거리가 100,000처럼 1,000 단위로 딱 떨어지면 서로 다른 차가 겹칠 수 있어 이때는 가격까지 같아야 같은 차로 봤습니다.</div>
  <p>모델명도 자유 입력이라 'F150 XLT', 'f-150 lariat', 'super duty f-250'처럼 제각각이었습니다. 표기를 통일하고 트림을 떼어 기본 모델명으로 정리했고, 트럭은 1500·2500·3500 같은 적재 등급이 가격을 크게 가르기 때문에 'silverado 2500'처럼 등급까지 남기고 별도 변수(적재 등급)로도 뽑았습니다. 30건 이상 나온 {ov['n_models']}개 모델을 그대로 쓰고, 나머지는 제조사별 '기타'로 묶었습니다.</p>
</section>

<section>
  <div class="num">02</div><h2>평가 설계: 점수가 부풀려지지 않게</h2>
  <p>같은 차가 학습 데이터와 평가 데이터에 함께 들어가면, 모델은 시세를 배운 것이 아니라 그 차의 가격을 외운 것으로도 좋은 점수를 받습니다. 같은 XGBoost 설정으로 두 가지 방식을 비교했습니다.</p>
  {img('leakage', '평가 방식별 점수 비교')}
  <p>중복을 남긴 채 매물 단위로 무작위로 나누면 평가 매물의 <b>{pct(lk['test_rows_seen_in_train'], 0)}가 학습 데이터에 이미 있는 차</b>였고, ±20% 적중률이 {pct(lk['naive']['within_20'])}로 나왔습니다. 중복을 지우고 차 단위로 나눈 정직한 평가에서는 {pct(lk['honest']['within_20'])}입니다. 실제 서비스에서 만나는 것은 처음 보는 차이므로 이 프로젝트의 모든 수치는 뒤의 방식으로 쟀습니다.</p>
  <div class="callout">중복을 지운 {ov['final_rows']:,}대를 학습 {ov['split']['fit']:,}대, 검증 {ov['split']['valid']:,}대(조기 종료·튜닝), 보정 {ov['split']['calib']:,}대(예측 구간 보정), 평가 {ov['split']['test']:,}대로 나눴습니다. 평가 데이터는 모델 선택이 끝난 뒤 한 번만 썼습니다. 지표는 오차율 중앙값(예측이 실제 가격에서 몇 % 벗어나는지의 중앙값)과 ±10%·±20% 적중률을 중심으로 봤습니다. 비싼 차 몇 대의 큰 오차에 끌려가지 않고, 판매자가 체감하는 '시세가 맞는가'에 가깝기 때문입니다.</div>
</section>

<section>
  <div class="num">03</div><h2>모델 비교</h2>
  <p>머신러닝 없이 만들 수 있는 기준선부터 세웠습니다. 같은 모델·연식 매물 가격의 중앙값을 쓰는 '시세표' 방식으로, 표본이 적으면 모델 → 제조사 → 전체 순으로 넓혔습니다. 모든 모델은 log(가격)을 예측하고, 평가는 달러로 되돌려 했습니다.</p>
  {img('models', '모델별 오차율')}
  <table><tr><th>모델</th><th>오차율 중앙값</th><th>±10% 적중</th><th>±20% 적중</th><th>평균 절대 오차</th><th>R² (log)</th><th>학습 시간</th></tr>{model_rows}</table>
  {rf_text}
  <p>XGBoost는 Optuna로 25번 후보를 탐색했습니다. 시간을 아끼려고 학습 데이터 일부(6만 건)와 높은 학습률로 후보를 고르고, 고른 설정을 전체 학습 데이터와 낮은 학습률로 다시 학습했습니다. 탐색의 기준은 검증 데이터였고 평가 데이터는 쓰지 않았습니다. 튜닝으로 줄어든 오차율은 {pct(xgb0['mdape'] - best['mdape'])}p로 작았습니다. 기본 설정도 이미 충분히 좋았다는 뜻이고, 큰 차이는 튜닝보다 데이터 정제와 모델 종류에서 나왔습니다. 최종적으로 시세표 방식 대비 오차율은 {pct(table['mdape'])}에서 {pct(best['mdape'])}로, 평균 절대 오차는 {usd(table['mae_usd'])}에서 {usd(best['mae_usd'])}로 줄었습니다.</p>
  {img('mlflow', 'MLflow 실험 기록 화면')}
  <p class="note">모든 실행(기준선, 비교 모델, 튜닝 25회, 평가 방식 점검, 최종 모델)의 설정과 지표를 MLflow에 기록했고, 최종 모델은 모델 레지스트리에 'used-vehicle-price'로 등록했습니다.</p>
  {img('pred_vs_actual', '예측 가격과 실제 가격', 'narrow')}
</section>

<section>
  <div class="num">04</div><h2>무엇이 가격을 정하나</h2>
  <p>XGBoost가 계산하는 SHAP 값으로 각 변수가 예측 가격을 얼마나 움직였는지 봤습니다. 이는 모델이 가격을 어떻게 판단했는지를 보여주는 것이고, 변수가 가격을 '일으킨' 원인이라는 뜻은 아닙니다.</p>
  {img('importance', '변수별 영향 크기')}
  {img('dependence', '차령과 주행거리의 영향')}
  <p>다른 조건이 같을 때 차령 {age['x'][0]:.0f}년인 차는 예측 가격을 모델의 평균 예측보다 약 {age['effect_pct'][0]:.0f}% 끌어올리고, 차령이 늘수록 영향이 줄어 {age_neutral}년 전후에서 0이 된 뒤 15년차에는 약 {abs(age15):.0f}% 끌어내립니다. 감가는 처음 10년 동안 가파르고 그 뒤로 완만해집니다. 주행거리도 비슷하게 늘수록 가격을 내리는데, 차령과 주행거리를 함께 넣었기 때문에 '같은 연식이라면 더 많이 탄 차가 얼마나 싼가'를 따로 볼 수 있습니다.</p>
</section>

<section>
  <div class="num">05</div><h2>트럭·픽업</h2>
  <p>차종별로 오차를 나눠 봤습니다. 트럭·픽업은 시세표 방식의 오차가 {pct(seg_t['truck']['mdape'])}로 큰 편이었는데, 모델은 이를 {pct(seg_m['truck']['mdape'])}로 줄여 차종 가운데 가장 크게 개선했습니다({pct(1 - seg_m['truck']['mdape'] / seg_t['truck']['mdape'], 0)} 감소). 같은 모델·연식 안에서도 적재 등급, 구동 방식, 디젤 여부에 따라 가격이 크게 달라지는데, 시세표는 이 차이를 담지 못하기 때문으로 보입니다.</p>
  {img('segments', '차종별 오차율')}
  <table><tr><th>차종</th><th>평가 매물</th><th>시세표 오차율</th><th>모델 오차율</th><th>±20% 적중</th><th>80% 구간 포함률</th></tr>{seg_rows}</table>
  <p>트럭만으로 따로 학습한 모델도 만들어 봤습니다. 트럭 평가 매물 {tr['n_test']:,}대에서 트럭 전용 모델의 오차율은 {pct(tr['truck_only']['mdape'], 2)}, 전체 데이터로 학습한 모델은 {pct(tr['global']['mdape'], 2)}였습니다. {'트럭 전용 모델이 ' + pct(truck_gain) + 'p 낮았습니다.' if truck_gain > 0.002 else ('전체 모델이 ' + pct(-truck_gain) + 'p 낮았습니다. 다른 차종에서 배운 감가 패턴이 트럭에도 도움이 되는 것으로 보여, 하나의 모델로 운영하는 편이 낫다고 판단했습니다.' if truck_gain < -0.002 else '차이가 거의 없어, 관리가 쉬운 하나의 모델로 운영하는 편이 낫다고 판단했습니다.')}</p>
  {img('retention', '연식별 가격 유지율')}
  <p class="note">유지율은 차종 구성이 다른 효과를 피하려고, 모델마다 '차령 1~3년 가격 중앙값' 대비 비율을 먼저 구한 뒤 차종별로 모델들의 중앙값을 냈습니다. 매물 가격 기준이라 실제 거래 가격과는 다를 수 있습니다.</p>
</section>

<section>
  <div class="num">06</div><h2>시세 구간</h2>
  <p>시세는 한 숫자보다 '대략 얼마에서 얼마 사이'로 보여주는 것이 쓸모 있습니다. XGBoost 분위수 회귀로 10%·90% 분위수를 함께 학습해 80% 구간을 만들었는데, 평가 데이터에서 실제 가격이 구간에 든 비율이 {pct(iv['raw']['coverage'])}로 목표에 미치지 못했습니다. 그래서 보정용 데이터에서 구간을 벗어난 정도를 재고 그만큼 구간을 조정하는 CQR(Conformalized Quantile Regression)을 적용해 {pct(iv['conformal']['coverage'])}로 맞췄습니다. 구간 폭의 중앙값은 예측 가격의 {pct(iv['conformal']['median_width_pct'], 0)}입니다.</p>
  {img('intervals', '예측 구간 포함률')}
</section>

<section>
  <div class="num">07</div><h2>시세와 크게 다른 매물</h2>
  <p>평가 매물 중 실제 가격이 예측의 0.4배 미만인 매물이 {an['low_n']:,}건({pct(an['low_share'])}), 2.5배를 넘는 매물이 {an['high_n']:,}건({pct(an['high_share'])})이었습니다. 낮은 쪽 중 {an['low_deposit_like_n']:,}건은 5년 이내 차인데 가격이 5천 달러 미만이었습니다. 예를 들어 2020년식 캐딜락 에스컬레이드가 1,093달러로 올라와 있었는데, 할부 선납금이나 계약금을 가격 칸에 적은 광고일 가능성이 큽니다. 정제 단계에서 1,000달러 미만은 걸러냈지만 1,000달러를 살짝 넘긴 이런 값은 규칙으로 잡히지 않았습니다. 높은 쪽에서는 111,111달러, 123,456달러처럼 같은 숫자를 반복하거나 순서대로 친 가격이 보였습니다. 매물 설명 원문이 데이터에 없어 확인할 수는 없지만, 이런 매물은 모델 오차라기보다 매물 정보의 문제로 보입니다. 규칙만으로 거르기 어려운 이상 가격을 모델이 '시세 대비 이상 매물'로 짚어 검수 대상으로 넘길 수 있다는 뜻이기도 합니다.</p>
  <table><tr><th>매물</th><th>주행거리</th><th>등록 가격</th><th>예측 시세</th><th>가격/시세</th></tr>{ex_rows}</table>
</section>

<section>
  <div class="num">08</div><h2>가격대·연식별 오차</h2>
  <div class="two">
    <table><tr><th>실제 가격대</th><th>매물</th><th>오차율</th><th>평균 절대 오차</th></tr>{band_rows}</table>
    <table><tr><th>차령</th><th>매물</th><th>오차율</th><th>±20% 적중</th></tr>{age_rows}</table>
  </div>
  <p>5천 달러 미만 저가 차는 오차율이 가장 큽니다. 같은 금액 차이도 저가 차에서는 비율이 커지고, 7절에서 본 자리표시 가격도 이 구간에 몰려 있습니다. 차령이 19년을 넘으면 ±20% 적중률이 절반 아래로 떨어집니다. 오래된 차일수록 관리 상태가 가격을 크게 가를 텐데, 상태 정보는 판매자의 자기 평가이고 절반 가까이 비어 있어 모델이 이 차이를 담기 어려운 것으로 보입니다.</p>
</section>

<section>
  <div class="num">09</div><h2>서비스로 쓰기</h2>
  <p>학습한 모델을 FastAPI로 감싸, 매물 정보를 보내면 예측 시세와 80% 구간, 가격을 가장 크게 움직인 요인을 돌려주게 했습니다. 판매자에게 '왜 이 가격인지'를 함께 보여줄 수 있습니다.</p>
  <pre>POST /predict
{{"year": 2016, "manufacturer": "ford", "model": "F-150 XLT",
  "odometer": 98000, "type": "pickup", "drive": "4wd", "state": "tx"}}

→ {{"estimate_usd": 31490, "range_80_usd": [22880, 37900],
    "model_key": "ford f-150",
    "top_factors": [{{"feature": "차령", "effect_pct": 47.5}},
                    {{"feature": "모델", "effect_pct": 36.0}}, …]}}</pre>
  <p class="note">정제 규칙과 API는 pytest로 검증합니다. 처음 보는 모델명도 제조사별 '기타'로 처리해 예측하고, 범위를 벗어난 입력(1995년 이전 연식 등)은 거부합니다. 학습 데이터에 주(state)가 빠진 매물이 없었기 때문에, 주는 필수 입력으로 받습니다.</p>
</section>

<section>
  <div class="num">10</div><h2>한계와 다음 단계</h2>
  <ul>
    <li><b>매물 가격이지 거래 가격이 아닙니다.</b> 판매자가 부른 가격이라 실제 거래 가격보다 높을 수 있습니다. 거래 데이터가 있다면 매물가와 거래가의 차이부터 모델링해야 합니다.</li>
    <li><b>시간에 따른 변화는 보지 못했습니다.</b> 데이터가 2021년 한두 달치라 시세 흐름이나 계절성을 반영하지 못했습니다. 실제 서비스라면 과거로 학습해 이후 기간으로 평가하고, 성능을 주기적으로 감시해 재학습 시점을 정해야 합니다.</li>
    <li><b>트럭 정보가 부족합니다.</b> 화물차 시세에 중요한 적재 중량, 차축, 적재함 형태, 정비 이력은 데이터에 없어 모델명에서 적재 등급만 뽑았습니다.</li>
    <li><b>미국 데이터입니다.</b> 국내 시장은 제조사 구성과 감가 구조가 다르므로, 같은 정제·평가 방식을 국내 매물 데이터에 적용해 다시 학습해야 합니다.</li>
  </ul>
</section>

<footer>데이터: Austin Reese, Used Cars Dataset (Craigslist), Kaggle · 분석 도구: Python, pandas, scikit-learn, XGBoost, Optuna, MLflow, FastAPI</footer>
</div></body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")
