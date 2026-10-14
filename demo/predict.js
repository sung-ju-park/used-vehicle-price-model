/* 브라우저에서 경량 XGBoost 모델로 시세를 계산한다. 서버 없이 model.json만 읽는다.
   src/export_demo.py가 만든 model.json의 규칙(predict_python)과 똑같이 나무를 따라 내려간다. */
(function (root) {
  "use strict";

  const TRUCK = /silverado|sierra|ram|f-2|f-3|f-4|1500|2500|3500/;

  function hexHas(hex, code) {
    // hex: 오른쪽으로 가는 범주 번호의 비트 묶음 (16진수, 맨 오른쪽 자리가 0~3번)
    const pos = hex.length - 1 - (code >> 2);
    if (pos < 0) return false;
    return ((parseInt(hex[pos], 16) >> (code & 3)) & 1) === 1;
  }

  function prepare(model) {
    for (const part of [model.point, model.quantile]) {
      for (const t of part.trees) {
        t.cat = new Map(Object.entries(t.c).map(([k, v]) => [Number(k), v]));
      }
    }
    model.catIndex = {};
    for (const [f, list] of Object.entries(model.categories)) {
      model.catIndex[f] = new Map(list.map((v, i) => [v, i]));
    }
    model.vocab = new Set(model.model_vocab);
    return model;
  }

  function walk(tree, x, contrib) {
    const n = tree.n, v = tree.v;
    let i = 0;
    while (n[i][0] !== -1) {
      const [l, r, f, thr, dl] = n[i];
      const xv = x[f];
      let next;
      if (Number.isNaN(xv)) next = dl ? l : r;
      else if (tree.cat.has(i)) next = hexHas(tree.cat.get(i), xv) ? r : l;
      else next = Math.fround(xv) < thr ? l : r;   // XGBoost는 float32로 비교한다
      if (contrib) contrib[f] += v[next] - v[i];
      i = next;
    }
    return v[i];
  }

  function predictPoint(model, x, withContrib) {
    const contrib = withContrib ? new Array(x.length).fill(0) : null;
    let y = model.point.base[0], bias = model.point.base[0];
    for (const t of model.point.trees) {
      y += walk(t, x, contrib);
      bias += t.v[0];
    }
    return { y, contrib, bias };
  }

  function predictQuantiles(model, x) {
    const q = model.quantile.base.slice();
    for (const t of model.quantile.trees) q[t.g] += walk(t, x, null);
    return q.sort((a, b) => a - b);   // 분위수가 엇갈리면 XGBoost처럼 크기순으로 정렬
  }

  function loadClass(text) {
    const m = (text || "").toLowerCase();
    if (/\b(?:3500|350|f-?350|f-?450|4500|5500)\b|3500hd/.test(m)) return "3500급 이상";
    if (/\b(?:2500|250|f-?250)\b|2500hd/.test(m)) return "2500급";
    return "일반";
  }

  /* 입력값 → 모델 입력 벡터. features.py의 build()와 같은 규칙 */
  function features(model, input) {
    const a = model.about;
    const age = Math.max(0, a.ref_year - input.year);
    const odo = Number(input.odometer);
    const row = {
      age: age,
      odometer: odo,
      miles_per_year: odo / Math.max(age, 0.5),
      cylinders: input.cylinders ? Number(String(input.cylinders).match(/\d+/)?.[0]) : NaN,
    };
    const manu = input.manufacturer;
    const base = input.model_base || "";
    const key = manu + " " + base;
    row.model_key = base && model.vocab.has(key) ? key : manu + " (기타)";
    for (const c of ["manufacturer", "type", "condition", "fuel", "title_status", "transmission", "drive", "size",
                     "paint_color", "state"]) {
      row[c] = c === "manufacturer" ? manu : (input[c] || "unknown");
    }
    const truckish = a.truck_types.includes(input.type) || TRUCK.test(base);
    row.load_class = truckish ? (input.load_class || loadClass(base)) : "해당 없음";

    const x = model.features.map((f) => {
      if (f in model.catIndex) {
        const k = model.catIndex[f].get(row[f]);
        return k === undefined ? NaN : k;
      }
      return row[f];
    });
    return { x, row };
  }

  function estimate(model, input) {
    const { x, row } = features(model, input);
    const p = predictPoint(model, x, true);
    const q = predictQuantiles(model, x);
    const qhat = model.quantile.qhat;
    const lo = Math.min(q[0] - qhat, p.y), hi = Math.max(q[2] + qhat, p.y);
    const factors = model.features.map((f, i) => ({ feature: f, label: model.feature_ko[f], logEffect: p.contrib[i] }))
      .sort((a, b) => Math.abs(b.logEffect) - Math.abs(a.logEffect));
    return {
      price: Math.exp(p.y), low: Math.exp(lo), high: Math.exp(hi), basePrice: Math.exp(p.bias),
      modelKey: row.model_key, knownModel: !row.model_key.endsWith("(기타)"), loadClass: row.load_class, factors,
    };
  }

  const api = { prepare, features, predictPoint, predictQuantiles, estimate, loadClass };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.UVPredict = api;
})(this);
