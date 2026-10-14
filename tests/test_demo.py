"""체험 페이지의 브라우저 계산기(demo/predict.js)가 XGBoost와 같은 값을 내는지 확인한다 (node 필요)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "demo"
pytestmark = pytest.mark.skipif(shutil.which("node") is None or not (DEMO / "model.json").exists(),
                                reason="node 또는 demo/model.json이 없음")

SCRIPT = """
const P = require(process.argv[1] + '/predict.js');
const m = P.prepare(JSON.parse(require('fs').readFileSync(process.argv[1] + '/model.json', 'utf8')));
const ck = m.check; let ep = 0, ec = 0, eq = 0;
ck.x.forEach((row, i) => {
  const x = row.map((v) => (v === null ? NaN : v));
  const p = P.predictPoint(m, x, true);
  ep = Math.max(ep, Math.abs(p.y - ck.pred[i]));
  p.contrib.forEach((c, j) => { ec = Math.max(ec, Math.abs(c - ck.contrib[i][j])); });
  P.predictQuantiles(m, x).forEach((v, j) => { eq = Math.max(eq, Math.abs(v - ck.quantile[i][j])); });
});
const f150 = P.estimate(m, { manufacturer: 'ford', model_base: 'f-150', year: 2016, odometer: 98000, state: 'tx',
                             type: 'pickup', drive: '4wd' });
const older = P.estimate(m, { manufacturer: 'ford', model_base: 'f-150', year: 2010, odometer: 98000, state: 'tx',
                              type: 'pickup', drive: '4wd' });
const more = P.estimate(m, { manufacturer: 'ford', model_base: 'f-150', year: 2016, odometer: 150000, state: 'tx',
                             type: 'pickup', drive: '4wd' });
console.log(JSON.stringify({ ep, ec, eq, f150, older: older.price, more: more.price }));
"""


@pytest.fixture(scope="module")
def result():
    out = subprocess.run(["node", "-e", SCRIPT, str(DEMO)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_browser_predictor_matches_xgboost(result):
    assert result["ep"] < 1e-4 and result["eq"] < 1e-4 and result["ec"] < 1e-4


def test_estimate_is_inside_range_and_monotone(result):
    r = result["f150"]
    assert r["low"] <= r["price"] <= r["high"]
    assert r["modelKey"] == "ford f-150" and r["knownModel"]
    assert result["older"] <= r["price"] and result["more"] <= r["price"]   # 단조 제약
