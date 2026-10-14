"""README와 리포트에 쓰는 그래프. 축은 하나만 쓰고, 계열이 둘 이상이면 범례와 직접 라벨을 함께 단다."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from config import FIG, PROCESSED, ROOT, SEGMENT_KO

# 한글 글꼴: 저장소의 fonts/ 폴더를 먼저 등록한다. 찾지 못하면 한글이 네모로 깨지므로 바로 멈춘다.
for d in [ROOT / "fonts", Path(os.environ.get("FONT_DIR", "/nonexistent"))]:
    if d.is_dir():
        for f in d.glob("*.[ot]tf"):
            font_manager.fontManager.addfont(str(f))
_avail = {f.name for f in font_manager.fontManager.ttflist}
_font = next((n for n in ["Pretendard", "Malgun Gothic", "AppleGothic", "NanumGothic"] if n in _avail), None)
if _font is None:
    raise RuntimeError("한글 글꼴을 찾지 못했습니다. fonts/ 폴더에 Pretendard(.ttf)를 넣으세요.")

INK, SUB, GRID = "#0b0b0b", "#52514e", "#e5e4df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9"
plt.rcParams.update({
    "font.family": _font, "axes.unicode_minus": False, "figure.dpi": 150, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID, "axes.labelcolor": SUB,
    "xtick.color": SUB, "ytick.color": SUB, "axes.titlecolor": INK, "axes.grid": False,
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=12.5, pad=12)


def _hbar(ax, labels, values, colors, fmt, xmax=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    top = xmax or max(values) * 1.28
    for yi, v in zip(y, values):
        ax.text(v + top * 0.012, yi, fmt(v), va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, top)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def funnel(r):
    f = r["funnel"]
    labels = [s["step"] for s in f]
    vals = [s["rows"] for s in f]
    colors = [GRAY] * (len(f) - 1) + [BLUE]
    colors[-2] = GRAY
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    _hbar(ax, labels, vals, [ORANGE if s["step"] == "중복 매물" else (BLUE if i == 0 else GRAY) for i, s in enumerate(f)],
          lambda v: f"{v:,}건")
    ax.set_xlabel("남은 매물 수")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _title(ax, "정제 단계별로 남은 매물 수: 가장 많이 빠진 것은 반복 게시된 중복 매물")
    _save(fig, "funnel")


def leakage(r):
    lk = r["leakage"]
    labels = ["중복을 남기고 무작위로 나눈 평가", "중복을 지우고 차 단위로 나눈 평가 (이 프로젝트)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.6))
    for ax, key, name, fmt in [(axes[0], "within_20", "실제 가격 ±20% 안에 든 비율 (높을수록 좋음)", lambda v: f"{v*100:.1f}%"),
                               (axes[1], "mdape", "오차율 중앙값 (낮을수록 좋음)", lambda v: f"{v*100:.1f}%")]:
        vals = [lk["naive"][key], lk["honest"][key]]
        _hbar(ax, labels if ax is axes[0] else ["", ""], vals, [GRAY, BLUE], fmt, max(vals) * 1.35)
        ax.set_title(name, loc="left", fontsize=10.5, color=SUB, pad=6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("같은 모델이라도 평가 방식에 따라 점수가 달라진다", x=0.0, ha="left", fontsize=12.5, color=INK, y=1.2)
    fig.text(0.0, 1.08, f"중복을 남기고 나누면 평가 매물의 {lk['test_rows_seen_in_train']*100:.0f}%가 학습 데이터에 이미 있던 차였다",
             ha="left", fontsize=10, color=SUB)
    _save(fig, "leakage")


def models(r):
    order = sorted([kv for kv in r["models"].items() if kv[0] != "기준선: 전체 중앙값"], key=lambda kv: kv[1]["mdape"])
    names = [k for k, _ in order]
    vals = [v["mdape"] for _, v in order]
    colors = [BLUE if n == r["selected"] else (GRAY if n.startswith("기준선") else "#8fb6e8") for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    _hbar(ax, names, vals, colors, lambda v: f"{v*100:.1f}%")
    ax.set_xlabel("검증 데이터 오차율 중앙값 (낮을수록 좋음)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _title(ax, "모델별 검증 오차율: 이 점수로 최종 모델을 골랐다")
    _save(fig, "models")


def importance(r):
    imp = r["contributions"]["importance"][:12]
    labels = [d["label"] for d in imp]
    vals = [np.expm1(d["mean_abs"]) * 100 for d in imp]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    _hbar(ax, labels, vals, [BLUE] * 3 + ["#8fb6e8"] * (len(vals) - 3), lambda v: f"{v:.1f}%")
    ax.set_xlabel("예측 가격을 평균적으로 움직이는 정도 (%)")
    _title(ax, "가격 예측에 영향을 크게 준 변수 (SHAP 값 기준)")
    _save(fig, "importance")


def dependence(r):
    c = r["contributions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, key, xl, title, xs in [(axes[0], "age", "차령 (년)", "차령에 따른 가격 영향", 1),
                                   (axes[1], "odometer", "주행거리 (천 마일)", "주행거리에 따른 가격 영향", 1000)]:
        x = np.array(c[key]["x"]) / xs
        y = np.array(c[key]["effect_pct"])
        ax.axhline(0, color=GRID, lw=1)
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=4)
        ax.set_xlabel(xl)
        ax.set_ylabel("평균 대비 가격 영향 (%)")
        ax.set_title(title, loc="left", fontsize=11, pad=8)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("다른 조건을 그대로 둘 때 차령과 주행거리가 예측 가격을 움직이는 정도", x=0.0, ha="left",
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "dependence")


def segments(r):
    model = {d["group"]: d for d in r["segments"]["segment_model"]}
    table = {d["group"]: d for d in r["segments"]["segment_table"]}
    order = [g for g in ["truck", "suv", "sedan", "other", "unknown"] if g in model]
    labels = [f"{SEGMENT_KO[g]}\n({model[g]['n']:,}건)" for g in order]
    y = np.arange(len(order))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    a = [table[g]["mdape"] for g in order]
    b = [model[g]["mdape"] for g in order]
    ax.barh(y + h / 2 + 0.01, a, height=h, color=GRAY, label="기준선: 시세표")
    ax.barh(y - h / 2 - 0.01, b, height=h, color=BLUE, label=r["selected"])
    for yi, va, vb in zip(y, a, b):
        ax.text(va + 0.003, yi + h / 2, f"{va*100:.1f}%", va="center", fontsize=9, color=SUB)
        ax.text(vb + 0.003, yi - h / 2, f"{vb*100:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(a + b) * 1.25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("오차율 중앙값 (낮을수록 좋음)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9.5)
    _title(ax, "차종별 오차율")
    _save(fig, "segments")


def retention(r):
    rt = r["retention"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ends = []
    for s, color in [("truck", ORANGE), ("suv", BLUE), ("sedan", AQUA)]:
        d = rt[s]
        ax.plot(d["age"], np.array(d["retention"]) * 100, color=color, lw=2, marker="o", ms=4,
                label=f"{SEGMENT_KO[s]} (모델 {d['n_models']}개)")
        ends.append([d["retention"][-1] * 100, d["age"][-1], SEGMENT_KO[s]])
    ends.sort()
    for i in range(1, len(ends)):              # 끝 라벨이 겹치지 않게 최소 간격을 둔다
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 7)
    for yv, xv, label in ends:
        ax.text(xv + 0.3, yv, label, va="center", fontsize=10, color=INK)
    ax.set_xlabel("차령 (년)")
    ax.set_ylabel("차령 1~3년 가격 대비 (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, max(max(rt[s]["age"]) for s in ["truck", "suv", "sedan"]) + 2)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    _title(ax, "같은 모델 안에서 본 연식별 가격 유지율: 트럭·픽업은 값이 천천히 떨어진다")
    _save(fig, "retention")


def intervals_fig(r):
    iv = r["intervals"]
    order = ["truck", "suv", "sedan", "other", "unknown"]
    labels = ["전체"] + [SEGMENT_KO[s] for s in order]
    a = [iv["raw"]["coverage"]] + [None] * len(order)
    b = [iv["conformal"]["coverage"]] + [iv["by_segment"][s]["coverage"] for s in order]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    x = np.arange(len(labels))
    ax.bar(x[0] - 0.2, a[0], width=0.38, color=GRAY, label="분위수 회귀만")
    ax.bar(x[0] + 0.2, b[0], width=0.38, color=BLUE, label="보정 후 (CQR)")
    ax.bar(x[1:], b[1:], width=0.5, color=BLUE)
    ax.text(x[0] - 0.2, a[0] + 0.01, f"{a[0]*100:.1f}%", ha="center", fontsize=9, color=SUB)
    for xi, v in zip(x, b):
        ax.text(xi + (0.2 if xi == 0 else 0), v + 0.01, f"{v*100:.1f}%", ha="center", fontsize=9, color=INK)
    ax.axhline(0.8, color=ORANGE, lw=1.4, ls="--", label="목표 80%")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 0.95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("실제 가격이 구간 안에 든 비율")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=3)
    _title(ax, "80% 시세 구간이 실제로 80%를 담는가")
    _save(fig, "intervals")


def pred_vs_actual(r):
    a = np.load(PROCESSED / "test_arrays.npy", allow_pickle=True).item()
    y, p = np.exp(a["y"]) / 1000, np.exp(a["pred"]) / 1000
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    hb = ax.hexbin(p, y, gridsize=70, xscale="log", yscale="log", cmap="Blues", mincnt=1, bins="log", linewidths=0)
    lim = [1, 150]
    ax.plot(lim, lim, color=INK, lw=1)
    for k, ls in [(1.2, "--"), (1 / 1.2, "--")]:
        ax.plot(lim, [lim[0] * k, lim[1] * k], color=ORANGE, lw=1, ls=ls)
    ax.text(100, 100 * 1.2 * 1.05, "±20%", color=ORANGE, fontsize=9.5, ha="right")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ticks = [1, 2, 5, 10, 20, 50, 100]
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("예측 가격 (천 달러, 로그 축)")
    ax.set_ylabel("실제 가격 (천 달러, 로그 축)")
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("매물 수", color=SUB)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    _title(ax, "예측 가격과 실제 가격 (평가 데이터)")
    _save(fig, "pred_vs_actual")


def all_figs(r):
    funnel(r); leakage(r); models(r); importance(r); dependence(r); segments(r); retention(r)
    intervals_fig(r); pred_vs_actual(r)
