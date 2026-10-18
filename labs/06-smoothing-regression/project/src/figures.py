"""Графіки роботи: ряди, згладжування, прогнози, KPI і набір рядів для селектора."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation
import plots
import selector

METHOD_COLORS = {
    "наївний": plots.MUTED,
    "експ. (AIC)": plots.COLORS[4],
    "експ. (CV)": plots.COLORS[0],
    "регресія (CV)": plots.COLORS[1],
    "комбінація": plots.COLORS[2],
    "гібрид": plots.COLORS[6],
    "дерево": plots.COLORS[3],
}


def legend(ax, **kw):
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, **kw)


def grid(runs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    return fig, list(zip(axes.flat, runs, strict=True))


def series(runs, out):
    fig, cells = grid(runs)
    for ax, run in cells:
        cfg = run["cfg"]
        y, w = cfg["series"], cfg["window"]
        ax.plot(y, color=plots.REAL, lw=0.6 if len(y) > 1000 else 0.9, label="ряд")
        if cfg["truth"] is not None:
            ax.plot(cfg["truth"], color=plots.BAD, lw=1.6, label="істинний тренд")
        ax.axvspan(0, w, color=plots.COLORS[0], alpha=0.08, lw=0, label=f"перше вікно, W = {w}")
        ax.set_title(cfg["title"])
        legend(ax, loc="upper left")
    plots.save(fig, out, "01_series.png")


def exponential_fit(runs, out):
    fig, cells = grid(runs)
    for ax, run in cells:
        cfg, exp = run["cfg"], run["exp"]
        w = cfg["window"]
        span = slice(max(0, w - 365), w) if w > 400 else slice(0, w)
        x = np.arange(w)[span]
        ax.plot(x, cfg["series"][span], color=plots.REAL, lw=0.8, alpha=0.6, label="ряд")
        if cfg["truth"] is not None:
            ax.plot(x, cfg["truth"][span], color=plots.BAD, lw=1.4, label="істинний тренд")
        ax.plot(
            x,
            exp["result"].fittedvalues[span],
            color=plots.COLORS[0],
            lw=1.4,
            label=f"{exp['best']} (AIC), α = {exp['table'].loc[exp['best'], 'α']:.2f}",
        )
        ax.set_title(cfg["title"])
        legend(ax, loc="upper left")
    plots.save(fig, out, "02_exponential_fit.png")


def alpha_curves(runs, out):
    fig, cells = grid(runs)
    for ax, run in cells:
        cfg, exp = run["cfg"], run["exp"]
        alphas = exp["alphas"]
        labels = [("опт." if a is None else f"{a:g}") for a in alphas]
        values = list(alphas.values())
        best = int(np.argmin(values))
        colors = [plots.GOOD if i == best else plots.COLORS[0] for i in range(len(values))]
        ax.bar(labels, values, color=colors, alpha=0.85)
        ax.set_ylim(0, max(values) * 1.15)
        ax.set_title(f"{cfg['title']}: {exp['cv_best']}")
        ax.set_xlabel("α (опт. – мінімум похибки на крок)")
        ax.set_ylabel(f"CV RMSE на горизонті {cfg['window'] // 4}")
    plots.save(fig, out, "03_alpha_cv.png")


def regression_fit(runs, out):
    fig, cells = grid(runs)
    for ax, run in cells:
        cfg, reg = run["cfg"], run["reg"]
        w = cfg["window"]
        ax.plot(cfg["series"][:w], color=plots.REAL, lw=0.7, alpha=0.6, label="ряд")
        if cfg["truth"] is not None:
            ax.plot(cfg["truth"][:w], color=plots.BAD, lw=1.4, label="істинний тренд")
        ax.plot(reg["fitted"], color=plots.COLORS[1], lw=1.3, label=reg["best"])
        ax.set_title(cfg["title"])
        legend(ax, loc="upper left")
    plots.save(fig, out, "04_regression_fit.png")


def forecasts(runs, out):
    fig, cells = grid(runs)
    shown = ["експ. (CV)", "регресія (CV)", "гібрид"]
    for ax, run in cells:
        cfg, results = run["cfg"], run["forecast"]
        y, w = cfg["series"], cfg["window"]
        s, _ = results[shown[0]]["paths"][-1]
        steps = len(results[shown[0]]["paths"][-1][1])
        # для добової температури – рік історії і рік прогнозу, інакше лінії зливаються
        if w > 400:
            steps = 365
        tail = min(w, 2 * steps)
        x0 = s + w - tail
        ax.plot(np.arange(x0, s + w + steps), y[x0 : s + w + steps], color=plots.REAL, lw=0.8)
        ax.axvline(s + w, color=plots.AXIS, lw=1)
        for name in shown:
            path = results[name]["paths"][-1][1][:steps]
            style = {"lw": 2.2, "ls": "--"} if name == "гібрид" else {"lw": 1.4}
            ax.plot(
                np.arange(s + w, s + w + steps),
                path,
                color=METHOD_COLORS[name],
                label=name,
                **style,
            )
        ax.set_title(f"{cfg['title']}, старт {s}")
        legend(ax, loc="upper left")
    plots.save(fig, out, "05_forecasts.png")


def kpi_heatmap(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [m for m in runs[0]["forecast"] if m != "наївний"]
    for ax, h in zip(axes, evaluation.HORIZONS, strict=True):
        matrix = []
        for run in runs:
            res = run["forecast"]
            base = res["наївний"]["summary"].loc[h, "RMSE"]
            matrix.append([res[m]["summary"].loc[h, "RMSE"] / base for m in names])
        matrix = np.array(matrix).T
        plots.heatmap(
            ax,
            np.log2(matrix),
            names,
            [r["cfg"]["short"] for r in runs],
            vmin=-2,
            vmax=2,
        )
        for text, v in zip(ax.texts, matrix.flat, strict=True):
            text.set_text(f"{v:.2f}")
        ax.set_title(f"RMSE / RMSE наївного, горизонт {h:g}W".replace(".", ","))
    plots.save(fig, out, "06_kpi_heatmap.png")


def benchmark(table, cv_runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    counts = pd.crosstab(table["тренд"], table["краще"]).reindex(columns=list(selector.FAMILIES))
    share = counts.div(counts.sum(axis=1), axis=0)
    left = np.zeros(len(share))
    for family, color in zip(selector.FAMILIES, (plots.COLORS[0], plots.COLORS[1]), strict=True):
        axes[0].barh(share.index, share[family], left=left, color=color, label=family)
        left += share[family].to_numpy()
    axes[0].set_xlabel("частка рядів, де сім'я краща")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    axes[0].set_title("краща сім'я за типом тренду")

    for family, color in zip(selector.FAMILIES, (plots.COLORS[0], plots.COLORS[1]), strict=True):
        part = table[table["краще"] == family]
        axes[1].scatter(
            part["R² тренду"], part["шум/рівень"], s=10, color=color, alpha=0.6, label=family
        )
    axes[1].set_xlabel("R² квадратичного тренду")
    axes[1].set_ylabel("СКВ приростів / СКВ ряду")
    axes[1].set_title("ознаки ряду і краща сім'я")
    axes[1].legend(loc="upper right")

    frame = pd.DataFrame(cv_runs).drop(columns="k")
    order = frame.mean().sort_values().index
    for i, name in enumerate(order):
        axes[2].barh(
            i,
            frame[name].mean(),
            color=plots.COLORS[6] if name == "гібрид" else plots.COLORS[0],
            alpha=0.8,
        )
        axes[2].scatter(frame[name], [i] * len(frame), color=plots.INK, s=10, zorder=3)
    axes[2].set_yticks(range(len(order)), order)
    axes[2].set_xlim(1, frame.to_numpy().max() * 1.05)
    axes[2].set_xlabel("RMSE / RMSE оракула (точки – блоки CV)")
    axes[2].set_title("стратегії вибору на нових рядах")
    plots.save(fig, out, "07_benchmark.png")
