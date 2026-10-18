"""Зведені графіки по обох рядах: похибка за горизонтом, діагностика ARIMA, оптимізація, KPI."""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

import evaluation
import plots
import study

LEADS = {
    "rates": [1, 2, 3, 5, 10, 20, 50, 100, 150, 200],
    "temp": [1, 2, 3, 5, 7, 10, 20, 50, 100, 200, 365, 730, 1460],
}


def horizon_errors(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(12, 4.2))
    for ax, run in zip(axes, runs, strict=True):
        cfg = run["cfg"]
        for name, (agg, _, _) in run["expert"].items():
            if name in cfg["shown"]:
                ax.plot(
                    agg.index, agg["RMSE"], marker="o", ms=4, color=study.color_of(name), label=name
                )
        ax.set_xticks(evaluation.HORIZONS, [f"{h:g}W" for h in evaluation.HORIZONS])
        ax.set_title(cfg["title"])
        ax.set_xlabel("горизонт екстраполяції")
        ax.set_ylabel(f"RMSE, {cfg['unit']}")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.suptitle("Похибка екстраполяції за горизонтом, параметри групи 1", fontsize=11)
    plots.save(fig, out, "07_horizon_errors.png")


def arima_diagnostics(runs, out):
    fig, axes = plt.subplots(2, len(runs), figsize=(12, 6))
    lags = np.arange(1, 31)
    for col, run in enumerate(runs):
        cfg = run["cfg"]
        for row, group in enumerate(("група 1", "група 2")):
            resid = run["resid"][group]
            values = acf(resid, nlags=30, fft=True)[1:]
            bound = 1.96 / np.sqrt(len(resid))
            ax = axes[row, col]
            ax.bar(lags, values, color=plots.COLORS[0] if row else plots.MUTED, width=0.6)
            ax.axhspan(-bound, bound, color=plots.MUTED, alpha=0.18, lw=0)
            ax.set_title(f"{cfg['title']}, ARIMA {group}", fontsize=9)
            ax.set_xlabel(f"лаг, {cfg['step']}")
            ax.set_ylim(-0.3, 0.3)
    fig.suptitle("ACF залишків моделей на першому вікні", fontsize=11)
    plots.save(fig, out, "08_arima_diagnostics.png")


def tuning_curves(runs, out):
    panels = [
        ("rates", "SMA"),
        ("rates", "EMA"),
        ("rates", "α-β фільтр"),
        ("temp", "SMA"),
        ("temp", "сезонне MA"),
        ("temp", "МНК + гармоніки"),
    ]
    by_key = {run["cfg"]["key"]: run for run in runs}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (key, name) in zip(axes.flat, panels, strict=True):
        run = by_key[key]
        curves = run["curves"][name]
        labels = [study.describe_choice(c) for c in curves[0]]
        for i, scores in enumerate(curves):
            ax.plot(
                range(len(labels)),
                list(scores.values()),
                marker="o",
                ms=3,
                lw=1.0,
                color=plots.COLORS[i % len(plots.COLORS)],
                alpha=0.8,
                label=f"старт {i + 1}",
            )
        ax.set_xticks(
            range(len(labels)), labels, rotation=60 if len(labels) > 6 else 0, fontsize=6.5
        )
        ax.set_title(f"{run['cfg']['title']}: {name}", fontsize=9)
        ax.set_ylabel(f"CV RMSE, {run['cfg']['unit']}")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Крос-валідація параметрів на цільовому горизонті", fontsize=11)
    plots.save(fig, out, "09_tuning_curves.png")


def optimization_gain(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(12, 4.6))
    for ax, run in zip(axes, runs, strict=True):
        gain = run["gain"]
        names = list(gain.index)
        x = np.arange(len(names))
        ax.bar(
            x - 0.2,
            gain["група 1, 2W"].astype(float),
            width=0.38,
            color=plots.MUTED,
            label="група 1",
        )
        ax.bar(
            x + 0.2,
            gain["група 2, 2W"].astype(float),
            width=0.38,
            color=plots.COLORS[0],
            label="група 2",
        )
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel(f"RMSE на 2W, {run['cfg']['unit']}")
        ax.set_title(run["cfg"]["title"])
        ax.legend(fontsize=7)
    fig.suptitle("Ефект оптимізації параметрів", fontsize=11)
    plots.save(fig, out, "10_optimization_gain.png")


def kpi_heatmap(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(12, 5))
    for ax, run in zip(axes, runs, strict=True):
        kpi = run["kpi"].sort_values("ранг")
        cols = [c for c in kpi.columns if c.startswith("RMSE")]
        values = kpi[cols].astype(float).to_numpy()
        ratio = values / values[list(kpi.index).index("наївний")]
        image = plots.heatmap(
            ax,
            ratio,
            list(kpi.index),
            [c.replace("RMSE ", "") for c in cols],
            cmap=plots.DIVERGING,
            vmin=0,
            vmax=2,
        )
        ax.set_title(f"{run['cfg']['title']}: RMSE / RMSE наївного")
        fig.colorbar(image, ax=ax, pad=0.02)
    plots.save(fig, out, "11_kpi_heatmap.png")


def lead_sweep(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(12, 4.4))
    tables = {}
    for ax, run in zip(axes, runs, strict=True):
        cfg = run["cfg"]
        methods = {name: cfg["expert"][name] for name in cfg["shown"]}
        table, starts = study.lead_errors(cfg, run["series"], methods, LEADS[cfg["key"]])
        tables[cfg["key"]] = (table, starts)
        for name, row in table.iterrows():
            ax.plot(row.index, row.values, marker="o", ms=3, color=study.color_of(name), label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"крок прогнозу k, {cfg['step']}")
        ax.set_ylabel(f"RMSE, {cfg['unit']}")
        ax.set_title(f"{cfg['title']}: {starts} стартів")
        ax.legend(fontsize=7)
    fig.suptitle("Похибка прогнозу за кроком", fontsize=11)
    plots.save(fig, out, "12_lead_sweep.png")
    return tables
