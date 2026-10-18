"""Рисунки звіту: по одній функції на кожну групу графіків."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, NullFormatter

import level1
import level3
import plots

SHORT = [m[:3] for m in level1.MONTHS]
METHOD_COLORS = dict(
    zip(
        (*level3.METHODS, level3.INTEGRATED, level3.CHOICE, level3.NAIVE),
        (*plots.COLORS[:7], plots.INK, plots.COLORS[7], plots.MUTED),
        strict=True,
    )
)
CLASS_COLORS = dict(zip(level3.CLASSES, plots.COLORS[:3], strict=True))


def sales(res, out):
    by_month = res["by_month"]
    x = np.arange(len(by_month))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, by_month["виручка"] / 1e3, 0.4, label="виручка", color=plots.COLORS[0])
    ax.bar(x + 0.2, by_month["витрати"] / 1e3, 0.4, label="витрати", color=plots.COLORS[3])
    ax.plot(x, by_month["прибуток"] / 1e3, "o-", color=plots.INK, label="прибуток")
    ax.axhline(0, color=plots.MUTED, lw=0.8)
    ax.set_xticks(x, SHORT)
    ax.set_ylabel("тис. грн")
    ax2 = ax.twinx()
    ax2.plot(x, by_month["рентабельність, %"], "s--", color=plots.BAD, ms=4, label="рентабельність")
    ax2.set_ylabel("рентабельність, %")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.set_ylim(-80, 560)
    ax2.set_ylim(-20, 45)
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left", ncol=4)
    ax.set_title("Показники ефективності мережі за місяцями 2021 р.")
    plots.save(fig, out, "01_kpi_months.png")

    profit, share = res["profit"], res["share"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, region in enumerate(res["by_region"].index):
        axes[0].plot(x, profit[region] / 1e3, "o-", ms=3, color=plots.COLORS[i], label=region)
    axes[0].axhline(0, color=plots.MUTED, lw=0.8)
    axes[0].set_xticks(x, SHORT)
    axes[0].set_ylabel("прибуток, тис. грн")
    axes[0].set_title("Динаміка прибутку за регіонами")
    axes[0].legend(ncol=2)
    total = res["by_region"]["прибуток"]
    axes[1].barh(total.index[::-1], total.to_numpy()[::-1] / 1e3, color=plots.COLORS[0])
    for i, (v, m) in enumerate(
        zip(
            total.to_numpy()[::-1],
            res["by_region"]["рентабельність, %"].to_numpy()[::-1],
            strict=True,
        )
    ):
        axes[1].text(v / 1e3, i, f" {v / 1e3:.1f} ({m:.1f} %)", va="center", fontsize=8)
    axes[1].set_xlim(0, total.max() / 1e3 * 1.35)
    axes[1].set_xlabel("тис. грн")
    axes[1].set_title("Прибуток за рік (рентабельність)")
    axes[1].grid(axis="y", visible=False)
    plots.save(fig, out, "02_profit_regions.png")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    plots.heatmap(
        ax,
        share.T.to_numpy(),
        list(share.columns),
        SHORT,
        cmap=plots.SEQUENTIAL,
        vmin=0,
        vmax=float(np.nanmax(share.to_numpy())),
        fmt="{:.1f}",
    )
    ax.set_xticks(range(len(SHORT)), SHORT, rotation=0, ha="center")
    ax.set_title("Частка регіону в прибутку місяця, %")
    plots.save(fig, out, "03_profit_share.png")

    models = res["models"]
    dates = res["dates"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for ax, (name, m) in zip(axes.flat, models.items(), strict=True):
        ax.plot(
            dates, m["y"] / 1e3, color=plots.COLORS[0], lw=0.7, alpha=0.7, label="прибуток за добу"
        )
        ax.scatter(
            dates[m["anomalies"]],
            m["y"][m["anomalies"]] / 1e3,
            s=12,
            color=plots.BAD,
            zorder=3,
            label="аномалії",
        )
        ax.plot(
            dates, m["fitted"] / 1e3, color=plots.INK, lw=1.8, label=f"МНК, степінь {m['degree']}"
        )
        ax.axhline(0, color=plots.MUTED, lw=0.6)
        ax.set_title(f"{name.capitalize()}: R² = {m['scores']['R2']:.3f}", fontsize=10)
        ax.set_ylabel("тис. грн")
        ax.legend(loc="lower right", fontsize=7)
    fig.autofmt_xdate()
    plots.save(fig, out, "04_profit_lsq.png")


def processes(cfgs, res, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, cfg in zip(axes.flat, cfgs, strict=True):
        r = res["results"][cfg["key"]]
        s = cfg["series"]
        ax.plot(s.index, s.to_numpy(), color=plots.COLORS[0], lw=1.1, label="ряд")
        ax.plot(s.index, r["fitted"], color=plots.INK, lw=1.4, label=f"МНК, степінь {r['degree']}")
        ax.plot(
            r["future"], r["forecast"], "--", color=plots.BAD, lw=1.6, label="прогноз МНК, 12 міс."
        )
        if len(r["shocks"]):
            ax.scatter(
                r["shocks"],
                s.loc[r["shocks"]].to_numpy(),
                s=28,
                facecolors="none",
                edgecolors=plots.COLORS[1],
                lw=1.4,
                zorder=3,
                label="шоки приростів",
            )
        ax.set_title(f"{cfg['title']}\n{r['props']['тип']}", fontsize=9.5)
        ax.set_ylabel(cfg["unit"])
        ax.legend(loc="upper left", fontsize=7)
    plots.save(fig, out, "05_processes.png")

    lags = np.arange(1, 25)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, cfg in enumerate(cfgs):
        r = res["results"][cfg["key"]]
        x = 100 * r["returns"]
        axes[0, i].hist(x, bins=40, color=plots.COLORS[i], alpha=0.8, edgecolor="white", lw=0.4)
        axes[0, i].set_title(f"{cfg['short']}: σ = {x.std():.2f} %", fontsize=9.5)
        axes[0, i].set_xlabel("приріст за місяць, %")
        axes[0, i].set_yticks([])
        rho = np.array([np.corrcoef(x[k:], x[:-k])[0, 1] for k in lags])
        bound = 2 / np.sqrt(len(x))
        axes[1, i].bar(lags, rho, color=plots.COLORS[i])
        axes[1, i].axhspan(-bound, bound, color=plots.MUTED, alpha=0.2, lw=0)
        axes[1, i].set_xticks([1, 6, 12, 18, 24])
        axes[1, i].set_xlabel("лаг, міс.")
        axes[1, i].set_ylim(-1, 1)
    axes[1, 0].set_ylabel("ACF приростів")
    plots.save(fig, out, "06_returns.png")


def study(cfgs, res, out):
    results = res["results"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, cfg in zip(axes.flat, cfgs, strict=True):
        r = results[cfg["key"]]
        run = r["runs"][-1]
        s = cfg["series"]
        start, w = run["start"], level3.WINDOW
        dates = s.index
        lo = max(0, start + w - 60)
        ax.plot(
            dates[lo : start + w + level3.STEPS],
            s.to_numpy()[lo : start + w + level3.STEPS],
            color=plots.REAL,
            lw=1.6,
            label="факт",
        )
        future = dates[start + w : start + w + level3.STEPS]
        for c, m in r["by_class"].items():
            ax.plot(future, run["full"][m], color=CLASS_COLORS[c], lw=1.3, label=f"{m} ({c})")
        ax.plot(
            future,
            run["full"][level3.INTEGRATED],
            "--",
            color=plots.INK,
            lw=2.0,
            label=level3.INTEGRATED,
        )
        ax.plot(
            future, run["full"][level3.NAIVE], ":", color=plots.MUTED, lw=1.2, label=level3.NAIVE
        )
        ax.axvline(dates[start + w - 1], color=plots.MUTED, lw=0.8)
        ax.set_title(cfg["title"], fontsize=9.5)
        ax.set_ylabel(cfg["unit"])
        ax.legend(fontsize=6.5, loc="best")
    plots.save(fig, out, "07_forecasts.png")

    methods = [*level3.METHODS, level3.CHOICE, level3.INTEGRATED]
    ranks = pd.DataFrame(
        {cfg["short"]: results[cfg["key"]]["table"].loc[methods, "ранг"] for cfg in cfgs}
    )
    ranks["середній"] = ranks.mean(axis=1)
    u = pd.DataFrame(
        {cfg["short"]: results[cfg["key"]]["table"].loc[methods, "U Тейла 24"] for cfg in cfgs}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plots.heatmap(
        axes[0],
        ranks.to_numpy(),
        methods,
        list(ranks.columns),
        cmap=plots.SEQUENTIAL.reversed(),
        vmin=1,
        vmax=len(methods),
        fmt="{:.1f}",
    )
    axes[0].set_title("Ранг за інтегральним показником Y (1 – кращий)")
    plots.heatmap(
        axes[1], np.log2(u.to_numpy()), methods, list(u.columns), vmin=-1.5, vmax=1.5, fmt="{:.2f}"
    )
    for text, value in zip(axes[1].texts, u.to_numpy().ravel(), strict=True):
        text.set_text(f"{value:.2f}")
    axes[1].set_title("U Тейла на горизонті 24 міс. (< 1 – краще за наївний)")
    plots.save(fig, out, "08_rating.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, cfg in zip(axes.flat, cfgs, strict=True):
        table = results[cfg["key"]]["table"]
        naive = table.loc[level3.NAIVE, [f"RMSE {h}" for h in level3.HORIZONS]].to_numpy()
        for m in methods:
            rel = table.loc[m, [f"RMSE {h}" for h in level3.HORIZONS]].to_numpy() / naive
            style = "--" if m == level3.INTEGRATED else "-"
            ax.plot(
                level3.HORIZONS,
                rel,
                style,
                marker="o",
                ms=3,
                color=METHOD_COLORS[m],
                lw=2 if m == level3.INTEGRATED else 1.1,
                label=m,
            )
        ax.axhline(1, color=plots.MUTED, lw=0.8)
        ax.set_xticks(level3.HORIZONS)
        ax.set_xlabel("горизонт, міс.")
        ax.set_title(cfg["short"], fontsize=10)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3, 4, 5])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE / RMSE наївного")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    plots.save(fig, out, "09_horizons.png", rect=(0, 0.08, 1, 1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, cfg in zip(axes.flat, cfgs, strict=True):
        r = results[cfg["key"]]
        weights = r["weights"]
        x = np.arange(len(weights))
        bottom = np.zeros(len(weights))
        for c in level3.CLASSES:
            ax.bar(x, weights[c], bottom=bottom, color=CLASS_COLORS[c], label=c, width=0.7)
            bottom += weights[c].to_numpy()
        ax.set_xticks(
            x, [cfg["series"].index[s + level3.WINDOW].strftime("%Y") for s in weights.index]
        )
        ax.set_xlabel("рік початку прогнозу")
        ax.set_title(cfg["short"], fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("вага класу")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    plots.save(fig, out, "10_weights.png", rect=(0, 0.05, 1, 1))


def verification(checks, out):
    labels = [f"{s}, {i // 3}" for i, s in enumerate(checks["сигнал"])]
    x = np.arange(len(checks))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    bars = {
        "кращий метод": ("RMSE кращого / σ", plots.COLORS[0]),
        level3.INTEGRATED: ("RMSE інтегрованого / σ", plots.INK),
        level3.CHOICE: ("RMSE адаптивного вибору / σ", plots.COLORS[7]),
    }
    for j, (label, (col, color)) in enumerate(bars.items()):
        axes[0].bar(x + (j - 1) * 0.26, checks[col], 0.26, color=color, label=label)
    axes[0].axhline(1, color=plots.MUTED, lw=0.8)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("RMSE на 24 кроки / σ шуму")
    axes[0].set_title("Точність відносно рівня шуму")
    axes[0].legend(fontsize=8)
    bottom = np.zeros(len(checks))
    for c in level3.CLASSES:
        axes[1].bar(
            x, checks[f"вага: {c}"], bottom=bottom, color=CLASS_COLORS[c], label=c, width=0.7
        )
        bottom += checks[f"вага: {c}"].to_numpy()
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("середня вага класу")
    axes[1].set_title("Ваги класів в інтегрованому прогнозі")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    axes[1].grid(axis="x", visible=False)
    plots.save(fig, out, "11_verification.png")
