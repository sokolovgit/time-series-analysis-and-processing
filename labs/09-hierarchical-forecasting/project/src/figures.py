"""Графіки роботи: дані, похибки методів, прогнози за країною, областями й типом місцевості."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis
import hierarchy
import plots

HISTORY_FROM = 2005
ZOOM_FROM = 2014
METHOD_COLORS = {
    "base": plots.MUTED,
    "BU": plots.COLORS[0],
    "TD-AHP": plots.COLORS[7],
    "TD-PHA": plots.COLORS[4],
    "TD-FP": plots.COLORS[3],
    "MO": plots.COLORS[6],
    "OLS": plots.COLORS[1],
    "WLS": plots.COLORS[5],
    "MinT": plots.COLORS[2],
}
EXAMPLES = ("Львівська", "Харківська", "Донецька", "Закарпатська")


def thousands(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}"))


def break_line(ax):
    ax.axvline(analysis.BREAK - 0.5, color=plots.AXIS, linewidth=1, linestyle="--")


def country(data, out):
    years, h = data["years"], data["h"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(years, data["total"], color=plots.INK, label="Σ 25 регіонів")
    a.plot(
        years,
        data["official"],
        "o",
        color=plots.COLORS[1],
        markersize=3,
        label="офіційний ряд «Україна»",
    )
    break_line(a)
    thousands(a)
    a.set_title("Живонароджені, Україна")
    a.set_ylabel("тис. осіб")
    a.legend()
    names = np.array(h["names"])
    for area, color in zip(hierarchy.LOCAL, plots.COLORS[:2], strict=True):
        cols = [i for i, n in enumerate(names) if n.endswith(area)]
        b.plot(years, h["Y"][:, cols].sum(axis=1), color=color, label=area)
    break_line(b)
    thousands(b)
    b.set_title("За типом місцевості")
    b.set_ylabel("тис. осіб")
    b.legend()
    plots.save(fig, out, "01_country.png")


def regions(data, out):
    h, years = data["h"], data["years"]
    fig, axes = plt.subplots(5, 5, figsize=(12, 10), sharex=True)
    for ax, region in zip(axes.flat, hierarchy.REGIONS, strict=True):
        for area, color in zip(hierarchy.LOCAL, plots.COLORS[:2], strict=True):
            name = f"{region}, {area}"
            if name in h["names"]:
                ax.plot(years, h["Y"][:, h["names"].index(name)], color=color, linewidth=1.1)
        break_line(ax)
        thousands(ax)
        ax.set_title(region, fontsize=9)
        ax.set_ylim(bottom=0)
    fig.legend(list(hierarchy.LOCAL), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    plots.save(fig, out, "02_regions.png", rect=(0, 0, 1, 0.97))


def incoherence(finals, out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    result = finals[3]
    h, F, years = result["h"], result["F"], result["years"]
    hist = h["years"] >= ZOOM_FROM
    ax.plot(h["years"][hist], h["Y"][hist, 0], color=plots.INK, label="факт")
    ax.plot(years, F[0], color=plots.COLORS[0], marker="o", markersize=3, label="ŷ країни")
    ax.plot(
        years,
        F[h["level"] == 1].sum(axis=0),
        color=plots.COLORS[1],
        marker="o",
        markersize=3,
        label="Σ ŷ областей",
    )
    ax.plot(
        years,
        F[h["bottom"]].sum(axis=0),
        color=plots.COLORS[2],
        marker="o",
        markersize=3,
        label="Σ ŷ міська + сільська",
    )
    thousands(ax)
    ax.set_ylabel("тис. осіб")
    ax.set_title("Базові прогнози трьох рівнів")
    ax.legend()
    plots.save(fig, out, "03_incoherence.png")


def backtest(tables, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 4]})
    vmax = max(t.to_numpy().max() for t in tables.values())
    for ax, (depth, table) in zip(axes, tables.items(), strict=True):
        plots.heatmap(
            ax,
            table.to_numpy(),
            list(table.index),
            list(table.columns),
            cmap=plots.SEQUENTIAL,
            vmin=0,
            vmax=vmax,
            fmt="{:.1f}",
        )
        ax.set_title(f"MAPE, %: {depth} рівні")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=0, ha="center")
    plots.save(fig, out, "04_backtest.png")


def horizon(errors, out):
    frame = errors[errors["depth"] == 3]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for level, ax in enumerate(axes):
        part = frame[frame["level"] == level].groupby(["method", "step"])["ape"].mean().unstack()
        for method in part.index:
            style = "--" if method == "base" else "-"
            ax.plot(
                part.columns,
                part.loc[method],
                style,
                marker="o",
                markersize=3,
                color=METHOD_COLORS[method],
                label=method,
            )
        ax.set_title(f"Рівень «{hierarchy.LEVELS[level]}»")
        ax.set_xlabel("крок прогнозу, роки")
        ax.set_xticks(part.columns)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MAPE, %")
    axes[-1].legend(ncol=2)
    plots.save(fig, out, "05_horizon.png")


def country_forecast(result, out):
    h, years = result["h"], result["years"]
    hist = h["years"] >= ZOOM_FROM
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(h["years"][hist], h["Y"][hist, 0], color=plots.INK, label="факт")
    last = h["Y"][-1, 0]
    for method, forecast in result["forecasts"].items():
        if method.startswith("TD"):
            continue
        ax.plot(
            np.r_[h["years"][-1], years],
            np.r_[last, forecast[0]],
            "--" if method == "base" else "-",
            color=METHOD_COLORS[method],
            label="base = TD-AHP, TD-PHA, TD-FP" if method == "base" else method,
        )
    thousands(ax)
    ax.set_ylabel("тис. осіб")
    ax.set_title(f"Прогноз живонароджених по Україні на {years[0]}–{years[-1]}")
    ax.legend(ncol=3)
    plots.save(fig, out, "06_country_forecast.png")


def regions_forecast(result, best, out):
    h, years = result["h"], result["years"]
    hist = h["years"] >= HISTORY_FROM
    fig, axes = plt.subplots(5, 5, figsize=(12, 10), sharex=True)
    for ax, region in zip(axes.flat, hierarchy.REGIONS, strict=True):
        i = h["names"].index(region)
        ax.plot(h["years"][hist], h["Y"][hist, i], color=plots.INK, linewidth=1.1)
        for method in (best, "TD-AHP"):
            path = np.r_[h["Y"][-1, i], result["forecasts"][method][i]]
            ax.plot(np.r_[h["years"][-1], years], path, color=METHOD_COLORS[method], linewidth=1.3)
        thousands(ax)
        ax.set_title(region, fontsize=9)
        ax.set_ylim(bottom=0)
    fig.legend(["факт", best, "TD-AHP"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    plots.save(fig, out, "07_regions_forecast.png", rect=(0, 0, 1, 0.97))


def middle_out(result, best, out):
    h, years = result["h"], result["years"]
    hist = h["years"] >= HISTORY_FROM
    fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(12, 3.4))
    for ax, region in zip(axes, EXAMPLES, strict=True):
        for area, color in zip(hierarchy.LOCAL, plots.COLORS[:2], strict=True):
            i = h["names"].index(f"{region}, {area}")
            ax.plot(h["years"][hist], h["Y"][hist, i], color=color, linewidth=1.1, label=area)
            for method, style in (("MO", "-"), (best, ":"), ("TD-AHP", "--")):
                path = np.r_[h["Y"][-1, i], result["forecasts"][method][i]]
                ax.plot(np.r_[h["years"][-1], years], path, style, color=color, linewidth=1.3)
        thousands(ax)
        ax.set_title(region, fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("тис. осіб")
    handles = [
        plt.Line2D([], [], color=plots.COLORS[0], label="міська"),
        plt.Line2D([], [], color=plots.COLORS[1], label="сільська"),
        plt.Line2D([], [], color=plots.INK_SECONDARY, linestyle="-", label="MO"),
        plt.Line2D([], [], color=plots.INK_SECONDARY, linestyle=":", label=best),
        plt.Line2D([], [], color=plots.INK_SECONDARY, linestyle="--", label="TD-AHP"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    plots.save(fig, out, "08_middle_out.png", rect=(0, 0, 1, 0.92))


def table(result, best, out):
    """Прогноз кращим методом за всіма вузлами і прогноз країни всіма методами – у xlsx."""
    h, years = result["h"], result["years"]
    nodes = pd.DataFrame(result["forecasts"][best].round(), index=h["names"], columns=years)
    nodes.insert(0, "рівень", [hierarchy.LEVELS[level] for level in h["level"]])
    nodes.insert(1, h["years"][-1], h["Y"][-1])
    methods = pd.DataFrame({m: f[0].round() for m, f in result["forecasts"].items()}, index=years).T
    with pd.ExcelWriter(out / "births_forecast.xlsx") as writer:
        nodes.to_excel(writer, sheet_name=f"прогноз {best}")
        methods.to_excel(writer, sheet_name="Україна, усі методи")
    return nodes


def all_figures(data, errors, tables, finals, best, out):
    country(data, out)
    regions(data, out)
    incoherence(finals, out)
    backtest(tables, out)
    horizon(errors, out)
    country_forecast(finals[3], out)
    regions_forecast(finals[3], best, out)
    middle_out(finals[3], best, out)
    return table(finals[3], best, out)
