"""Графіки: ряд, кандидати, бектест, залишки, фінальний прогноз."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAD = "#d03b3b"
DPI = 140
UNIT = "Викиди CO₂, млрд т"


def setup():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)


def series(out, years, y):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=(3, 2))
    top.plot(years, y, marker="o", ms=3, color=INK)
    top.set_title("Світові викиди CO₂, 1970–2024")
    top.set_ylabel(UNIT)
    growth = 100 * np.diff(y) / y[:-1]
    bottom.bar(years[1:], growth, color=np.where(growth < 0, BAD, COLORS[0]), width=0.8)
    bottom.axhline(0, color=AXIS, lw=0.8)
    bottom.set_title("Річний приріст")
    bottom.set_ylabel("%")
    bottom.set_xlabel("Рік")
    save(fig, out, "01_series.png")


def candidates(out, years, y, fits, horizon):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, y, "o", ms=3, color=INK, label="фактичні дані")
    future = np.arange(years[-1] + 1, years[-1] + horizon + 1)
    for (name, state, forecast), color in zip(fits, COLORS, strict=False):
        ax.plot(years, state["fitted"], color=color, label=name)
        ax.plot(future, forecast, color=color, ls="--")
    ax.axvline(years[-1] + 0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_title(f"Моделі тренду за МНК і їх екстраполяція на {horizon} років")
    ax.set_xlabel("Рік")
    ax.set_ylabel(UNIT)
    ax.legend(ncol=2)
    save(fig, out, "02_candidates.png")


def backtest(out, curves, best, hidden=("середнє",)):
    """Середнє не показане: його MAPE ~30% стискає решту кривих до низу графіка."""
    fig, ax = plt.subplots(figsize=(10, 5))
    h = np.arange(1, len(next(iter(curves.values()))) + 1)
    styles = {"наївний": (MUTED, "--"), "дрейф": (INK_SECONDARY, ":")}
    shown = [name for name in curves if name not in hidden]
    models = iter(COLORS)
    for name in shown:
        color, ls = styles.get(name) or (next(models), "-")
        ax.plot(
            h,
            curves[name],
            marker="o",
            ms=3,
            color=color,
            ls=ls,
            lw=2.6 if name == best else 1.2,
            label=name,
        )
    ax.set_title("Точність прогнозу поза вибіркою (rolling origin)")
    ax.set_xlabel("Горизонт прогнозу, років")
    ax.set_ylabel("MAPE, %")
    ax.set_xticks(h)
    ax.set_ylim(0, None)
    ax.legend(ncol=2)
    save(fig, out, "03_backtest.png")


def origins(out, years, y, starts, horizon, forecast, name, baseline):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, y, color=INK, lw=1.2, label="фактичні дані")
    for i, start in enumerate(starts):
        h = min(horizon, len(y) - start)
        span = years[start : start + h]
        ax.plot(span, forecast(y[:start], h), color=COLORS[0], lw=2, label=name if i == 0 else None)
        ax.plot(
            span,
            baseline(y[:start], h),
            color=COLORS[1],
            ls="--",
            label="наївний" if i == 0 else None,
        )
        ax.axvline(years[start] - 0.5, color=MUTED, lw=0.6, ls=":")
    ax.set_title(f"Прогнози на {horizon} років з різних точок старту")
    ax.set_xlabel("Рік")
    ax.set_ylabel(UNIT)
    ax.legend()
    save(fig, out, "04_origins.png")


def residuals(out, years, e, u, rho_e, rho_u):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(years, e, marker="o", ms=3, color=MUTED, label="eₜ тренду")
    axes[0].plot(years[1:], u, marker="o", ms=3, color=COLORS[0], label="uₜ після AR(1)")
    axes[0].axhline(0, color=AXIS, lw=0.8)
    axes[0].set_title("Залишки в логарифмічній шкалі")
    axes[0].set_xlabel("Рік")
    axes[0].legend()
    lags = np.arange(1, len(rho_e) + 1)
    axes[1].bar(lags - 0.2, rho_e, color=MUTED, width=0.4, label="eₜ")
    axes[1].bar(lags + 0.2, rho_u, color=COLORS[0], width=0.4, label="uₜ")
    bound = 1.96 / np.sqrt(len(u))
    for b in (bound, -bound):
        axes[1].axhline(b, color=BAD, ls="--", lw=0.8)
    axes[1].set_title("Автокореляція залишків")
    axes[1].set_xlabel("Лаг")
    axes[1].set_xticks(lags)
    axes[1].legend()
    axes[2].hist(u, bins=12, color=COLORS[0], edgecolor="white")
    axes[2].set_title("Розподіл uₜ")
    save(fig, out, "05_residuals.png")


def forecast(out, years, y, fitted, future, mean, model_band, empirical_band, name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, y, "o", ms=3, color=INK, label="фактичні дані")
    ax.plot(years, fitted, color=COLORS[0], label="тренд")
    ax.plot(future, mean, color=COLORS[0], lw=2, ls="--", label=f"прогноз ({name})")
    ax.fill_between(future, *empirical_band, color=COLORS[1], alpha=0.18, label="95% емпіричний")
    ax.fill_between(future, *model_band, color=COLORS[0], alpha=0.25, label="95% за моделлю")
    ax.axvline(years[-1] + 0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_title(f"Прогноз на {future[0]}–{future[-1]}")
    ax.set_xlabel("Рік")
    ax.set_ylabel(UNIT)
    ax.legend(loc="upper left")
    save(fig, out, "06_forecast.png")
