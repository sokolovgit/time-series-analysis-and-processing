"""Група 1: курс USD Ощадбанку – декомпозиція, властивості, кореляції, режими, синтез моделі."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analysis
import loaders
import patterns
import plots
import synthesis

PERIOD = 5
WINDOW = 10
LAGS = 30
MAX_LAG = 10
RUNS = 200
PATHS_SHOWN = 3

NAMES = {"buy": "купівля", "sell": "продаж", "nbu": "НБУ"}
REGIMES = ["низька премія", "курс = НБУ", "шок", "висока премія", "після девальвації"]
SHOWN_METRICS = ["σ buy", "σ Δbuy", "r Δbuy Δsell", "r buy nbu", "H Δbuy", "ексцес Δbuy"]


def features(frame):
    """Ознаки режиму: рівень НБУ, премія buy − nbu, спред sell − buy, волатильність |Δbuy|."""
    return pd.DataFrame(
        {
            "НБУ": frame["nbu"],
            "премія": frame["buy"] - frame["nbu"],
            "спред": frame["sell"] - frame["buy"],
            "волатильність": frame["buy"]
            .diff()
            .abs()
            .fillna(0.0)
            .rolling(WINDOW, min_periods=1)
            .mean(),
        }
    )


def metrics(frame):
    d = frame.diff().dropna()
    return {
        "m buy": frame["buy"].mean(),
        "σ buy": frame["buy"].std(),
        "σ Δbuy": d["buy"].std(),
        "σ Δnbu": d["nbu"].std(),
        "m спред": (frame["sell"] - frame["buy"]).mean(),
        "max спред": (frame["sell"] - frame["buy"]).max(),
        "r Δbuy Δsell": d["buy"].corr(d["sell"]),
        "r Δbuy Δnbu": d["buy"].corr(d["nbu"]),
        "r buy nbu": frame["buy"].corr(frame["nbu"]),
        "H Δbuy": analysis.hurst(d["buy"].to_numpy())[0],
        "acf1 Δbuy": d["buy"].autocorr(1),
        "ексцес Δbuy": stats.kurtosis(d["buy"]),
        "p ADF buy": analysis.stationarity(frame["buy"])["p_ADF"],
    }


def stage_input(frame, gaps, out):
    days = frame.index.normalize().nunique()
    print(
        f"   записів {len(frame)} за {days} днів "
        f"({frame.index[0]:%d.%m.%Y} – {frame.index[-1]:%d.%m.%Y})"
    )
    print(
        f"   записів на день: до {frame.index.value_counts().max()} – "
        "ряд подій, а не рівномірна сітка"
    )
    print(f"   заповнено пропусків (нулів у файлі): {gaps}")
    rows = {NAMES[c]: analysis.describe(frame[c]) for c in NAMES}
    rows |= {f"Δ{NAMES[c]}": analysis.describe(frame[c].diff().dropna()) for c in NAMES}
    print(analysis.table(rows).round(3).to_string())

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=(2, 1))
    for c in NAMES:
        top.plot(frame.index, frame[c], label=NAMES[c], drawstyle="steps-post")
    top.set_title("Курс USD в Ощадбанку")
    top.set_ylabel("грн")
    top.legend()
    bottom.plot(
        frame.index,
        frame["sell"] - frame["buy"],
        color=plots.COLORS[3],
        drawstyle="steps-post",
        label="спред",
    )
    bottom.plot(
        frame.index,
        frame["buy"] - frame["nbu"],
        color=plots.COLORS[6],
        drawstyle="steps-post",
        label="премія",
    )
    bottom.set_ylabel("грн")
    bottom.legend()
    plots.save(fig, out, "01_rates_input.png")


def stage_decomposition(frame, out):
    rows = {}
    for c in NAMES:
        res = analysis.decompose(frame[c].reset_index(drop=True), PERIOD)
        f_trend, f_season = analysis.strength(
            res.trend.values, res.seasonal.values, res.resid.values
        )
        rows[NAMES[c]] = {
            "F тренду": f_trend,
            "F сезонності": f_season,
            "σ залишку": res.resid.std(),
        }
        if c == "buy":
            buy = res
    print(f"   STL з періодом {PERIOD} записів, сила компонент F = max(0, 1 − Var(R)/Var(X + R)):")
    print(analysis.table(rows).round(3).to_string())
    print("   НБУ – кусково-сталий ряд: STL розкладає його без залишку, тому F не визначена")

    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    parts = [
        (buy.observed, "Ряд"),
        (buy.trend, "Тренд"),
        (buy.seasonal, "Сезонна компонента"),
        (buy.resid, "Залишок"),
    ]
    for ax, (values, title) in zip(axes, parts, strict=True):
        ax.plot(frame.index, values.values, color=plots.COLORS[0], lw=1.0)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_title("STL-декомпозиція курсу купівлі", loc="center")
    plots.save(fig, out, "02_rates_decomposition.png")


def stage_properties(frame, out):
    rows = {}
    for c in NAMES:
        rows[NAMES[c]] = analysis.stationarity(frame[c])
        rows[f"Δ{NAMES[c]}"] = analysis.stationarity(frame[c].diff().dropna())
    print("   тести стаціонарності (α = 0.05):")
    print(analysis.table(rows).to_string(float_format=lambda v: f"{v:.4f}"))

    hurst = {}
    for c in NAMES:
        d = frame[c].diff().dropna().to_numpy()
        hurst[f"Δ{NAMES[c]}"] = {
            "H класичний": analysis.hurst(d, corrected=False)[0],
            "H скоригований": analysis.hurst(d)[0],
        }
    print("   показник Херста приростів:")
    print(analysis.table(hurst).round(3).to_string())

    d = frame["buy"].diff().dropna()
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(12, 3.8))
    left.hist(d, bins=60, color=plots.COLORS[0], edgecolor="white", linewidth=0.4)
    left.set_yscale("log")
    left.set_title("Прирости купівлі (лог. шкала)")
    left.set_xlabel("Δbuy, грн")
    lags = np.arange(LAGS + 1)
    middle.plot(
        lags, analysis.autocorrelation(frame["buy"], LAGS), marker="o", ms=3, label="рівень"
    )
    middle.plot(lags, analysis.autocorrelation(d, LAGS), marker="o", ms=3, label="прирости")
    bound = 1.96 / np.sqrt(len(d))
    middle.axhspan(-bound, bound, color=plots.MUTED, alpha=0.15, lw=0)
    middle.set_title("Автокореляція")
    middle.set_xlabel("лаг, записів")
    middle.legend()
    h, sizes, rs = analysis.hurst(d.to_numpy())
    middle_ref = [analysis.expected_rs(n) for n in sizes]
    right.loglog(sizes, rs, "o", ms=5, color=plots.COLORS[0], label=f"Δbuy, H = {h:.2f}")
    right.loglog(sizes, middle_ref, color=plots.MUTED, lw=1.2, label="E[R/S] білого шуму")
    right.set_title("R/S-аналіз")
    right.set_xlabel("довжина вікна n")
    right.legend()
    plots.save(fig, out, "03_rates_properties.png")


def stage_correlation(frame, out, labels, colors):
    levels = patterns.correlation(frame)
    steps = patterns.correlation(frame.diff().dropna())
    names = list(NAMES.values())
    print("   кореляція рівнів:")
    print(levels.set_axis(names).set_axis(names, axis=1).round(3).to_string())
    print("   кореляція приростів:")
    print(steps.set_axis(names).set_axis(names, axis=1).round(3).to_string())
    d = frame.diff().dropna()
    lag, r = patterns.peak_lag(d["nbu"], d["buy"], MAX_LAG)
    print(f"   крос-кореляція Δnbu → Δbuy: максимум r = {r:.3f} на лагу {lag}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), width_ratios=(1, 1, 1.3))
    plots.heatmap(axes[0], levels, names, names)
    axes[0].set_title("Рівні")
    plots.heatmap(axes[1], steps, names, names)
    axes[1].set_title("Прирости")
    for k in np.unique(labels):
        mask = labels == k
        axes[2].scatter(
            frame["nbu"][mask], frame["buy"][mask], s=14, color=colors[k], label=REGIMES[k]
        )
    axes[2].set_title("Купівля проти НБУ за режимами")
    axes[2].set_xlabel("НБУ, грн")
    axes[2].set_ylabel("купівля, грн")
    axes[2].legend(fontsize=7)
    plots.save(fig, out, "04_rates_correlation.png")
    return levels, steps


def stage_regimes(frame, out):
    feats = features(frame)
    k, labels, scores = patterns.best_kmeans(feats)
    labels = patterns.relabel(labels, np.arange(len(labels)))
    print(
        "   силует K-means: "
        + ", ".join(f"k={key}: {v:.3f}" for key, v in scores.items())
        + f" → k = {k}"
    )
    summary = feats.groupby(labels).mean()
    summary.index = [REGIMES[i] for i in summary.index]
    print(summary.round(3).to_string())
    segs = patterns.segments(labels)
    print("   ділянки режимів у часі:")
    for label, start, end in segs:
        print(
            f"     {frame.index[start]:%d.%m.%Y} – {frame.index[end - 1]:%d.%m.%Y}  "
            f"{end - start:>4} записів  {REGIMES[label]}"
        )

    colors = {i: plots.COLORS[i] for i in range(k)}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=(2, 1))
    spans = [(frame.index[s], frame.index[min(e, len(frame) - 1)], label) for label, s, e in segs]
    plots.shade(top, spans, colors, REGIMES)
    top.plot(
        frame.index, frame["buy"], color=plots.INK, lw=1.2, drawstyle="steps-post", label="купівля"
    )
    top.plot(
        frame.index,
        frame["nbu"],
        color=plots.INK_SECONDARY,
        lw=1.0,
        ls="--",
        drawstyle="steps-post",
        label="НБУ",
    )
    top.set_title("Режими курсу, виявлені кластеризацією")
    top.set_ylabel("грн")
    top.legend(ncol=4, fontsize=7)
    bottom.plot(list(scores), list(scores.values()), marker="o", color=plots.COLORS[0])
    bottom.axvline(k, color=plots.MUTED, ls=":")
    bottom.set_title("Коефіцієнт силуету", fontsize=9)
    bottom.set_xticks(list(scores))
    bottom.set_xlabel("кількість кластерів k")
    plots.save(fig, out, "05_rates_regimes.png")
    return labels, segs, colors


def stage_synthesis(frame, segs, out, rng):
    real = frame.reset_index(drop=True)
    model = synthesis.fit_rates(real, segs)
    print("   параметри моделі на ділянках (μ, φ премії і спреду):")
    for seg in model:
        (mu_p, phi_p), (mu_s, phi_s) = seg["premium"], seg["spread"]
        kind = "блукання" if seg["floating"] else "фіксований"
        print(
            f"     {REGIMES[seg['label']]:<18} n={seg['length']:>3}  НБУ {kind:<10}  "
            f"премія μ={mu_p:6.2f} φ={phi_p:5.2f}   спред μ={mu_s:5.2f} φ={phi_s:5.2f}"
        )
    report, samples = synthesis.monte_carlo(
        metrics, real, lambda g: synthesis.synth_rates(model, g), RUNS, rng
    )
    print(f"   адекватність за {RUNS} реалізаціями (90 % інтервал):")
    print(report.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"   у межах: {int(report['в межах'].sum())} з {len(report)} показників")

    paths = [synthesis.synth_rates(model, rng) for _ in range(RUNS)]
    buys = np.array([p["buy"].to_numpy() for p in paths])
    low, high = np.percentile(buys, (5, 95), axis=0)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    top.fill_between(
        frame.index,
        low,
        high,
        color=plots.COLORS[0],
        alpha=0.15,
        lw=0,
        label="модель, 90 % реалізацій",
    )
    for i in range(PATHS_SHOWN):
        top.plot(
            frame.index,
            buys[i],
            lw=0.8,
            color=plots.COLORS[0],
            alpha=0.7,
            drawstyle="steps-post",
            label="модельні реалізації" if i == 0 else None,
        )
    top.plot(
        frame.index,
        frame["buy"],
        color=plots.REAL,
        lw=1.4,
        drawstyle="steps-post",
        label="реальний курс",
    )
    top.set_title("Купівля: реальний ряд і модельні реалізації")
    top.set_ylabel("грн")
    top.legend(fontsize=7)
    sample = paths[0]
    bottom.plot(
        frame.index,
        sample["sell"] - sample["buy"],
        color=plots.COLORS[3],
        lw=0.9,
        drawstyle="steps-post",
        label="спред моделі",
    )
    bottom.plot(
        frame.index,
        frame["sell"] - frame["buy"],
        color=plots.REAL,
        lw=1.0,
        drawstyle="steps-post",
        label="спред реальний",
    )
    bottom.set_ylabel("грн")
    bottom.legend(fontsize=7)
    plots.save(fig, out, "06_rates_synthetic.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 5))
    for ax, name in zip(axes.flat, SHOWN_METRICS, strict=True):
        plots.mc_panel(ax, samples[name], report.loc[name, "реальний"], name)
    fig.suptitle("Показники на модельних реалізаціях і реальне значення (лінія)", fontsize=11)
    plots.save(fig, out, "07_rates_adequacy.png")
    return report


def run(path, out, rng):
    frame, gaps = loaders.rates(path)
    print("\n1.1. Вхідні дані")
    stage_input(frame, gaps, out)
    print("\n1.2. Декомпозиція")
    stage_decomposition(frame, out)
    print("\n1.3. Властивості")
    stage_properties(frame, out)
    print("\n1.4. Кластеризація режимів")
    labels, segs, colors = stage_regimes(frame, out)
    print("\n1.5. Кореляційний аналіз")
    stage_correlation(frame, out, labels, colors)
    print("\n1.6. Синтез моделі та перевірка адекватності")
    return stage_synthesis(frame, segs, out, rng)
