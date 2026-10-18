"""Графіки роботи: ряди, декомпозиція, структура мереж, навчання, прогнози і KPI."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import evaluation
import networks
import plots
import strategies
import study
import verify

KIND_COLORS = dict(zip(networks.KINDS, plots.COLORS[:3], strict=True))
STRATEGY_COLORS = dict(
    zip(strategies.STRATEGIES, (plots.COLORS[4], plots.COLORS[3], plots.COLORS[6]), strict=True)
)
ENSEMBLE = plots.COLORS[6]
DETERMINISTIC = plots.COLORS[7]


def legend(ax, **kw):
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, **kw)


def grid(cfgs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    return fig, list(zip(axes.flat, cfgs, strict=True))


def tail(cfg):
    """Скільки останніх точок вікна показувати: для добової температури – рік."""
    return 365 if cfg["window"] > 400 else cfg["window"]


def series(cfgs, out):
    fig, cells = grid(cfgs)
    for ax, cfg in cells:
        y, w = cfg["series"], cfg["window"]
        ax.plot(y, color=plots.REAL, lw=0.6 if len(y) > 1000 else 0.9, label="ряд")
        if cfg["truth"] is not None:
            ax.plot(cfg["truth"], color=plots.BAD, lw=1.6, label="істинний тренд")
        for i, s in enumerate(cfg["origins"]):
            ax.axvspan(
                s,
                s + w,
                color=plots.COLORS[i],
                alpha=0.08,
                lw=0,
                label=f"вікно спостереження {i + 1}, W = {w}",
            )
        ax.set_title(cfg["title"])
        legend(ax, loc="upper left")
    plots.save(fig, out, "01_series.png")


def decomposition(cfgs, props, out):
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.6), gridspec_kw={"height_ratios": [1.4, 1]})
    for j, cfg in enumerate(cfgs):
        p = props[cfg["key"]]
        w = cfg["window"]
        top, bottom = axes[0, j], axes[1, j]
        top.plot(cfg["series"][:w], color=plots.REAL, lw=0.7, alpha=0.7, label="ряд")
        season = f", s = {p['period']}" if p["period"] else ""
        top.plot(p["d"], color=DETERMINISTIC, lw=1.6, label=f"D(t): степінь {p['degree']}{season}")
        top.set_title(cfg["title"], fontsize=10)
        legend(top, loc="upper left")
        bottom.plot(p["resid"], color=plots.COLORS[0], lw=0.6)
        bottom.axhline(0, color=plots.AXIS, lw=1)
        bottom.set_title(f"залишок r(t), R² D(t) = {p['R²']:.2f}".replace(".", ","), fontsize=9)
    plots.save(fig, out, "02_decomposition.png")


def layer_text(layer):
    name = type(layer).__name__
    shape = "×".join(str(d) for d in layer.output.shape[1:])
    if name == "InputLayer":
        return "Input", f"вхід {shape}", None
    if name == "Dense":
        act = layer.get_config()["activation"]
        act = "" if act == "linear" else f", {act}"
        return f"Dense {layer.units}{act}", f"вихід {shape}", layer.count_params()
    if name == "LSTM":
        return f"LSTM {layer.units}, tanh", f"вихід {shape}", layer.count_params()
    if name == "Conv1D":
        return (
            f"Conv1D {layer.filters}, k = {layer.kernel_size[0]}, relu",
            f"вихід {shape}",
            layer.count_params(),
        )
    return name, f"вихід {shape}", None


def architecture(models, chosen, out):
    """Схеми обраних мереж: шари зверху вниз з розмірністю виходу і кількістю параметрів."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    tallest = max(len(m.layers) for m in models.values())
    for ax, (kind, model) in zip(axes, models.items(), strict=True):
        layers = model.layers
        n = len(layers)
        ax.set_xlim(0, 1)
        ax.set_ylim(n - tallest - 0.3, n)
        ax.axis("off")
        color = KIND_COLORS[kind]
        for i, layer in enumerate(layers):
            y = n - 1 - i
            title, shape, params = layer_text(layer)
            box = FancyBboxPatch(
                (0.08, y + 0.12),
                0.84,
                0.62,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=color if i not in (0, n - 1) else "white",
                edgecolor=color,
                alpha=0.9 if i not in (0, n - 1) else 1.0,
                lw=1.4,
            )
            ax.add_patch(box)
            ink = "white" if i not in (0, n - 1) else plots.INK
            ax.text(0.5, y + 0.52, title, ha="center", va="center", fontsize=9, color=ink)
            sub = shape if params is None else f"{shape}; параметрів {params}"
            ax.text(0.5, y + 0.3, sub, ha="center", va="center", fontsize=7.5, color=ink)
            if i < n - 1:
                ax.annotate(
                    "",
                    xy=(0.5, y - 0.26 + 0.12),
                    xytext=(0.5, y + 0.12),
                    arrowprops={"arrowstyle": "-|>", "color": plots.MUTED, "lw": 1.2},
                )
        c = chosen[kind]
        ax.set_title(
            f"{kind}: {c['розмір']}, втрати {c['втрати']}\nпараметрів {model.count_params()}",
            fontsize=10,
        )
    plots.save(fig, out, "03_architecture.png")


def structure(struct, cfgs, out):
    table = struct["table"]
    keys = [c["key"] for c in cfgs]
    rows, labels = [], []
    for (kind, size, loss), part in table.groupby(["мережа", "розмір", "втрати"], sort=False):
        part = part.set_index("ряд").loc[keys]
        rows.append(part)
        labels.append(f"{kind} · {size} · {loss}")
    chosen = {f"{k} · {v['розмір']} · {v['втрати']}" for k, v in struct["chosen"].items()}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.55]})
    for ax, ind in zip(axes[:2], study.INDICATORS[:2], strict=True):
        matrix = np.array([r[ind].to_numpy() for r in rows])
        plots.heatmap(
            ax,
            matrix,
            labels,
            [c["short"] for c in cfgs],
            cmap=plots.SEQUENTIAL,
            vmin=matrix.min(),
            vmax=matrix.max(),
            fmt="{:.3f}",
        )
        ax.set_title(f"{ind}, стандартизований залишок")
        for tick in ax.get_yticklabels():
            if tick.get_text() in chosen:
                tick.set_fontweight("bold")
                tick.set_color(plots.GOOD)
    score = struct["summary"]["ранг"].to_numpy()
    colors = [plots.GOOD if lbl in chosen else plots.COLORS[0] for lbl in labels]
    axes[2].barh(range(len(labels)), score, color=colors, alpha=0.85)
    axes[2].set_yticks(range(len(labels)), [""] * len(labels))
    axes[2].invert_yaxis()
    axes[2].set_xlabel("середній ранг (менше – краще)")
    axes[2].set_title("за трьома індикаторами")
    plots.save(fig, out, "04_structure.png")


def learning(cfgs, runs, out):
    fig, cells = grid(cfgs)
    for ax, cfg in cells:
        s = cfg["origins"][0]
        fits = runs[cfg["key"]]["fits"][s]
        for kind in networks.KINDS:
            r = fits[study.method_name(kind, "гібрид")]
            epochs = np.arange(1, len(r["loss"]) + 1)
            color = KIND_COLORS[kind]
            ax.plot(epochs, r["loss"], color=color, lw=1, ls="--", alpha=0.7)
            ax.plot(epochs, r["val_loss"], color=color, lw=1.6, label=f"{kind}, зуп. {r['епох']}")
            ax.scatter(r["епох"], r["val_loss"][r["епох"] - 1], color=color, s=18, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("епоха (пунктир – навчальна, суцільна – валідаційна)")
        ax.set_ylabel("втрати")
        ax.set_title(f"{cfg['title']}, гібрид")
        legend(ax, loc="upper right")
    plots.save(fig, out, "05_learning.png")


def fit(cfgs, runs, out):
    fig, cells = grid(cfgs)
    for ax, cfg in cells:
        s, w = cfg["origins"][0], cfg["window"]
        k = tail(cfg)
        x = np.arange(s + w - k, s + w)
        ax.plot(
            x, cfg["series"][s + w - k : s + w], color=plots.REAL, lw=0.8, alpha=0.6, label="ряд"
        )
        if cfg["truth"] is not None:
            ax.plot(x, cfg["truth"][s + w - k : s + w], color=plots.BAD, lw=1.4, label="тренд")
        fits = runs[cfg["key"]]["fits"][s]
        for kind in networks.KINDS:
            fitted = fits[study.method_name(kind, "гібрид")]["fitted"][w - k :]
            ax.plot(x, fitted, color=KIND_COLORS[kind], lw=1.1, label=kind)
        ax.set_title(cfg["title"])
        legend(ax, loc="upper left", ncol=2)
    plots.save(fig, out, "06_fit.png")


def forecast_panel(ax, cfg, run, s, names, colors, styles=None):
    y, w = cfg["series"], cfg["window"]
    steps = int(max(evaluation.HORIZONS) * w)
    k = min(tail(cfg), w)
    x0 = s + w - k
    end = min(len(y), s + w + steps)
    ax.plot(np.arange(x0, end), y[x0:end], color=plots.REAL, lw=0.7, alpha=0.7, label="ряд")
    ax.axvline(s + w, color=plots.AXIS, lw=1)
    ax.axvline(s + w + steps // 2, color=plots.AXIS, lw=0.8, ls=":")
    for name in names:
        style = (styles or {}).get(name, {"lw": 1.4})
        ax.plot(
            np.arange(s + w, s + w + steps),
            run["paths"][s][name],
            color=colors[name],
            label=name,
            **style,
        )
    ax.set_title(f"{cfg['title']}, старт {s}")


def strategies_plot(cfgs, runs, out):
    fig, cells = grid(cfgs)
    names = [f"ансамбль, {st}" for st in strategies.STRATEGIES] + ["D(t)"]
    colors = {f"ансамбль, {st}": c for st, c in STRATEGY_COLORS.items()} | {"D(t)": DETERMINISTIC}
    styles = {"D(t)": {"lw": 1.2, "ls": "--"}, "ансамбль, гібрид": {"lw": 2.2}}
    for ax, cfg in cells:
        forecast_panel(ax, cfg, runs[cfg["key"]], cfg["origins"][0], names, colors, styles)
        legend(ax, loc="upper left", fontsize=7)
    plots.save(fig, out, "07_strategies.png")


def forecasts(cfgs, runs, out):
    fig, cells = grid(cfgs)
    names = [study.method_name(k, "гібрид") for k in networks.KINDS] + ["ансамбль, гібрид"]
    colors = {study.method_name(k, "гібрид"): c for k, c in KIND_COLORS.items()}
    colors["ансамбль, гібрид"] = ENSEMBLE
    styles = {"ансамбль, гібрид": {"lw": 2.2, "ls": "--"}}
    for ax, cfg in cells:
        forecast_panel(ax, cfg, runs[cfg["key"]], cfg["origins"][-1], names, colors, styles)
        legend(ax, loc="upper left", fontsize=7)
    plots.save(fig, out, "08_forecasts.png")


def kpi_heatmap(cfgs, kpi, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.6))
    names = [m for m in study.methods() if m != "наївний"]
    for ax, h in zip(axes, evaluation.HORIZONS, strict=True):
        matrix = []
        for cfg in cfgs:
            res = kpi[cfg["key"]]
            base = res["наївний"]["summary"].loc[h, "RMSE"]
            matrix.append([res[m]["summary"].loc[h, "RMSE"] / base for m in names])
        matrix = np.array(matrix).T
        plots.heatmap(ax, np.log2(matrix), names, [c["short"] for c in cfgs], vmin=-2, vmax=2)
        for text, v in zip(ax.texts, matrix.flat, strict=True):
            text.set_text(f"{v:.2f}")
        ax.set_title(f"RMSE / RMSE наївного, горизонт {h:g}W".replace(".", ","))
    plots.save(fig, out, "09_kpi_heatmap.png")


def verification(checks, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    results = checks["results"]
    t = np.arange(2 * verify.N)
    trend = next(x for x in checks["tasks"] if x["meta"][:2] == ("trend", 0))["y"]
    ax = axes[0]
    ax.plot(t[: verify.N], trend, color=plots.REAL, lw=0.7, alpha=0.7, label="ряд")
    ax.plot(t, verify.SLOPE * t, color=plots.BAD, lw=1.2, ls="--", label="істинний тренд")
    for st in strategies.STRATEGIES:
        f = results[f"verify|trend|0|LSTM|{st}"]["forecast"]
        ax.plot(t[verify.N :], f, color=STRATEGY_COLORS[st], lw=1.6, label=f"LSTM, {st}")
    ax.axvline(verify.N, color=plots.AXIS, lw=1)
    ax.set_title(
        f"лінійний тренд {verify.SLOPE}·t: екстраполяція на N = {verify.N}".replace(".", ",")
    )
    legend(ax, loc="upper left")
    ax = axes[1]
    sine = next(x for x in checks["tasks"] if x["meta"][:2] == ("sine", 0))["y"]
    x = np.arange(verify.N - 100, verify.N + 100)
    truth = verify.AMPLITUDE * np.sin(2 * np.pi * x / verify.PERIOD)
    ax.plot(x[:100], sine[-100:], color=plots.REAL, lw=0.8, alpha=0.7, label="ряд")
    ax.plot(x, truth, color=plots.BAD, lw=1, ls="--", label="сигнал")
    for kind in networks.KINDS:
        f = results[f"verify|sine|0|{kind}|рівні"]["forecast"][:100]
        ax.plot(x[100:], f, color=KIND_COLORS[kind], lw=1.4, label=kind)
    ax.axvline(verify.N, color=plots.AXIS, lw=1)
    ax.set_title(f"синусоїда, період {verify.PERIOD}: рекурсія на 4 періоди")
    legend(ax, loc="lower left", ncol=2)
    plots.save(fig, out, "10_verification.png")


def horizon_curves(cfgs, runs, out):
    """RMSE на кожному кроці прогнозу, усереднена по стартах і згладжена ковзним середнім."""
    fig, cells = grid(cfgs)
    names = (
        ["наївний", "D(t)"] + [f"ансамбль, {st}" for st in strategies.STRATEGIES] + [study.DAMPED]
    )
    colors = {"наївний": plots.MUTED, "D(t)": DETERMINISTIC, study.DAMPED: plots.INK}
    colors |= {f"ансамбль, {st}": c for st, c in STRATEGY_COLORS.items()}
    for ax, cfg in cells:
        y, w = cfg["series"], cfg["window"]
        steps = int(max(evaluation.HORIZONS) * w)
        h = np.arange(1, steps + 1)
        k = max(1, steps // 20)
        for name in names:
            sq = [
                (runs[cfg["key"]]["paths"][s][name] - y[s + w : s + w + steps]) ** 2
                for s in cfg["origins"]
            ]
            rmse = np.sqrt(np.convolve(np.mean(sq, axis=0), np.ones(k) / k, mode="same"))
            style = {"lw": 2.0, "ls": "--"} if name == study.DAMPED else {"lw": 1.3}
            ax.plot(h, rmse, color=colors[name], label=name, **style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("крок прогнозу h")
        ax.set_ylabel("RMSE")
        ax.set_title(cfg["title"])
        legend(ax, loc="upper left", fontsize=7)
    plots.save(fig, out, "11_horizon.png")
