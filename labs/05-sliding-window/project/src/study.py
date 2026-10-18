"""Етапи дослідження одного ряду: властивості, згладжування, прогноз, оптимізація, порівняння."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arima
import evaluation
import learning
import plots
import properties
import smoothing

LAGS = 30
CV_FOLDS = 5
TOP_AIC = 5

MOVING = ["SMA", "WMA", "медіана", "EMA", "ковзний МНК", "сезонне MA"]
AUTOREG = ["ARIMA", "ARIMA (AIC)", "ARIMA (CV)", "ARIMA(1,1,15)"]
LEARNING = ["МНК-поліном", "МНК + гармоніки", "α-β фільтр"]


def naive(train, h, start):
    return np.full(h, train[-1])


def arima_forecaster(order, harmonics=0, drift=False):
    def forecast(train, h, start):
        return arima.forecast(arima.fit(train, order, harmonics, start, drift), h)

    return forecast


def rates_config():
    return {
        "key": "rates",
        "title": "курс купівлі USD",
        "unit": "грн",
        "step": "запис",
        "window": 100,
        "origins": range(0, 49, 12),
        "cv_horizon": 25,
        "lead_step": 4,
        "seasonal": False,
        "expert": {
            "наївний": naive,
            "SMA": lambda tr, h, s: smoothing.forecast_flat(tr, h, 10),
            "WMA": lambda tr, h, s: smoothing.forecast_flat(tr, h, 10, "WMA"),
            "медіана": lambda tr, h, s: smoothing.forecast_flat(tr, h, 10, "median"),
            "EMA": lambda tr, h, s: smoothing.forecast_ema(tr, h, 0.3),
            "ковзний МНК": lambda tr, h, s: smoothing.forecast_line(tr, h, 20),
            "ARIMA": arima_forecaster((1, 1, 1)),
            "МНК-поліном": lambda tr, h, s: learning.forecast_poly(tr, h, 2),
            "α-β фільтр": lambda tr, h, s: learning.forecast_alpha_beta(tr, h, 20),
        },
        "expert_params": {
            "SMA": "w = 10",
            "WMA": "w = 10",
            "медіана": "w = 10",
            "EMA": "α = 0,3",
            "ковзний МНК": "w = 20",
            "ARIMA": "(1, 1, 1)",
            "МНК-поліном": "степінь 2",
            "α-β фільтр": "N = 20",
        },
        "tuned": {
            "SMA": (
                lambda w: lambda tr, h, s: smoothing.forecast_flat(tr, h, w),
                [2, 3, 5, 7, 10, 15, 20, 30, 50],
            ),
            "WMA": (
                lambda w: lambda tr, h, s: smoothing.forecast_flat(tr, h, w, "WMA"),
                [2, 3, 5, 7, 10, 15, 20, 30, 50],
            ),
            "медіана": (
                lambda w: lambda tr, h, s: smoothing.forecast_flat(tr, h, w, "median"),
                [2, 3, 5, 7, 10, 15, 20, 30, 50],
            ),
            "EMA": (
                lambda a: lambda tr, h, s: smoothing.forecast_ema(tr, h, a),
                [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            ),
            "ковзний МНК": (
                lambda w: lambda tr, h, s: smoothing.forecast_line(tr, h, w),
                [5, 10, 20, 30, 50, 75, 100],
            ),
            "МНК-поліном": (
                lambda d: lambda tr, h, s: learning.forecast_poly(tr, h, d),
                [0, 1, 2, 3],
            ),
            "α-β фільтр": (
                lambda n: lambda tr, h, s: learning.forecast_alpha_beta(tr, h, n),
                [5, 10, 20, 30, 50, 100],
            ),
        },
        "arima_space": {
            "p_max": 3,
            "d_values": (0, 1),
            "q_max": 3,
            "harmonics": (0,),
            "drifts": (False, True),
        },
        "baseline": {"ARIMA(1,1,15)": arima_forecaster((1, 1, 15))},
        "shown": ["наївний", "EMA", "ковзний МНК", "ARIMA", "МНК-поліном", "α-β фільтр"],
    }


def temperature_config():
    return {
        "key": "temp",
        "title": "температура, Київ",
        "unit": "°C",
        "step": "доба",
        "window": 730,
        "origins": range(0, 1461, 365),
        "cv_horizon": 182,
        "lead_step": 30,
        "seasonal": True,
        "expert": {
            "наївний": naive,
            "SMA": lambda tr, h, s: smoothing.forecast_flat(tr, h, 30),
            "EMA": lambda tr, h, s: smoothing.forecast_ema(tr, h, 0.1),
            "ковзний МНК": lambda tr, h, s: smoothing.forecast_line(tr, h, 60),
            "сезонне MA": lambda tr, h, s: smoothing.forecast_seasonal(tr, h, 2, 7),
            "ARIMA": arima_forecaster((1, 0, 0), harmonics=3),
            "МНК-поліном": lambda tr, h, s: learning.forecast_poly(tr, h, 2),
            "МНК + гармоніки": lambda tr, h, s: learning.forecast_harmonic(tr, h, 3, s),
            "α-β фільтр": lambda tr, h, s: learning.forecast_alpha_beta(tr, h, 60),
        },
        "expert_params": {
            "SMA": "w = 30",
            "EMA": "α = 0,1",
            "ковзний МНК": "w = 60",
            "сезонне MA": "2 роки, ±7 діб",
            "ARIMA": "3 гармоніки + AR(1)",
            "МНК-поліном": "степінь 2",
            "МНК + гармоніки": "3 гармоніки",
            "α-β фільтр": "N = 60",
        },
        "tuned": {
            "SMA": (
                lambda w: lambda tr, h, s: smoothing.forecast_flat(tr, h, w),
                [7, 15, 30, 60, 90, 180, 365],
            ),
            "EMA": (
                lambda a: lambda tr, h, s: smoothing.forecast_ema(tr, h, a),
                [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            ),
            "ковзний МНК": (
                lambda w: lambda tr, h, s: smoothing.forecast_line(tr, h, w),
                [30, 60, 90, 180, 365],
            ),
            "сезонне MA": (
                lambda p: lambda tr, h, s: smoothing.forecast_seasonal(tr, h, *p),
                list(itertools.product([1, 2], [0, 3, 7, 15, 30])),
            ),
            "МНК-поліном": (
                lambda d: lambda tr, h, s: learning.forecast_poly(tr, h, d),
                [0, 1, 2, 3],
            ),
            "МНК + гармоніки": (
                lambda k: lambda tr, h, s: learning.forecast_harmonic(tr, h, k, s),
                [1, 2, 3, 4],
            ),
            "α-β фільтр": (
                lambda n: lambda tr, h, s: learning.forecast_alpha_beta(tr, h, n),
                [30, 60, 90, 180, 365],
            ),
        },
        "arima_space": {
            "p_max": 2,
            "d_values": (0,),
            "q_max": 2,
            "harmonics": (1, 2, 3, 4),
            "drifts": (False,),
        },
        "baseline": {},
        "shown": ["наївний", "EMA", "сезонне MA", "ARIMA", "МНК-поліном", "МНК + гармоніки"],
    }


def color_of(name):
    order = ["наївний", "SMA", "WMA", "медіана", "EMA", "ковзний МНК", "сезонне MA", "ARIMA"]
    if name in order:
        return plots.COLORS[order.index(name)]
    return {
        "МНК-поліном": plots.INK_SECONDARY,
        "МНК + гармоніки": plots.MUTED,
        "α-β фільтр": "#8f1f1f",
    }.get(name, plots.COLORS[7])


def x_axis(cfg, series):
    return series.index if cfg["seasonal"] else np.arange(len(series))


# --- Етап 1. Властивості ---------------------------------------------------------------


def stage_properties(cfg, series, out):
    y = series.to_numpy()
    if cfg["seasonal"]:
        derived, derived_name = properties.anomalies(y), "аномалія"
    else:
        derived, derived_name = np.diff(y), "приріст"
    rows = {cfg["title"]: properties.describe(y), derived_name: properties.describe(derived)}
    print(pd.DataFrame(rows).T.round(3).to_string())
    st = {cfg["title"]: properties.stationarity(y), derived_name: properties.stationarity(derived)}
    print(pd.DataFrame(st).T.to_string(float_format=lambda v: f"{v:.4f}"))
    acf, pacf, bound = properties.correlograms(derived, LAGS)
    significant = [k for k in range(1, LAGS + 1) if abs(pacf[k]) > bound]
    print(
        f"   {derived_name}: ACF(1) = {acf[1]:.3f}, PACF значущі на лагах {significant[:8]}, "
        f"межа ±{bound:.3f}, p Льюнга–Бокса = {properties.white_noise_p(derived):.2e}"
    )

    fig = plt.figure(figsize=(11, 6))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    ax = fig.add_subplot(grid[0, :])
    ax.plot(x_axis(cfg, series), y, lw=0.7 if cfg["seasonal"] else 1.2, color=plots.COLORS[0])
    ax.set_title(f"Ряд: {cfg['title']}")
    ax.set_ylabel(cfg["unit"])
    lags = np.arange(LAGS + 1)
    for col, (values, title) in enumerate(
        [(acf, f"ACF: {derived_name}"), (pacf, f"PACF: {derived_name}")]
    ):
        ax = fig.add_subplot(grid[1, col])
        ax.bar(lags[1:], values[1:], color=plots.COLORS[0], width=0.6)
        ax.axhspan(-bound, bound, color=plots.MUTED, alpha=0.18, lw=0)
        ax.set_title(title)
        ax.set_xlabel(f"лаг, {cfg['step']}")
    plots.save(
        fig,
        out,
        f"01_{cfg['key']}_properties.png" if cfg["key"] == "rates" else "02_temp_properties.png",
    )
    return {"acf1": acf[1]}


# --- Етап 2. Згладжування ковзним вікном ------------------------------------------------


def rolling_line(y, w):
    out = np.empty(len(y))
    for t in range(len(y)):
        lo = max(0, t - w + 1)
        out[t] = smoothing.local_line(y[lo : t + 1], t + 1 - lo)[0] if t - lo >= 1 else y[t]
    return out


def stage_smoothing(cfg, series, out):
    y = series.to_numpy()
    w = 10 if not cfg["seasonal"] else 30
    smoothed = {
        f"SMA, w = {w}": smoothing.smooth(y, w),
        f"WMA, w = {w}": smoothing.smooth(y, w, "WMA"),
        f"медіана, w = {w}": smoothing.smooth(y, w, "median"),
        f"EMA, α = {2 / (w + 1):.2f}": smoothing.ema(y, 2 / (w + 1)),
        f"ковзний МНК, w = {2 * w}": rolling_line(y, 2 * w),
    }
    rows = {}
    for name, s in smoothed.items():
        r = y - s
        rows[name] = {
            "σ відхилення": r[w:].std(),
            "зміщення": r[w:].mean(),
            "гладкість": np.std(np.diff(s[w:])) / np.std(np.diff(y[w:])),
        }
    print(f"   каузальне згладжування, σ ряду приростів = {np.std(np.diff(y)):.3f} {cfg['unit']}:")
    print(pd.DataFrame(rows).T.round(3).to_string())

    fig, ax = plt.subplots(figsize=(11, 4.2))
    sl = slice(120, 260) if not cfg["seasonal"] else slice(730, 1095)
    x = x_axis(cfg, series)[sl]
    ax.plot(
        x,
        y[sl],
        color=plots.REAL,
        lw=1.0,
        label="ряд",
        drawstyle="steps-post" if not cfg["seasonal"] else "default",
    )
    for (name, s), color in zip(smoothed.items(), plots.COLORS[1:], strict=False):
        ax.plot(x, s[sl], lw=1.4, color=color, label=name)
    ax.set_title(f"Каузальне згладжування ковзним вікном: {cfg['title']}")
    ax.set_ylabel(cfg["unit"])
    ax.legend(ncol=3, fontsize=7)
    plots.save(
        fig, out, "03_rates_smoothing.png" if cfg["key"] == "rates" else "04_temp_smoothing.png"
    )
    return rows


# --- Етап 3. Прогноз з експертними параметрами ------------------------------------------


def horizon_table(results, metric="RMSE"):
    return pd.DataFrame({name: agg[metric] for name, (agg, _, _) in results.items()}).T


def stage_forecast(cfg, series, out):
    y = series.to_numpy()
    results = {
        name: evaluation.backtest(y, cfg["window"], cfg["origins"], f)
        for name, f in cfg["expert"].items()
    }
    table = horizon_table(results)
    table["MASE, 2W"] = horizon_table(results, "MASE")[2.0]
    if not cfg["seasonal"]:
        table["MAPE, 2W"] = horizon_table(results, "MAPE")[2.0]
    print(f"   W = {cfg['window']}, стартів {len(cfg['origins'])}; RMSE за горизонтами (частки W):")
    print(table.round(3).to_string())
    cover = results["ARIMA"][0].get("покриття")
    if cover is not None:
        print(
            "   покриття 90 % інтервалу ARIMA: "
            + ", ".join(f"{h}W – {c:.2f}" for h, c in cover.items())
        )

    s = list(cfg["origins"])[-1]
    W = cfg["window"]
    steps = 2 * W
    x = x_axis(cfg, series)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(x[s : s + W + steps], y[s : s + W + steps], color=plots.REAL, lw=0.9, label="ряд")
    for name in cfg["shown"]:
        _, _, paths = results[name]
        mean, interval = next((m, i) for o, m, i in paths if o == s)
        ax.plot(x[s + W : s + W + steps], mean, lw=1.6, color=color_of(name), label=name)
        if interval is not None:
            ax.fill_between(
                x[s + W : s + W + steps],
                interval[:, 0],
                interval[:, 1],
                color=color_of(name),
                alpha=0.12,
                lw=0,
            )
    for h in evaluation.HORIZONS:
        ax.axvline(x[min(s + W + int(h * W) - 1, len(x) - 1)], color=plots.AXIS, lw=0.8, ls=":")
    ax.axvline(x[s + W - 1], color=plots.INK_SECONDARY, lw=1.0)
    ax.set_title(f"Екстраполяція на 0,5–2 інтервали спостереження: {cfg['title']}")
    ax.set_ylabel(cfg["unit"])
    if not cfg["seasonal"]:
        ax.set_xlabel("номер запису")
    # поліном і ковзний МНК на довгому горизонті розходяться на порядки, тож масштаб – за даними
    seen = y[s : s + W + steps]
    margin = 0.35 * (seen.max() - seen.min())
    ax.set_ylim(seen.min() - margin, seen.max() + margin)
    ax.legend(ncol=4, fontsize=7, loc="upper left")
    plots.save(
        fig, out, "05_rates_forecast.png" if cfg["key"] == "rates" else "06_temp_forecast.png"
    )
    return results, table


# --- Етап 4. Якість моделі ARIMA --------------------------------------------------------


def describe_model(label, model):
    params = pd.Series(model.params, index=model.param_names)
    print(f"   {label}")
    print("     параметри: " + ", ".join(f"{k} = {v:.4f}" for k, v in params.items()))
    print(
        f"     AIC = {model.aic:.2f}, BIC = {model.bic:.2f}, "
        f"p Льюнга–Бокса залишків = {arima.ljung_box(model):.3f}"
    )


def stage_arima(cfg, series):
    """Якість моделей груп 1 і 2 на першому вікні: параметри, AIC, BIC, білість залишків."""
    y = series.to_numpy()
    s = next(iter(cfg["origins"]))
    train = y[s : s + cfg["window"]]
    order, harmonics = ((1, 0, 0), 3) if cfg["seasonal"] else ((1, 1, 1), 0)
    first = arima.fit(train, order, harmonics, s)
    best_order, best_k, drift, rows = arima.select(train, start=s, **cfg["arima_space"])
    second = arima.fit(train, best_order, best_k, s, drift)
    print(f"   перебрано {len(rows)} моделей; п'ять найкращих за AIC (AIC, BIC, p Льюнга–Бокса):")
    for aic, bic, lb, o, k, d in rows[:5]:
        print(f"     {describe_choice((o, k, d)):<26} {aic:9.2f} {bic:9.2f} {lb:7.3f}")
    describe_model(f"група 1: {cfg['expert_params']['ARIMA']}", first)
    describe_model(
        f"група 2 (AIC + Льюнг–Бокс): {describe_choice((best_order, best_k, drift))}", second
    )
    return {
        "група 1": first.resid[max(first.model.k_diff, 1) :],
        "група 2": second.resid[max(second.model.k_diff, 1) :],
    }


# --- Етап 5. Оптимізація параметрів ----------------------------------------------------


def tuned_forecaster(cfg, name, chosen, curves):
    make, candidates = cfg["tuned"][name]

    def forecast(train, h, start):
        best, scores = evaluation.tune(train, make, candidates, cfg["cv_horizon"], CV_FOLDS, start)
        chosen.append(best)
        curves.append(scores)
        return make(best)(train, h, start)

    return forecast


def arima_aic(cfg, chosen):
    space = cfg["arima_space"]

    def forecast(train, h, start):
        order, k, drift, _ = arima.select(train, start=start, **space)
        chosen.append((order, k, drift))
        return arima.forecast(arima.fit(train, order, k, start, drift), h)

    return forecast


def arima_cv(cfg, chosen):
    """Серед TOP_AIC найкращих за AIC моделей – та, що має найменшу CV-похибку на горизонті."""
    space = cfg["arima_space"]

    def forecast(train, h, start):
        rows = arima.grid(train, start=start, **space)[:TOP_AIC]
        scores = {}
        for _, _, _, order, k, drift in rows:
            f = arima_forecaster(order, k, drift)
            scores[(order, k, drift)] = evaluation.cv_score(
                train, f, cfg["cv_horizon"], CV_FOLDS, start
            )
        best = min(scores, key=lambda c: scores[c])
        chosen.append(best)
        return arima.forecast(arima.fit(train, best[0], best[1], start, best[2]), h)

    return forecast


def describe_choice(choice):
    if isinstance(choice, tuple) and len(choice) == 3 and isinstance(choice[0], tuple):
        order, k, drift = choice
        text = f"({order[0]},{order[1]},{order[2]})"
        if k:
            text = f"{k} гарм. + ARMA({order[0]},{order[2]})"
        return text + (" + дрейф" if drift else "")
    if isinstance(choice, tuple):
        return f"{choice[0]} р., ±{choice[1]}"
    return f"{choice:g}"


def stage_tuning(cfg, series, expert_results, out):
    y = series.to_numpy()
    tuned, chosen, curves = {}, {}, {}
    for name in cfg["tuned"]:
        chosen[name], curves[name] = [], []
        tuned[name] = evaluation.backtest(
            y,
            cfg["window"],
            cfg["origins"],
            tuned_forecaster(cfg, name, chosen[name], curves[name]),
        )
    for name, factory in [("ARIMA (AIC)", arima_aic), ("ARIMA (CV)", arima_cv)]:
        chosen[name] = []
        tuned[name] = evaluation.backtest(
            y, cfg["window"], cfg["origins"], factory(cfg, chosen[name])
        )
    for name, f in cfg["baseline"].items():
        tuned[name] = evaluation.backtest(y, cfg["window"], cfg["origins"], f)

    print("   обрані параметри за стартами:")
    for name, values in chosen.items():
        print(f"     {name:<16} " + "; ".join(describe_choice(v) for v in values))

    rows = {}
    for name, (agg, table, _) in tuned.items():
        base_name = "ARIMA" if name.startswith("ARIMA") else name
        base = expert_results[base_name][1]
        rmse_tuned = table[table["горизонт"] == 2.0]["RMSE"].to_numpy()
        rmse_base = base[base["горизонт"] == 2.0]["RMSE"].to_numpy()
        rows[name] = {
            "група 1, 1W": expert_results[base_name][0].loc[1.0, "RMSE"],
            "група 2, 1W": agg.loc[1.0, "RMSE"],
            "група 1, 2W": expert_results[base_name][0].loc[2.0, "RMSE"],
            "група 2, 2W": agg.loc[2.0, "RMSE"],
            "кращий старт": f"{int((rmse_tuned < rmse_base).sum())}/{len(rmse_base)}",
        }
    gain = pd.DataFrame(rows).T
    print("   RMSE до і після оптимізації (база для ARIMA – модель групи 1):")
    print(gain.to_string(float_format=lambda v: f"{v:.3f}"))
    return tuned, chosen, curves, gain


# --- Етап 6. Порівняння класів методів ---------------------------------------------------


def best_variants(cfg, expert_results, tuned):
    """Для кожного методу – варіант з меншою RMSE на 2W: з параметрами групи 1 чи групи 2."""
    out = {"наївний": expert_results["наївний"]}
    names = [n for n in [*MOVING, "ARIMA", *LEARNING] if n in expert_results]
    for name in names:
        candidates = {name: expert_results[name]}
        if name in tuned:
            candidates[name + "*"] = tuned[name]
        if name == "ARIMA":
            for variant in ("ARIMA (AIC)", "ARIMA (CV)"):
                candidates[variant] = tuned[variant]
        label, result = min(candidates.items(), key=lambda kv: kv[1][0].loc[2.0, "RMSE"])
        out[label] = result
    return out


def stage_comparison(cfg, series, expert_results, tuned):
    best = best_variants(cfg, expert_results, tuned)
    rows = {}
    for label, (agg, _, _) in best.items():
        base = label.rstrip("*").split(" (")[0]
        cls = (
            "еталон"
            if base == "наївний"
            else (
                "ковзне вікно"
                if base in MOVING
                else ("автокореляційний" if base == "ARIMA" else "статистичне навчання")
            )
        )
        rows[label] = (
            {"клас": cls}
            | {f"RMSE {h}W": agg.loc[h, "RMSE"] for h in evaluation.HORIZONS}
            | {
                "MAE 2W": agg.loc[2.0, "MAE"],
                "зміщення 2W": agg.loc[2.0, "зміщення"],
                "MASE 2W": agg.loc[2.0, "MASE"],
            }
        )
    kpi = pd.DataFrame(rows).T
    kpi["ранг"] = kpi["RMSE 2.0W"].astype(float).rank().astype(int)
    print("   найкращий варіант кожного методу (* – параметри групи 2):")
    print(kpi.sort_values("ранг").to_string(float_format=lambda v: f"{v:.3f}"))
    return kpi, best


def lead_errors(cfg, series, methods, leads):
    """RMSE за кроком прогнозу k (1..2W) з частими стартами: де саме метод виграє."""
    y = series.to_numpy()
    W = cfg["window"]
    starts = range(0, len(y) - W - max(leads), cfg["lead_step"])
    out = {}
    for name, f in methods.items():
        sq = np.zeros(len(leads))
        for s in starts:
            mean = evaluation.point(f(y[s : s + W], max(leads), s))
            sq += (np.asarray(mean)[np.array(leads) - 1] - y[s + W + np.array(leads) - 1]) ** 2
        out[name] = np.sqrt(sq / len(starts))
    return pd.DataFrame(out, index=leads).T, len(starts)


def run(cfg, series, out):
    print("\nЕтап 1. Властивості")
    stage_properties(cfg, series, out)
    print("\nЕтап 2. Згладжування ковзним вікном")
    stage_smoothing(cfg, series, out)
    print("\nЕтап 3. Екстраполяція з параметрами групи 1")
    expert_results, table = stage_forecast(cfg, series, out)
    print("\nЕтап 4. Якість моделі ARIMA")
    resid = stage_arima(cfg, series)
    print("\nЕтап 5. Оптимізація параметрів (група 2)")
    tuned, chosen, curves, gain = stage_tuning(cfg, series, expert_results, out)
    print("\nЕтап 6. Порівняння ковзного вікна і статистичного навчання")
    kpi, best = stage_comparison(cfg, series, expert_results, tuned)
    return {
        "cfg": cfg,
        "series": series,
        "expert": expert_results,
        "table": table,
        "resid": resid,
        "tuned": tuned,
        "chosen": chosen,
        "curves": curves,
        "gain": gain,
        "kpi": kpi,
        "best": best,
    }
