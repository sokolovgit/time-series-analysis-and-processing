"""Дослідження одного ряду: властивості, група 1 (експоненційні моделі), група 2 (регресійні),
група 3 (порівняння сімей і вибір між ними)."""

import numpy as np
import pandas as pd

import evaluation
import exponential
import features
import regression
import selector

METHODS = [
    "наївний",
    "експ. (AIC)",
    "експ. (CV)",
    "регресія (CV)",
    "комбінація",
    "гібрид",
    "дерево",
]


def configs(quadratic, linear, rates, temperature):
    """Ряди і параметри вікон. Горизонт CV-вибору всередині вікна – чверть W."""
    n_syn = len(quadratic[0])
    return [
        {
            "key": "quadratic",
            "short": "1. квадратичний",
            "title": "квадратичний тренд + нормальна похибка",
            "series": quadratic[0],
            "truth": quadratic[1],
            "window": 200,
            "origins": range(0, n_syn - 400 + 1, 50),
            "period": None,
        },
        {
            "key": "linear",
            "short": "2. лінійний",
            "title": "лінійний тренд + експоненційна похибка",
            "series": linear[0],
            "truth": linear[1],
            "window": 200,
            "origins": range(0, n_syn - 400 + 1, 50),
            "period": None,
        },
        {
            "key": "rates",
            "short": "3. курс",
            "title": "курс купівлі USD, Ощадбанк",
            "series": np.asarray(rates, dtype=float),
            "truth": None,
            "window": 100,
            "origins": range(0, len(rates) - 200 + 1, 37),
            "period": None,
        },
        {
            "key": "temperature",
            "short": "4. температура",
            "title": "температура Києва, доба",
            "series": np.asarray(temperature, dtype=float),
            "truth": None,
            "window": 1095,
            "origins": range(0, len(temperature) - 2 * 1095 + 1, 365),
            "period": None,
        },
    ]


def table(rows):
    return pd.DataFrame(rows).T


def stage_properties(cfg):
    y, w = cfg["series"], cfg["window"]
    first = y[:w]
    season, share = features.period(first, cfg["period"])
    print(f"   n = {len(y)}, вікно W = {w}, стартів {len(cfg['origins'])}")
    stats = features.describe(y) | features.stationarity(first)
    print(
        "   "
        + ", ".join(
            f"{k} = {v:.3f}" if isinstance(v, float) else f"{k} = {v}" for k, v in stats.items()
        )
    )
    found, found_share = features.period(first)
    print(
        f"   періодограма першого вікна: період {found or 'не виявлено'}"
        f" (частка піку {found_share:.3f}); у моделях s = {season or '–'}"
    )
    feats = features.selector_features(first, cfg["period"])
    print("   ознаки для селектора: " + ", ".join(f"{k} = {v:.3f}" for k, v in feats.items()))
    return {"season": season, "share": share, "features": feats, "detected": found}


def stage_exponential(cfg, season):
    """Група 1: усі експоненційні моделі на першому вікні, вибір за AIC і за CV."""
    y, w = cfg["series"], cfg["window"]
    first = y[:w]
    best, rows = exponential.select(first, season)
    out = {}
    for name, row in rows.items():
        p = exponential.params(row["result"])
        out[name] = {"AIC": row["AIC"], "СКВ на крок": row["СКВ на крок"]} | p
        if cfg["truth"] is not None:
            fitted = row["result"].fittedvalues
            out[name]["СКВ до тренду"] = np.sqrt(np.mean((fitted - cfg["truth"][:w]) ** 2))
    frame = table(out)
    print(f"   кандидати на першому вікні, обрано за AIC: {best}")
    print(frame.round(3).to_string())
    cv_best, alpha, structures, alphas = exponential.select_cv(first, season, w // 4)
    print(f"   CV-вибір на горизонті W/4 = {w // 4}: {cv_best}, α = {fmt_alpha(alpha)}")
    print("   CV RMSE структур: " + ", ".join(f"{k} {v:.3f}" for k, v in structures.items()))
    print("   CV RMSE за α: " + ", ".join(f"{fmt_alpha(k)} {v:.3f}" for k, v in alphas.items()))
    return {
        "table": frame,
        "best": best,
        "result": rows[best]["result"],
        "cv_best": cv_best,
        "alpha": alpha,
        "structures": structures,
        "alphas": alphas,
    }


def fmt_alpha(alpha):
    return "опт." if alpha is None else f"{alpha:g}"


def stage_regression(cfg, season):
    """Група 2: регресійні моделі на першому вікні, вибір за CV."""
    y, w = cfg["series"], cfg["window"]
    first = y[:w]
    best, rows = regression.select(first, w // 4, season)
    out = {}
    for name, row in rows.items():
        kind, kw = row["spec"]
        _, fitted = regression.run(kind, first, 1, kw)
        out[name] = {
            "параметрів": row["параметрів"],
            "CV RMSE": row["CV RMSE"],
            "SE": row["SE"],
            "СКВ підгонки": row["СКВ підгонки"],
        }
        if cfg["truth"] is not None:
            out[name]["СКВ до тренду"] = np.sqrt(np.mean((fitted - cfg["truth"][:w]) ** 2))
    frame = table(out)
    print(f"   кандидати на першому вікні, обрано за CV з правилом 1-SE: {best}")
    print(frame.round(3).to_string())
    kind, kw = rows[best]["spec"]
    return {"table": frame, "best": best, "fitted": regression.run(kind, first, 1, kw)[1]}


def forecasts_at(cfg, train, steps, tree, k):
    """Прогнози всіх методів з одного старту; експоненційна і регресійна сім'ї рахуються раз."""
    season, _ = features.period(train, cfg["period"])
    fam = selector.family_forecasts(train, steps, season)
    f = fam["forecast"]
    choice = selector.hybrid_choice(*fam["CV"].values(), k).item()
    pick = selector.choose(tree, features.selector_features(train, cfg["period"]))
    return {
        "наївний": np.full(steps, train[-1]),
        "експ. (AIC)": exponential.forecast_auto(train, steps, season),
        "експ. (CV)": f[selector.FAMILIES[0]],
        "регресія (CV)": f[selector.FAMILIES[1]],
        "комбінація": f["комбінація"],
        "гібрид": f[choice],
        "дерево": f[pick],
    } | {"_вибір": (choice, pick, fam["моделі"])}


def stage_forecast(cfg, tree, k):
    """Walk-forward перевірка всіх методів на горизонтах 0,5W і W."""
    y, w = cfg["series"], cfg["window"]
    steps = int(max(evaluation.HORIZONS) * w)
    cache = {}
    for s in cfg["origins"]:
        cache[s] = forecasts_at(cfg, y[s : s + w], steps, tree, k)
        choice, pick, models = cache[s]["_вибір"]
        print(
            f"   старт {s}: експ. – {models[selector.FAMILIES[0]]}, "
            f"регресія – {models[selector.FAMILIES[1]]}; гібрид → {choice}, дерево → {pick}"
        )
    results = {}
    for name in METHODS:
        summary, _, paths = evaluation.backtest(
            y, w, cfg["origins"], lambda tr, h, s, n=name: cache[s][n][:h]
        )
        results[name] = {"summary": summary, "paths": paths}
    for h in evaluation.HORIZONS:
        frame = pd.DataFrame({m: r["summary"].loc[h] for m, r in results.items()}).T
        print(f"   KPI прогнозу, горизонт {h:g}W = {int(h * w)} кроків:")
        print(frame.round(3).to_string())
    return results


def run(cfg, tree, k):
    props = stage_properties(cfg)
    print("\n   ГРУПА 1. Експоненційне згладжування")
    exp = stage_exponential(cfg, props["season"])
    print("\n   ГРУПА 2. Регресійні моделі")
    reg = stage_regression(cfg, props["season"])
    print("\n   ГРУПА 3. Порівняння сімей і вибір")
    forecast = stage_forecast(cfg, tree, k)
    return {"cfg": cfg, "props": props, "exp": exp, "reg": reg, "forecast": forecast}
