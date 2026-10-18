"""Дослідження: властивості рядів, синтез структури мереж, прогноз у межах і поза вікном."""

import numpy as np
import pandas as pd

import evaluation
import features
import networks
import runner
import strategies

SEEDS = (0, 1)
HOLDOUT = 4
DAMPED = "гібрид, вага ACF"
INDICATORS = ("СКВ на крок", "СКВ рекурсії", "параметрів")


def configs(quadratic, linear, rates, temperature):
    """Ряди, вікно спостереження W, довжина входу мережі L і старти walk-forward."""
    return [
        {
            "key": "quadratic",
            "short": "1. квадратичний",
            "title": "квадратичний тренд + нормальна похибка",
            "series": quadratic[0],
            "truth": quadratic[1],
            "window": 200,
            "lookback": 20,
            "origins": (0, 50, 100),
            "period": None,
        },
        {
            "key": "linear",
            "short": "2. лінійний",
            "title": "лінійний тренд + експоненційна похибка",
            "series": linear[0],
            "truth": linear[1],
            "window": 200,
            "lookback": 20,
            "origins": (0, 50, 100),
            "period": None,
        },
        {
            "key": "rates",
            "short": "3. курс",
            "title": "курс купівлі USD, Ощадбанк",
            "series": np.asarray(rates, dtype=float),
            "truth": None,
            "window": 150,
            "lookback": 10,
            "origins": (0, 24, 48),
            "period": None,
        },
        {
            "key": "temperature",
            "short": "4. температура",
            "title": "температура Києва, доба",
            "series": np.asarray(temperature, dtype=float),
            "truth": None,
            "window": 1095,
            "lookback": 30,
            "origins": (0, 730, 1460),
            "period": None,
        },
    ]


def properties(cfg):
    """Статистики, стаціонарність, сезонність і декомпозиція y = D(t) + r(t) першого вікна."""
    y, w = cfg["series"], cfg["window"]
    first = y[:w]
    period, share = features.period(first, cfg["period"])
    degree, cv = strategies.select_degree(first, period)
    d = strategies.deterministic(first, degree, period)(np.arange(w))
    resid = first - d
    stats = features.describe(y) | features.stationarity(first)
    resid_stat = features.stationarity(resid)
    print(f"   n = {len(y)}, вікно W = {w}, вхід мережі L = {cfg['lookback']}")
    print(
        "   "
        + ", ".join(
            f"{k} = {v:.3f}" if isinstance(v, float) else f"{k} = {v}" for k, v in stats.items()
        )
    )
    print(
        f"   періодограма першого вікна: період {period or 'не виявлено'} (частка піку {share:.3f})"
    )
    print(
        "   CV RMSE степеня тренду: "
        + ", ".join(f"{k} – {v:.3f}" for k, v in cv.items())
        + f"; обрано {degree}"
    )
    r2 = 1 - np.var(resid) / np.var(first)
    acf = features.acf(resid, (1, 2, 5))
    print(
        f"   D(t) пояснює R² = {r2:.3f}; залишок: σ = {resid.std():.3f}, "
        f"ACF(1, 2, 5) = {', '.join(f'{a:.3f}' for a in acf)}, "
        f"p_ADF = {resid_stat['p_ADF']:.3g}, стаціонарний: {resid_stat['стаціонарний']}"
    )
    return {"period": period, "degree": degree, "cv": cv, "d": d, "resid": resid, "R²": r2}


def spec(cfg, props, units, loss, seed, degree=None, period=None):
    return {
        "lookback": cfg["lookback"],
        "units": units,
        "loss": loss,
        "seed": seed,
        "degree": props["degree"] if degree is None else degree,
        "period": props["period"] if period is None else period,
    }


def structure_tasks(cfgs, props):
    tasks = []
    for cfg in cfgs:
        w = cfg["window"]
        for kind in networks.KINDS:
            for size, units in networks.SIZES.items():
                for loss in networks.LOSSES:
                    for seed in SEEDS:
                        tasks.append(
                            {
                                "key": f"structure|{cfg['key']}|{kind}|{size}|{loss}|{seed}",
                                "strategy": "гібрид",
                                "kind": kind,
                                "y": cfg["series"][:w],
                                "steps": 1,
                                "spec": spec(cfg, props[cfg["key"]], units, loss, seed),
                                "meta": (cfg["key"], kind, size, loss, seed),
                            }
                        )
    return tasks


def structure(cfgs, props, cache, fresh):
    """Синтез структури: для кожного типу мережі обирається розмір і функція втрат з найменшим
    середнім рангом за трьома індикаторами на всіх рядах."""
    tasks = structure_tasks(cfgs, props)
    results = runner.run(tasks, cache, fresh)
    rows = []
    for t in tasks:
        r = results[t["key"]]
        series, kind, size, loss, seed = t["meta"]
        rows.append(
            {
                "ряд": series,
                "мережа": kind,
                "розмір": size,
                "втрати": loss,
                "seed": seed,
                "епох": r["епох"],
                "секунд": r["секунд"],
            }
            | {k: r[k] for k in INDICATORS}
        )
    raw = pd.DataFrame(rows)
    table = raw.groupby(["мережа", "розмір", "втрати", "ряд"], sort=False).mean(numeric_only=True)
    table = table.drop(columns="seed").reset_index()
    ranks = table.copy()
    for ind in INDICATORS:
        ranks[ind] = table.groupby(["мережа", "ряд"])[ind].rank()
    ranks["ранг"] = ranks[list(INDICATORS)].mean(axis=1)
    score = ranks.groupby(["мережа", "розмір", "втрати"], sort=False)["ранг"].mean()
    chosen = {}
    for kind in networks.KINDS:
        size, loss = score.loc[kind].idxmin()
        chosen[kind] = {"розмір": size, "втрати": loss, "units": networks.SIZES[size]}
    summary = table.groupby(["мережа", "розмір", "втрати"], sort=False).mean(numeric_only=True)
    summary["ранг"] = score
    print(summary.round(3).to_string())
    print(
        "   обрані структури: "
        + "; ".join(f"{k} – {v['розмір']}, {v['втрати']}" for k, v in chosen.items())
    )
    return {"table": table, "summary": summary, "chosen": chosen, "results": results, "raw": raw}


def forecast_tasks(cfgs, props, chosen):
    tasks = []
    for cfg in cfgs:
        y, w = cfg["series"], cfg["window"]
        steps = int(max(evaluation.HORIZONS) * w)
        for s in cfg["origins"]:
            train = y[s : s + w]
            period, _ = features.period(train, cfg["period"])
            degree, _ = strategies.select_degree(train, period)
            for kind in networks.KINDS:
                c = chosen[kind]
                for strategy in strategies.STRATEGIES:
                    tasks.append(
                        {
                            "key": f"forecast|{cfg['key']}|{s}|{kind}|{strategy}|"
                            f"{c['розмір']}|{c['втрати']}",
                            "strategy": strategy,
                            "kind": kind,
                            "y": train,
                            "steps": steps,
                            "spec": spec(
                                cfg, props[cfg["key"]], c["units"], c["втрати"], 0, degree, period
                            ),
                            "meta": (cfg["key"], s, kind, strategy, degree, period),
                        }
                    )
                    tasks.append(inner_task(cfg, props, c, s, kind, strategy, degree, period))
    return tasks


def inner_task(cfg, props, c, s, kind, strategy, degree, period):
    """Внутрішня валідація адаптивного методу: мережа бачить вікно без останніх W/4 точок
    і прогнозує їх. Структура D(t) – та сама, що на повному вікні: на коротшому вікні
    періодограма може не виділити річний період, і гібрид виродився б у стратегію на рівнях."""
    w = cfg["window"]
    hold = w // HOLDOUT
    train = cfg["series"][s : s + w - hold]
    return {
        "key": f"inner|{cfg['key']}|{s}|{kind}|{strategy}|{c['розмір']}|{c['втрати']}|"
        f"{degree}|{period}",
        "strategy": strategy,
        "kind": kind,
        "y": train,
        "steps": hold,
        "spec": spec(cfg, props[cfg["key"]], c["units"], c["втрати"], 0, degree, period),
        "meta": None,
    }


def method_name(kind, strategy):
    return f"{kind}, {strategy}"


def forecasts(cfgs, props, chosen, cache, fresh):
    """Прогнози всіх мереж і стратегій з кожного старту; ансамбль – середнє трьох мереж,
    адаптивний метод – ансамбль стратегії з найменшою похибкою на внутрішній валідації."""
    tasks = forecast_tasks(cfgs, props, chosen)
    results = runner.run(tasks, cache, fresh)
    runs = {cfg["key"]: {"paths": {}, "fits": {}, "choice": {}, "adaptive": {}} for cfg in cfgs}
    for t in tasks:
        if t["meta"] is None:
            continue
        key, s, kind, strategy, degree, period = t["meta"]
        r = results[t["key"]]
        run = runs[key]
        run["paths"].setdefault(s, {})[method_name(kind, strategy)] = r["forecast"]
        run["fits"].setdefault(s, {})[method_name(kind, strategy)] = r
        run["choice"][s] = (degree, period)
    for cfg in cfgs:
        run = runs[cfg["key"]]
        y, w = cfg["series"], cfg["window"]
        steps = int(max(evaluation.HORIZONS) * w)
        for s, paths in run["paths"].items():
            degree, period = run["choice"][s]
            train = y[s : s + w]
            paths["наївний"] = np.full(steps, train[-1])
            paths["D(t)"] = strategies.deterministic(train, degree, period)(np.arange(w, w + steps))
            for strategy in strategies.STRATEGIES:
                paths[f"ансамбль, {strategy}"] = np.mean(
                    [paths[method_name(k, strategy)] for k in networks.KINDS], axis=0
                )
            paths[DAMPED] = damped(train, degree, period, paths["ансамбль, гібрид"])
            scores = adaptive_scores(cfg, s, chosen, results, degree, period)
            best = min(scores, key=scores.get)
            run["adaptive"][s] = (best, scores)
            paths["адаптивний"] = paths[DAMPED if best == "гібрид" else f"ансамбль, {best}"]
            print(
                f"   {cfg['short']}, старт {s}: RMSE ансамблів на відкладених W/{HOLDOUT} – "
                + ", ".join(f"{k} {v:.3f}" for k, v in scores.items())
                + f"; обрано {best}"
            )
    return runs


def adaptive_scores(cfg, s, chosen, results, degree, period):
    """RMSE ансамблю кожної стратегії на останніх W/4 точках вікна за прогнозом з внутрішньої
    валідації: ŷ = (1/3) Σ ŷ_мережі (для гібриду – з вагою ACF), RMSE = √(mean (ŷ − y)²)."""
    w = cfg["window"]
    hold = w // HOLDOUT
    actual = cfg["series"][s + w - hold : s + w]
    scores = {}
    for strategy in strategies.STRATEGIES:
        paths = [
            results[
                f"inner|{cfg['key']}|{s}|{kind}|{strategy}|{chosen[kind]['розмір']}|"
                f"{chosen[kind]['втрати']}|{degree}|{period}"
            ]["forecast"]
            for kind in networks.KINDS
        ]
        path = np.mean(paths, axis=0)
        if strategy == "гібрид":
            path = damped(cfg["series"][s : s + w - hold], degree, period, path)
        scores[strategy] = np.sqrt(np.mean((path - actual) ** 2))
    return scores


def damped(train, degree, period, hybrid):
    """Гібрид з вагою ACF: ŷₕ = D(n + h) + wₕ·(ŷₕ^гібрид − D(n + h))."""
    n = len(train)
    d = strategies.deterministic(train, degree, period)
    trend = d(np.arange(n, n + len(hybrid)))
    weight = strategies.acf_weight(train - d(np.arange(n)), len(hybrid))
    return trend + weight * (hybrid - trend)


def methods():
    names = ["наївний", "D(t)"]
    for strategy in strategies.STRATEGIES:
        names += [method_name(k, strategy) for k in networks.KINDS] + [f"ансамбль, {strategy}"]
    return [*names, DAMPED, "адаптивний"]


def evaluate(cfg, run):
    """KPI прогнозу поза вікном на горизонтах 0,1W, 0,5W і W, усереднені по стартах."""
    y, w = cfg["series"], cfg["window"]
    out = {}
    for name in methods():
        summary, _, paths = evaluation.backtest(
            y, w, cfg["origins"], lambda tr, h, s, n=name: run["paths"][s][n][:h]
        )
        out[name] = {"summary": summary, "paths": paths}
    for h in evaluation.HORIZONS:
        frame = pd.DataFrame({m: r["summary"].loc[h] for m, r in out.items()}).T
        print(f"   KPI прогнозу поза вікном, горизонт {h:g}W = {int(h * w)} кроків:")
        print(frame.round(3).to_string())
    return out


def in_sample(cfg, run):
    """Прогноз на крок у першому вікні: KPI до ряду і до тренду, де він відомий."""
    y, w, s = cfg["series"], cfg["window"], cfg["origins"][0]
    rows = {}
    for name, r in run["fits"][s].items():
        fitted = r["fitted"]
        ok = ~np.isnan(fitted)
        row = evaluation.metrics(y[s : s + w][ok], fitted[ok])
        if cfg["truth"] is not None:
            row["СКВ до тренду"] = np.sqrt(np.mean((fitted[ok] - cfg["truth"][s : s + w][ok]) ** 2))
        row |= {"епох": r["епох"], "секунд": r["секунд"]}
        rows[name] = row
    frame = pd.DataFrame(rows).T
    print(f"   прогноз на крок на ділянці спостереження, старт {s}:")
    print(frame.round(3).to_string())
    return frame
