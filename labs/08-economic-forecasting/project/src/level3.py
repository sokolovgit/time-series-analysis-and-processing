"""Рівень III: один і той самий ряд обробляють три класи методів – статистичне навчання,
апроксимація і глибинне навчання; результати зводяться в інтегровану оцінку і прогноз."""

import time

import numpy as np
import pandas as pd

import adaptive
import exponential
import features
import filters
import integral
import learn
import regression
import runner
import strategies

WINDOW, HOLD, LOOKBACK = 120, 12, 12
HORIZONS = (6, 12, 24)
STEPS = max(HORIZONS)
ORIGINS = 6
DEGREES = (1, 2, 3)
NETS = {"MLP": (32, "MSE"), "LSTM": (16, "MSE"), "CNN": (16, "MSE")}
CLASSES = {
    "статистичне навчання": ("МНК-поліном", "α-β фільтр"),
    "апроксимація": ("експоненційне", "регресійне"),
    "глибинне навчання": tuple(NETS),
}
METHODS = tuple(m for ms in CLASSES.values() for m in ms)
INTEGRATED, CHOICE, NAIVE = "інтегрований", "адаптивний вибір", "наївний"
INNER = ("RMSE", "MAE", "|зміщення|")
WEIGHTS = {
    "RMSE 6": 0.2,
    "RMSE 12": 0.2,
    "RMSE 24": 0.2,
    "|зміщення| 24": 0.15,
    "розкид RMSE 24": 0.15,
    "lg мс": 0.1,
}


def origins(n, window=WINDOW, count=ORIGINS):
    return np.linspace(0, n - window - STEPS, count).astype(int)


def lsq(train, steps, period):
    """МНК-поліном степеня, обраного за відкладеним хвостом: ŷ(n + h) = Σ cₖ ((n + h)/n)ᵏ."""
    degree, _, _ = learn.select_degree(train, DEGREES)
    _, coeffs = learn.fit(train, degree)
    return learn.extrapolate(coeffs, len(train), steps)


def alpha_beta(train, steps, period):
    """Адаптивний α-β фільтр з перезапуском за CUSUM; прогноз з кінцевого стану: ŷ(n + h) = x + v·h.

    Стан фільтра зі зростаючою пам'яттю після n кроків – це МНК-пряма за останніми n вимірами,
    тому його можна відновити прямо з пам'яті на останньому кроці."""
    _, _, memory, _, _ = adaptive.run(train, 2)
    x, v, _ = filters.fit_state(train[-memory[-1] :], 2)
    return x + v * np.arange(1, steps + 1)


def smoothing(train, steps, period):
    """Експоненційне згладжування: структура і α – за CV-похибкою на чверті вікна."""
    best, alpha, _, _ = exponential.select_cv(train, period, len(train) // 4)
    spec, s = exponential.candidates(period)[best]
    return exponential.fit(train, spec, s, alpha).forecast(steps)


def regressive(train, steps, period):
    """Регресійне згладжування: тренд, гармоніки або лаги – за правилом 1-SE на CV."""
    name, rows = regression.select(train, len(train) // 4, period)
    kind, kw = rows[name]["spec"]
    return regression.run(kind, train, steps, kw)[0]


def naive(train, steps, period):
    """Еталон: останнє значення або значення того самого місяця минулого року."""
    return np.resize(train[-period:], steps) if period else np.full(steps, train[-1])


CLASSICAL = {
    "МНК-поліном": lsq,
    "α-β фільтр": alpha_beta,
    "експоненційне": smoothing,
    "регресійне": regressive,
    NAIVE: naive,
}


def classical(train, steps, period):
    forecasts, seconds = {}, {}
    for name, method in CLASSICAL.items():
        start = time.perf_counter()
        forecasts[name] = np.asarray(method(train, steps, period), dtype=float)
        seconds[name] = time.perf_counter() - start
    return forecasts, seconds


def slices(y, s, window):
    """Повне вікно y[s : s + W] і внутрішнє – без останніх HOLD точок для валідації."""
    full = y[s : s + window]
    return {"inner": (full[:-HOLD], HOLD), "full": (full, STEPS)}


def net_tasks(cfg, s, period, degree):
    tasks = []
    for part, (train, steps) in slices(cfg["series"], s, cfg["window"]).items():
        for kind, (units, loss) in NETS.items():
            tasks.append(
                {
                    "key": f"{cfg['key']}|{len(cfg['series'])}|{s}|{part}|{kind}|{units}|{loss}|"
                    f"{degree}|{period}",
                    "strategy": "гібрид",
                    "kind": kind,
                    "y": train,
                    "steps": steps,
                    "spec": {
                        "lookback": LOOKBACK,
                        "units": units,
                        "loss": loss,
                        "seed": 0,
                        "degree": degree,
                        "period": period,
                    },
                    "meta": (cfg["key"], s, part, kind),
                }
            )
    return tasks


def errors(actual, forecast):
    e = np.asarray(forecast, dtype=float) - np.asarray(actual, dtype=float)
    return {"RMSE": np.sqrt(np.mean(e**2)), "MAE": np.mean(np.abs(e)), "|зміщення|": abs(e.mean())}


def integrate(inner, truth, full):
    """Інтегрований прогноз з трьох класів.

    1. Кожен метод оцінюється на внутрішній валідації трьома критеріями (RMSE, MAE, |зміщення|),
       інтегральний показник Y – за нелінійною схемою компромісів з рівними вагами.
    2. Лідер класу c – метод класу з найменшим Y.
    3. Прогноз – зважена сума прогнозів лідерів: ŷ = Σ w_c ŷ_c,  w_c ∝ (Y_c − 1)⁻¹.
       Y − 1 = Σ λₖ y⁰ₖ / (1 − y⁰ₖ) – нуль для ідеального методу, тому вага обернена до частки
       похибки. Адаптивний вибір – прогноз одного методу з найменшим Y серед усіх.
    """
    table = pd.DataFrame({m: errors(truth, inner[m]) for m in METHODS}).T
    y = integral.rate(table, dict.fromkeys(INNER, 1.0))["Y"]
    leaders = {c: min(ms, key=lambda m: y[m]) for c, ms in CLASSES.items()}
    w = np.array([1.0 / max(y[m] - 1.0, 1e-12) for m in leaders.values()])
    w /= w.sum()
    forecast = sum(wi * full[m] for wi, m in zip(w, leaders.values(), strict=True))
    weights = dict(zip(leaders, w, strict=True))
    return forecast, full[y.idxmin()], {"leaders": leaders, "weights": weights, "Y": y}


def process(cfgs, cache, fresh):
    """Усі методи на всіх стартах; мережі навчаються паралельно, решта – у головному процесі."""
    plan, tasks = [], []
    for cfg in cfgs:
        y = cfg["series"]
        for s in origins(len(y), cfg["window"], cfg.get("origins", ORIGINS)):
            full = y[s : s + cfg["window"]]
            period = cfg["period"] if "period" in cfg else features.annual(full)[0]
            degree, _ = strategies.select_degree(full, period)
            plan.append((cfg, int(s), period, degree))
            tasks += net_tasks(cfg, int(s), period, degree)
    nets = runner.run(tasks, cache, fresh)
    by_meta = {t["meta"]: nets[t["key"]] for t in tasks}

    runs = []
    for cfg, s, period, degree in plan:
        y, w = cfg["series"], cfg["window"]
        out = {"key": cfg["key"], "start": s, "period": period, "degree": degree}
        for part, (train, steps) in slices(y, s, w).items():
            forecasts, seconds = classical(train, steps, period)
            for kind in NETS:
                r = by_meta[(cfg["key"], s, part, kind)]
                forecasts[kind], seconds[kind] = r["forecast"], r["секунд"]
            out[part] = forecasts
            out[f"{part} секунд"] = seconds
        truth_inner = y[s + w - HOLD : s + w]
        combo, choice, info = integrate(out["inner"], truth_inner, out["full"])
        out["full"][INTEGRATED], out["full"][CHOICE] = combo, choice
        total = sum(out["full секунд"][m] + out["inner секунд"][m] for m in METHODS)
        out["full секунд"][INTEGRATED] = out["full секунд"][CHOICE] = total
        out["info"] = info
        out["truth"] = y[s + w : s + w + STEPS]
        runs.append(out)
    return runs


def criteria(runs, key):
    """Критерії методу на ряду: RMSE на горизонтах 6, 12, 24, |зміщення| і розкид RMSE на 24
    між стартами, час обчислень. Усі – по стартах walk-forward, усі мінімізуються."""
    rows = []
    for run in (r for r in runs if r["key"] == key):
        for m, f in run["full"].items():
            row = {"метод": m, "старт": run["start"], "секунд": run["full секунд"][m]}
            for h in HORIZONS:
                row[f"RMSE {h}"] = errors(run["truth"][:h], f[:h])["RMSE"]
            row["зміщення 24"] = float(np.mean(f - run["truth"]))
            rows.append(row)
    raw = pd.DataFrame(rows)
    table = raw.groupby("метод", sort=False).agg(
        **{f"RMSE {h}": (f"RMSE {h}", "mean") for h in HORIZONS},
        **{
            "|зміщення| 24": ("зміщення 24", lambda b: np.mean(np.abs(b))),
            "розкид RMSE 24": ("RMSE 24", "std"),
            "секунд": ("секунд", "mean"),
        },
    )
    # вартість – у порядках величини: інакше секунди мереж на тлі мілісекунд МНК виродили б
    # частку інших альтернатив у нуль, і критерій вартості задавив би точність
    table["lg мс"] = np.log10(1 + 1000 * table["секунд"])
    return table, raw


def evaluate(cfg, runs):
    table, raw = criteria(runs, cfg["key"])
    rated = table.drop(index=NAIVE)[list(WEIGHTS)]
    rating = integral.rate(rated, WEIGHTS)
    ranks, entropy = integral.sensitivity(rated, WEIGHTS)
    table["U Тейла 24"] = table["RMSE 24"] / table.loc[NAIVE, "RMSE 24"]
    table = table.join(rating)
    by_class = {c: min(ms, key=lambda m: rating.loc[m, "Y"]) for c, ms in CLASSES.items()}
    own = [r for r in runs if r["key"] == cfg["key"]]
    leaders = pd.DataFrame([r["info"]["leaders"] for r in own], index=[r["start"] for r in own])
    weights = pd.DataFrame([r["info"]["weights"] for r in own], index=leaders.index)
    return {
        "table": table,
        "raw": raw,
        "ranks": ranks,
        "entropy": entropy,
        "by_class": by_class,
        "leaders": leaders,
        "weights": weights,
        "runs": own,
    }


def run(cfgs, cache, fresh):
    runs = process(cfgs, cache, fresh)
    results = {}
    for cfg in cfgs:
        res = evaluate(cfg, runs)
        results[cfg["key"]] = res
        own = res["runs"]
        print(f"\n   {cfg['title']}: стартів {len(own)}, вікно W = {cfg['window']}")
        print(
            "   період: "
            + ", ".join(str(r["period"] or "–") for r in own)
            + "; степінь D(t) для мереж: "
            + ", ".join(str(r["degree"]) for r in own)
        )
        cols = [*WEIGHTS, "секунд", "U Тейла 24", "Y", "ранг"]
        print(res["table"][cols].sort_values("Y").round(3).to_string())
        print("   ранги за різних ваг (задані / рівні / ентропійні):")
        print(res["ranks"].T.to_string())
        print("   ентропійні ваги: " + ", ".join(f"{k} {v:.2f}" for k, v in res["entropy"].items()))
        print(
            "   найкращий метод класу: "
            + "; ".join(f"{c} – {m}" for c, m in res["by_class"].items())
        )
        print("   лідери класів на внутрішній валідації і ваги інтегрованого прогнозу:")
        print(
            res["leaders"]
            .astype(str)
            .add(" (")
            .add(res["weights"].round(2).astype(str))
            .add(")")
            .to_string()
        )
    return {"runs": runs, "results": results}
