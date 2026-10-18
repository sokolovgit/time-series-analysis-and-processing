"""Верифікація мереж і стратегій на сигналах з відомою поведінкою."""

import numpy as np
import pandas as pd

import networks
import runner
import synthetic

RUNS = 2
N = 300
SIGMA = 1.0
PHI = 0.7
PERIOD, AMPLITUDE, SINE_NOISE = 25, 1.0, 0.1
SLOPE = 0.2


def spec(lookback, seed, degree=0):
    return {
        "lookback": lookback,
        "units": 32,
        "loss": "MSE",
        "seed": seed,
        "degree": degree,
        "period": None,
    }


def signals(rng):
    t = np.arange(2 * N, dtype=float)
    out = []
    for i in range(RUNS):
        out.append(("ar1", i, 10 + synthetic.ar1(rng, N, PHI, SIGMA)))
        sine = AMPLITUDE * np.sin(2 * np.pi * t / PERIOD)
        out.append(("sine", i, sine[:N] + rng.normal(0, SINE_NOISE, N)))
        out.append(("trend", i, SLOPE * t[:N] + rng.normal(0, SIGMA, N)))
    return out


def tasks_for(rng):
    tasks = []
    for name, i, y in signals(rng):
        strategies = ("рівні", "прирости", "гібрид") if name == "trend" else ("рівні",)
        lookback = {"ar1": 5, "sine": 30, "trend": 20}[name]
        for kind in networks.KINDS:
            for strategy in strategies:
                tasks.append(
                    {
                        "key": f"verify|{name}|{i}|{kind}|{strategy}",
                        "strategy": strategy,
                        "kind": kind,
                        "y": y,
                        "steps": N,
                        "spec": spec(lookback, i, degree=1),
                        "meta": (name, i, kind, strategy),
                    }
                )
    return tasks


def run(rng, cache, fresh):
    tasks = tasks_for(rng)
    results = runner.run(tasks, cache, fresh)
    t = np.arange(N, 2 * N, dtype=float)
    rows = []
    for task in tasks:
        name, _, kind, strategy = task["meta"]
        r, y = results[task["key"]], task["y"]
        f = r["forecast"]
        row = {"сигнал": name, "мережа": kind, "стратегія": strategy}
        if name == "ar1":
            fitted = r["fitted"]
            ok = ~np.isnan(fitted)
            mu = 10.0
            theory = mu + PHI ** np.arange(1, 21) * (y[-1] - mu)
            row["СКВ на крок / σ"] = np.sqrt(np.mean((fitted[ok] - y[ok]) ** 2)) / SIGMA
            row["max |ŷ − теорія|, 20 кроків"] = np.abs(f[:20] - theory).max()
        elif name == "sine":
            truth = AMPLITUDE * np.sin(2 * np.pi * t / PERIOD)
            row["СКВ до сигналу, 4 періоди"] = np.sqrt(np.mean((f[:100] - truth[:100]) ** 2))
        else:
            row["СКВ до тренду, N кроків"] = np.sqrt(np.mean((f - SLOPE * t) ** 2))
        rows.append(row)
    frame = pd.DataFrame(rows)
    ar1 = frame[frame["сигнал"] == "ar1"].groupby("мережа", sort=False).mean(numeric_only=True)
    print(f"   AR(1), φ = {PHI}, σ = {SIGMA}, L = 5, {RUNS} реалізації; теорія μ + φʰ(y − μ):")
    print(ar1.dropna(axis=1).round(3).to_string())
    sine = frame[frame["сигнал"] == "sine"].groupby("мережа", sort=False).mean(numeric_only=True)
    print(
        f"   синусоїда, період {PERIOD}, амплітуда {AMPLITUDE}, шум σ = {SINE_NOISE}, L = 30, "
        "рекурсія на 100 кроків:"
    )
    print(sine.dropna(axis=1).round(3).to_string())
    trend = frame[frame["сигнал"] == "trend"]
    pivot = trend.pivot_table(
        index="мережа", columns="стратегія", values="СКВ до тренду, N кроків", sort=False
    )
    print(
        f"   лінійний тренд {SLOPE}·t + N(0, {SIGMA}²), прогноз на N = {N} кроків, СКВ до тренду:"
    )
    print(pivot.round(3).to_string())
    return {"frame": frame, "results": results, "tasks": tasks}
