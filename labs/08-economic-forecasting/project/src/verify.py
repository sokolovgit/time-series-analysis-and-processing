"""Верифікація: схема компромісів на матрицях з відомою відповіддю і вся технологія на сигналах,
для яких відомо, який клас методів має виграти."""

import numpy as np
import pandas as pd

import integral
import level3

RUNS = 2
N = level3.WINDOW + level3.STEPS + 36
ORIGINS = 3
SIGMA = 3.0
SLOPE = 0.5
AMPLITUDE, YEAR = 15.0, 12
CYCLE = 7

NOISE = {"лінійний тренд": SIGMA, "тренд + сезон 12": SIGMA, "нелінійний цикл 7": SIGMA / 2}
NO_SEASON = "статистичне навчання"
NOISE_LEVEL, CLOSE, SMALL_WEIGHT = 1.3, 1.3, 0.2


def signals(rng):
    """Лінійний тренд точно описують моделі всіх трьох класів. Річну сезонність і несиметричний
    цикл періоду 7 не описують поліном і α-β фільтр – клас статистичного навчання тут
    неадекватний і має втратити вагу в інтегрованому прогнозі."""
    t = np.arange(N, dtype=float)
    phase = 2 * np.pi * t / CYCLE
    out = []
    for i in range(RUNS):
        noise = rng.normal(0, SIGMA, N)
        out.append(("лінійний тренд", i, 100 + SLOPE * t + noise))
        season = AMPLITUDE * np.sin(2 * np.pi * t / YEAR)
        out.append(("тренд + сезон 12", i, 100 + SLOPE * t + season + noise))
        cycle = AMPLITUDE * (np.sin(phase) + 0.6 * np.sin(2 * phase) + 0.3 * np.cos(3 * phase))
        out.append(("нелінійний цикл 7", i, 100 + cycle + noise / 2))
    return out


def compromise_checks():
    """Три властивості схеми: домінуюча альтернатива перша, масштаб критерію не впливає на ранг,
    за однакової суми нормованих критеріїв рівномірна альтернатива краща за нерівномірну."""
    base = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 1.0, 2.0]],
        index=["A", "B", "C"],
        columns=["k1", "k2", "k3"],
    )
    w = dict.fromkeys(base.columns, 1.0)
    scaled = base.assign(k2=base["k2"] * 1000)
    shares = pd.DataFrame(
        [[0.5, 0.1], [0.1, 0.5], [0.3, 0.3], [0.1, 0.1]],
        index=["нерівна 1", "нерівна 2", "рівна", "інша"],
        columns=["k1", "k2"],
    )
    y = integral.rate(shares, dict.fromkeys(shares.columns, 1.0))["Y"]
    return {
        "A домінує B": integral.rate(base, w).loc["A", "Y"] < integral.rate(base, w).loc["B", "Y"],
        "ранги не залежать від масштабу": integral.rate(base, w)["ранг"].equals(
            integral.rate(scaled, w)["ранг"]
        ),
        "рівна краща за нерівні при тій самій сумі": y["рівна"]
        < min(y["нерівна 1"], y["нерівна 2"]),
    }


def run(rng, cache, fresh):
    for name, ok in compromise_checks().items():
        print(f"   {name}: {'так' if ok else 'НІ'}")
    cfgs = [
        {
            "key": f"verify-{name}-{i}",
            "title": f"{name}, реалізація {i}",
            "name": name,
            "series": y,
            "window": level3.WINDOW,
            "origins": ORIGINS,
        }
        for name, i, y in signals(rng)
    ]
    runs = level3.process(cfgs, cache, fresh)
    rows = []
    for cfg in cfgs:
        res = level3.evaluate(cfg, runs)
        table = res["table"]
        single = table.loc[list(level3.METHODS)]
        name = str(cfg["name"])
        best = single["RMSE 24"].idxmin()
        sigma = NOISE[name]
        weights = res["weights"].mean()
        best_ratio = float(single.loc[best, "RMSE 24"]) / sigma
        combo_ratio = float(table.loc[level3.INTEGRATED, "RMSE 24"]) / sigma
        rows.append(
            {
                "сигнал": name,
                "кращий метод": best,
                "RMSE кращого / σ": best_ratio,
                "RMSE інтегрованого / σ": combo_ratio,
                "RMSE адаптивного вибору / σ": table.loc[level3.CHOICE, "RMSE 24"] / sigma,
            }
            | {f"вага: {c}": weights[c] for c in level3.CLASSES}
            | {
                "на рівні шуму": best_ratio < NOISE_LEVEL,
                "інтегрований близький": combo_ratio < CLOSE * best_ratio,
                "неадекватний клас відсічено": name == "лінійний тренд"
                or weights[NO_SEASON] < SMALL_WEIGHT,
            }
        )
    checks = pd.DataFrame(rows)
    print(checks.round(3).to_string())
    flags = ["на рівні шуму", "інтегрований близький", "неадекватний клас відсічено"]
    for f in flags:
        print(f"   {f}: {checks[f].sum()} з {len(checks)}")
    return checks
