"""Ретроспективна перевірка методів з ковзним початком прогнозу і підсумковий прогноз."""

import numpy as np
import pandas as pd

import base
import hierarchy
import reconcile

ORIGINS = (2015, 2016, 2017, 2018)
HORIZON = 3
FINAL_HORIZON = 5


def backtest(births, depth):
    """Похибки кожного методу для кожного вузла, старту і кроку прогнозу.

    MAPE = 100·|ỹ − y| / y; MASE ділить похибку на середню зміну ряду за рік на ділянці
    навчання, тож вузли різного масштабу порівнюються на одній шкалі.
    """
    full = hierarchy.build(births, depth)
    frames = []
    for origin in ORIGINS:
        h = hierarchy.build(births, depth, last=origin)
        F, E, _ = base.forecast(h["Y"], HORIZON)
        future = (full["years"] > origin) & (full["years"] <= origin + HORIZON)
        actual = full["Y"][future].T
        scale = np.abs(np.diff(h["Y"], axis=0)).mean(axis=0)[:, None]
        n = len(h["names"])
        for method, forecast in reconcile.run(h, F, E, h["Y"]).items():
            error = np.abs(forecast - actual)
            frames.append(
                pd.DataFrame(
                    {
                        "depth": depth,
                        "origin": origin,
                        "method": method,
                        "node": np.repeat(h["names"], HORIZON),
                        "level": np.repeat(h["level"], HORIZON),
                        "step": np.tile(np.arange(1, HORIZON + 1), n),
                        "ape": (100 * error / actual).ravel(),
                        "ase": (error / scale).ravel(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def summary(errors, metric="ape"):
    """Метод × рівень: середня похибка; «загалом» – середнє рівнів з однаковою вагою."""
    table = errors.pivot_table(index="method", columns="level", values=metric, aggfunc="mean")
    table.columns = [hierarchy.LEVELS[c] for c in table.columns]
    table["загалом"] = table.mean(axis=1)
    order = ["base"] + [m for m in reconcile.METHODS if m in table.index]
    return table.loc[order]


def best(table):
    return table.drop(index="base")["загалом"].idxmin()


def final(births, depth):
    """Прогноз на FINAL_HORIZON років з повних даних усіма методами."""
    h = hierarchy.build(births, depth)
    F, E, models = base.forecast(h["Y"], FINAL_HORIZON)
    years = np.arange(h["years"][-1] + 1, h["years"][-1] + 1 + FINAL_HORIZON)
    return {
        "h": h,
        "F": F,
        "E": E,
        "models": models,
        "years": years,
        "forecasts": reconcile.run(h, F, E, h["Y"]),
        "lambda": reconcile.shrink_cov(E)[1],
    }


def incoherence(h, F):
    """Розбіжність базових прогнозів: (ŷ_країна − Σ ŷ_нижнього рівня) / ŷ_країна, %."""
    return 100 * (F[0] - F[h["bottom"]].sum(axis=0)) / F[0]
