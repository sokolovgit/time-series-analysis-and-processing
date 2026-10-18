"""Верифікація методів: незміщеність, когерентність, синтетична ієрархія, звірка зі scikit-hts."""

import collections
import collections.abc

import numpy as np
import pandas as pd

import reconcile

# scikit-hts 0.5.12 імпортує Iterable зі старого місця, прибраного в Python 3.10
collections.Iterable = collections.abc.Iterable  # ty: ignore[unresolved-attribute]

from hts.revision import RevisionMethod  # noqa: E402

LIBRARY = {"NONE": "base", "BU": "BU", "OLS": "OLS", "WLSS": "WLS"}


def unbiasedness(h, E, Y):
    """‖G·S − I‖ для лінійних методів, G знаходиться підстановкою F = I.

    G·S = I означає, що незміщені базові прогнози лишаються незміщеними після узгодження.
    """
    n = len(h["names"])
    out = {}
    for name in reconcile.LINEAR:
        G = reconcile.METHODS[name](h, np.eye(n), E, Y)
        out[name] = float(np.abs(G @ h["S"] - np.eye(h["S"].shape[1])).max())
    return out


def coherence(h, forecasts):
    """max |ỹ − S·ỹ_нижнього рівня| – нуль, якщо прогноз сумується за ієрархією."""
    S, bottom = h["S"], h["bottom"]
    return {m: float(np.abs(f - S @ f[bottom]).max()) for m, f in forecasts.items()}


def synthetic(h, E, Y, rng):
    """Когерентні базові прогнози S·b мають лишатися незмінними після узгодження."""
    b = rng.uniform(1_000, 20_000, size=(h["S"].shape[1], 5))
    F = h["S"] @ b
    return {
        name: float(np.abs(h["S"] @ reconcile.METHODS[name](h, F, E, Y) - F).max() / F.max())
        for name in reconcile.methods(h["depth"])
    }


def library(h, F, forecasts):
    """Ті самі базові прогнози, узгоджені RevisionMethod зі scikit-hts; max відносна різниця."""
    frames = {name: pd.DataFrame({"yhat": F[i]}) for i, name in enumerate(h["names"])}
    out = {}
    for method, ours in LIBRARY.items():
        theirs = RevisionMethod(method, h["S"], None).revise(forecasts=frames, mse=None, nodes=None)
        out[f"{method} ↔ {ours}"] = float(np.abs(theirs.T - forecasts[ours]).max() / F.max())
    return out


def run(final, rng):
    h, F, E = final["h"], final["F"], final["E"]
    report = {
        "незміщеність ‖GS − I‖": unbiasedness(h, E, h["Y"]),
        "когерентність max|ỹ − Sỹ_b|": coherence(h, final["forecasts"]),
        "когерентний вхід, відн. зміна": synthetic(h, E, h["Y"], rng),
        "scikit-hts, відн. різниця": library(h, F, final["forecasts"]),
    }
    for title, values in report.items():
        print(f"  {title}:")
        for name, value in values.items():
            print(f"    {name:14s} {value:.2e}")
    return report
