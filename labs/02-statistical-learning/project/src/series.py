"""Вхідні дані: адитивна модель тренду з шумом та аномальними вимірами."""

import numpy as np
import pandas as pd

A, B, C = 1e-4, 1e-2, 10.0
SIGMA = 5.0


def trend(n, a=A, b=B, c=C):
    """Ідеальний тренд S(t) = a·t² + b·t + c."""
    t = np.arange(n, dtype=float)
    return a * t**2 + b * t + c


def sample(n, rng, sigma=SIGMA):
    """Адитивна модель Y(t) = S(t) + ε(t), ε ~ N(0, σ²)."""
    return trend(n) + rng.normal(0.0, sigma, n)


def add_anomalies(data, rng, fraction=0.1, q=5.0, sigma=SIGMA):
    """Адитивні викиди зі зсувом від q·σ у випадковий бік, у частці fraction відліків."""
    out = data.copy()
    index = rng.choice(len(data), int(len(data) * fraction), replace=False)
    shift = q + np.abs(rng.normal(0.0, 1.0, len(index)))
    out[index] += rng.choice([-1.0, 1.0], len(index)) * shift * sigma
    return out, np.sort(index)


def real_rates(path, column="Купівля"):
    """Курс USD з файлу Ощадбанку. Нулі – це пропуски, заповнюємо інтерполяцією."""
    values = pd.read_excel(path)[column].to_numpy(dtype=float)
    gaps = values == 0.0
    if gaps.any():
        i = np.arange(len(values), dtype=float)
        values[gaps] = np.interp(i[gaps], i[~gaps], values[~gaps])
    return values, int(gaps.sum())
