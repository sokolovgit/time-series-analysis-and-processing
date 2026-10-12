"""Вхідні дані: квадратичний і кусковий тренди з нормальним шумом та аномальними вимірами."""

import numpy as np
import pandas as pd

A, B, C = 1e-4, 1e-2, 10.0
SIGMA = 5.0

SLOPE = 0.05
BREAKS = (0.3, 0.6)
CURVATURE = 4e-3


def quadratic(n, a=A, b=B, c=C):
    """Ідеальний тренд S(t) = a·t² + b·t + c."""
    t = np.arange(n, dtype=float)
    return a * t**2 + b * t + c


def piecewise(n, slope=SLOPE, breaks=BREAKS, a=CURVATURE, c=C):
    """Лінійний тренд, який на [t₁, t₂) розганяється з прискоренням 2a, а далі знову лінійний.

    S(t) = c + k·t + a·(t − t₁)² на ділянці розгону, похідна в точках зламу неперервна.
    """
    t = np.arange(n, dtype=float)
    t1, t2 = (int(n * b) for b in breaks)
    s = c + slope * t
    s += np.where(t >= t1, a * (np.minimum(t, t2) - t1) ** 2, 0.0)
    s += np.where(t >= t2, 2 * a * (t2 - t1) * (t - t2), 0.0)
    return s


def noisy(ideal, rng, sigma=SIGMA):
    """Адитивна модель Y(t) = S(t) + ε(t), ε ~ N(0, σ²)."""
    return ideal + rng.normal(0.0, sigma, len(ideal))


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
