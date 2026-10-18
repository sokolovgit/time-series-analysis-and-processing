"""Синтетичні ряди: моделі варіанта 8 з ЛР 1 і ряди з випадковими властивостями для селектора."""

import numpy as np

N = 500
QUADRATIC = (1e-4, 1e-2, 10.0)
SLOPE, LEVEL = 0.05, 10.0
SIGMA = 5.0

BENCH_N = 300
TRENDS = ("немає", "лінійний", "квадратичний", "блукання", "блукання з дрейфом", "злам")
PERIODS = (12, 24)


def quadratic_normal(rng, n=N):
    """Квадратичний тренд a·t² + b·t + c і нормальна похибка N(0, σ²). Повертає (ряд, тренд)."""
    t = np.arange(n, dtype=float)
    a, b, c = QUADRATIC
    trend = a * t**2 + b * t + c
    return trend + rng.normal(0.0, SIGMA, n), trend


def linear_exponential(rng, n=N):
    """Лінійний тренд і центрована експоненційна похибка Exp(σ) − σ з нульовим середнім."""
    t = np.arange(n, dtype=float)
    trend = LEVEL + SLOPE * t
    return trend + rng.exponential(SIGMA, n) - SIGMA, trend


def ar1(rng, n, phi, sigma):
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def benchmark_series(rng, n=BENCH_N):
    """Один ряд з випадковими властивостями: тип тренду, сезонність, закон і пам'ять похибки."""
    t = np.arange(n, dtype=float)
    kind = TRENDS[rng.integers(len(TRENDS))]
    scale = rng.uniform(5, 20)
    if kind == "лінійний":
        trend = scale * t / n * rng.choice([-1, 1]) * 2
    elif kind == "квадратичний":
        trend = scale * ((t / n) ** 2 * 3 - t / n) * rng.choice([-1, 1])
    elif kind == "блукання":
        trend = np.cumsum(rng.normal(0, scale / 20, n))
    elif kind == "блукання з дрейфом":
        trend = np.cumsum(rng.normal(scale / n * 2 * rng.choice([-1, 1]), scale / 20, n))
    elif kind == "злам":
        cut = rng.integers(n // 3, 2 * n // 3)
        trend = np.where(t < cut, scale * t / n, scale * cut / n - 2 * scale * (t - cut) / n)
    else:
        trend = np.zeros(n)
    period = int(rng.choice(PERIODS)) if rng.random() < 0.5 else 0
    seasonal = (
        rng.uniform(1, 2) * scale / 4 * np.sin(2 * np.pi * t / period) if period else np.zeros(n)
    )
    sigma = rng.uniform(0.5, 2.0) * scale / 8
    phi = rng.uniform(0.0, 0.8)
    noise = ar1(rng, n, phi, sigma)
    if rng.random() < 0.5:
        noise += rng.exponential(sigma, n) - sigma
    meta = {"тренд": kind, "період": period, "φ": phi}
    return 50 + trend + seasonal + noise, meta
