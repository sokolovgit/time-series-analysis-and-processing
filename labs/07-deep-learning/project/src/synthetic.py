"""Синтетичні ряди варіанта 8 з ЛР 1 і тестові сигнали для верифікації."""

import numpy as np

N = 500
QUADRATIC = (1e-4, 1e-2, 10.0)
SLOPE, LEVEL = 0.05, 10.0
SIGMA = 5.0


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
