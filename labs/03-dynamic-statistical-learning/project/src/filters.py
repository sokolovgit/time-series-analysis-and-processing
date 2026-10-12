"""Рекурентні α-β (order = 2) та α-β-γ (order = 3) фільтри, крок T = 1."""

from functools import cache

import numpy as np


def gains(n, order):
    """Коефіцієнти фільтра на n-му кроці – це рекурентна форма МНК за n вимірами.

    α-β:   α = 2(2n − 1) / n(n + 1),  β = 6 / n(n + 1)
    α-β-γ: α = 3(3n² − 3n + 2) / n(n + 1)(n + 2),  β = 18(2n − 1) / n(n + 1)(n + 2),
           γ = 60 / n(n + 1)(n + 2)
    """
    if order == 2:
        d = n * (n + 1)
        return 2 * (2 * n - 1) / d, 6 / d, 0.0
    d = n * (n + 1) * (n + 2)
    return 3 * (3 * n * n - 3 * n + 2) / d, 18 * (2 * n - 1) / d, 60 / d


def predict(x, v, a):
    """Екстраполяція стану на крок: x⁻ = x + v + a/2,  v⁻ = v + a."""
    return x + v + a / 2, v + a


def correct(x_pred, v_pred, a, residual, g):
    """Корекція за нев'язкою r = y − x⁻: x = x⁻ + αr,  v = v⁻ + βr,  a = a + γr."""
    alpha, beta, gamma = g
    return x_pred + alpha * residual, v_pred + beta * residual, a + gamma * residual


def run(data, order, memory=None):
    """Фільтр по всій вибірці. memory=None – пам'ять зростає, інакше n = min(k, memory).

    data може мати форму (n,) або (n, M) – тоді M реалізацій обробляються одночасно.
    Повертає оцінки рівня та нев'язки прогнозу на крок.
    """
    x = np.zeros(data.shape[1:])
    v = np.zeros_like(x)
    a = np.zeros_like(x)
    estimate = np.empty_like(data, dtype=float)
    innovation = np.empty_like(estimate)
    for k in range(len(data)):
        n = k + 1 if memory is None else min(k + 1, memory)
        x_pred, v_pred = predict(x, v, a)
        innovation[k] = data[k] - x_pred
        x, v, a = correct(x_pred, v_pred, a, innovation[k], gains(n, order))
        estimate[k] = x
    # перші order відліків фільтр лише набирає стан, прогноз ще не визначений
    innovation[:order] = np.nan
    return estimate, innovation


@cache
def prediction_factor(n, order):
    """Дисперсія прогнозу МНК на крок уперед у частках σ²: ρ = hᵀ(FᵀF)⁻¹h.

    Час τ = (t − n)/n, тому точка прогнозу – τ = 0 і h = (1, 0, 0), а ρ = [(FᵀF)⁻¹]₀₀.
    """
    if n < order:
        return np.inf
    tau = (np.arange(n, dtype=float) - n) / n
    f = np.vander(tau, order, increasing=True)
    return float(np.linalg.inv(f.T @ f)[0, 0])


def fit_state(window, order):
    """Стан (x, v, a) в останній точці вікна з МНК-полінома степеня order − 1."""
    n = len(window)
    t = np.arange(n, dtype=float) - (n - 1)
    c = np.polynomial.polynomial.polyfit(t, window, min(order, n) - 1)
    c = np.pad(c, (0, 3 - len(c)))
    return c[0], c[1], 2 * c[2] if order == 3 else 0.0
