"""Методи статистичного навчання для порівняння: МНК-поліном, МНК з гармоніками, α-β фільтр."""

import numpy as np

from arima import fourier


def forecast_poly(train, h, degree=2):
    """МНК-поліном за всім вікном спостереження, екстрапольований на h кроків."""
    t = np.arange(len(train))
    coefs = np.polyfit(t, np.asarray(train, dtype=float), degree)
    return np.polyval(coefs, len(train) + np.arange(h))


def forecast_harmonic(train, h, harmonics=3, start=0):
    """МНК-модель y(t) = a + b·t + Σ гармонік за всім вікном, екстрапольована на h кроків."""
    t = start + np.arange(len(train) + h)
    x = np.column_stack([np.ones(len(t)), t, fourier(t, harmonics)])
    beta, *_ = np.linalg.lstsq(x[: len(train)], np.asarray(train, dtype=float), rcond=None)
    return x[len(train) :] @ beta


def alpha_beta(y, memory):
    """α-β фільтр з обмеженою пам'яттю N: α = 2(2N − 1)/(N(N + 1)), β = 6/(N(N + 1)).

    На перших N кроках пам'ять зростає (n = k + 1), далі коефіцієнти фіксовані.
    Повертає оцінку рівня і швидкості на останньому кроці.
    """
    x, v = float(y[0]), 0.0
    for k in range(1, len(y)):
        n = min(k + 1, memory)
        d = n * (n + 1)
        alpha, beta = 2 * (2 * n - 1) / d, 6 / d
        x_pred = x + v
        residual = y[k] - x_pred
        x, v = x_pred + alpha * residual, v + beta * residual
    return x, v


def forecast_alpha_beta(train, h, memory):
    """Екстраполяція стану α-β фільтра: ŷ(t + k) = x + v·k."""
    x, v = alpha_beta(np.asarray(train, dtype=float), memory)
    return x + v * np.arange(1, h + 1)
