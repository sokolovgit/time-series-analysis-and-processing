"""Автокореляційні моделі: AR, ARMA, ARIMA і регресія на гармоніки з ARMA-похибкою."""

import itertools
import warnings

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

YEAR = 365.25
ALPHA = 0.10
LB_LAGS = 20


def fourier(t, k):
    """Гармоніки річного циклу [cos(2πjt/P), sin(2πjt/P)], j = 1..k."""
    t = np.asarray(t, dtype=float)
    return np.column_stack(
        [f(2 * np.pi * j * t / YEAR) for j in range(1, k + 1) for f in (np.cos, np.sin)]
    )


def fit(y, order, harmonics=0, start=0, drift=False):
    """ARIMA(p, d, q). З harmonics > 0 – регресія на тренд і гармоніки з ARMA-похибкою:

    y(t) = a + b·t + Σ гармонік + u(t), u ~ ARMA(p, q).
    start – номер першого відліку y від початку ряду, щоб фаза гармонік не залежала від вікна.
    drift – для d = 1 додає сталий нахил: Δy(t) = c + ARMA.
    """
    exog = fourier(start + np.arange(len(y)), harmonics) if harmonics else None
    if harmonics:
        trend = "ct"
    elif order[1] == 0:
        trend = "c"
    else:
        trend = "t" if drift else "n"
    with warnings.catch_warnings():
        # на коротких вікнах МНК-старт часто дає нестаціонарні параметри – statsmodels
        # переходить на нульовий старт і попереджає, але оцінка коректна
        warnings.simplefilter("ignore")
        model = ARIMA(np.asarray(y, dtype=float), exog=exog, order=order, trend=trend).fit()
    model.harmonics, model.start, model.n, model.drift = harmonics, start, len(y), drift
    return model


def forecast(model, h):
    """Прогноз на h кроків і 90 % інтервал."""
    exog = None
    if model.harmonics:
        exog = fourier(model.start + model.n + np.arange(h), model.harmonics)
    result = model.get_forecast(h, exog=exog)
    return result.predicted_mean, result.conf_int(alpha=ALPHA)


def ljung_box(model):
    """p-значення тесту Льюнга–Бокса залишків: H0 – залишки некорельовані."""
    resid = model.resid[max(model.model.k_diff, 1) :]
    return float(acorr_ljungbox(resid, lags=[LB_LAGS])["lb_pvalue"].iloc[0])


def grid(y, p_max, d_values, q_max, harmonics=(0,), start=0, drifts=(False,)):
    """Перебір порядків за AIC. Повертає список (AIC, BIC, p_LB, order, harmonics, drift),
    відсортований за AIC. Дрейф має сенс лише для d = 1."""
    rows = []
    space = itertools.product(range(p_max + 1), d_values, range(q_max + 1), harmonics, drifts)
    for p, d, q, k, drift in space:
        if drift and d == 0:
            continue
        try:
            model = fit(y, (p, d, q), k, start, drift)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rows.append((model.aic, model.bic, ljung_box(model), (p, d, q), k, drift))
    return sorted(rows, key=lambda row: row[0])


def select(y, p_max, d_values, q_max, harmonics=(0,), start=0, drifts=(False,)):
    """Найкраща за AIC модель серед тих, чиї залишки проходять тест Льюнга–Бокса;
    якщо таких немає – найкраща за AIC загалом."""
    rows = grid(y, p_max, d_values, q_max, harmonics, start, drifts)
    passed = [row for row in rows if row[2] > 0.05]
    best = (passed or rows)[0]
    return best[3], best[4], best[5], rows
