"""Властивості ряду, що визначають вибір алгоритму: статистики, стаціонарність, ACF і PACF."""

import warnings

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

ALPHA = 0.05


def describe(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "m": x.mean(),
        "σ": x.std(ddof=1),
        "min": x.min(),
        "max": x.max(),
        "асиметрія": stats.skew(x),
        "ексцес": stats.kurtosis(x),
    }


def stationarity(x):
    """ADF (H0: одиничний корінь) і KPSS (H0: стаціонарність)."""
    x = np.asarray(x, dtype=float)
    adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
    with warnings.catch_warnings():
        # KPSS повертає p лише в межах таблиці [0.01, 0.1] і попереджає про обрізання
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
    return {
        "ADF": adf_stat,
        "p_ADF": adf_p,
        "KPSS": kpss_stat,
        "p_KPSS": kpss_p,
        "стаціонарний": adf_p < ALPHA and kpss_p > ALPHA,
    }


def correlograms(x, lags):
    x = np.asarray(x, dtype=float)
    return acf(x, nlags=lags, fft=True), pacf(x, nlags=lags), 1.96 / np.sqrt(len(x))


def white_noise_p(x, lags=20):
    """p-значення Льюнга–Бокса: H0 – ряд некорельований."""
    return float(acorr_ljungbox(np.asarray(x, dtype=float), lags=[lags])["lb_pvalue"].iloc[0])


def anomalies(y, harmonics=3, period=365.25):
    """Відхилення від МНК-моделі a + b·t + Σ гармонік – для ACF/PACF сезонного ряду."""
    t = np.arange(len(y))
    columns = [np.ones(len(t)), t]
    for j in range(1, harmonics + 1):
        columns += [np.cos(2 * np.pi * j * t / period), np.sin(2 * np.pi * j * t / period)]
    x = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(x, np.asarray(y, dtype=float), rcond=None)
    return np.asarray(y, dtype=float) - x @ beta
