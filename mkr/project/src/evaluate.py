"""Оцінка якості: метрики, інформаційні критерії, бектест, тести залишків і порівняння прогнозів."""

import numpy as np
from scipy import stats

LAGS = 10


def metrics(actual, model):
    e = actual - model
    rss = e @ e
    return {
        "R2": 1.0 - rss / np.sum((actual - actual.mean()) ** 2),
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(rss / len(e)),
        "MAPE": 100.0 * np.mean(np.abs(e / actual)),
    }


def criteria(actual, model, k):
    """Скоригований R², AIC = n·ln(RSS/n) + 2k, BIC = n·ln(RSS/n) + k·ln n."""
    n = len(actual)
    e = actual - model
    rss = e @ e
    r2 = 1.0 - rss / np.sum((actual - actual.mean()) ** 2)
    return {
        "R2adj": 1.0 - (1.0 - r2) * (n - 1) / (n - k),
        "AIC": n * np.log(rss / n) + 2 * k,
        "BIC": n * np.log(rss / n) + k * np.log(n),
    }


def naive(train, h):
    """Наївний прогноз: ŷ(t+j) = y(t)."""
    return np.full(h, train[-1])


def drift(train, h):
    """Дрейф: ŷ(t+j) = y(t) + j·(y(t) − y(1)) / (t − 1)."""
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


def mean(train, h):
    return np.full(h, train.mean())


BASELINES = {"наївний": naive, "дрейф": drift, "середнє": mean}


def backtest(y, forecast, start, horizon):
    """Rolling origin з розширюваним вікном: для кожного t ≥ start модель навчається
    на y[:t] і прогнозує y[t : t+horizon].

    Повертає матрицю відносних помилок (ŷ − y) / y розміру origins × horizon,
    NaN там, де прогноз виходить за кінець ряду.
    """
    n = len(y)
    errors = np.full((n - start, horizon), np.nan)
    for row, origin in enumerate(range(start, n)):
        h = min(horizon, n - origin)
        errors[row, :h] = (forecast(y[:origin], h) - y[origin : origin + h]) / y[
            origin : origin + h
        ]
    return errors


def mape_by_horizon(errors):
    return 100.0 * np.nanmean(np.abs(errors), axis=0)


def diebold_mariano(e1, e2, h):
    """Тест Дібольда–Маріано з поправкою Харві–Лейборна–Ньюболда.

    d = e1² − e2², дисперсія середнього з автоковаріаціями до лагу h − 1,
    бо помилки h-крокових прогнозів з сусідніх точок перекриваються.
    H0: точність однакова. DM < 0 означає, що перший прогноз точніший.
    """
    mask = ~(np.isnan(e1) | np.isnan(e2))
    d = e1[mask] ** 2 - e2[mask] ** 2
    n = len(d)
    centered = d - d.mean()
    gamma = [centered @ centered / n] + [centered[k:] @ centered[:-k] / n for k in range(1, h)]
    variance = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = d.mean() / np.sqrt(variance)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, stats.t.cdf(dm, n - 1), n


def acf(e, lags=LAGS):
    e = e - e.mean()
    return np.array([e[k:] @ e[: len(e) - k] / (e @ e) for k in range(1, lags + 1)])


def residual_tests(e, k, lags=LAGS):
    """Перевірки адекватності: нульове середнє (t-тест), нормальність (Шапіро–Вілк),
    незалежність (Дарбін–Уотсон DW = Σ(eₜ − eₜ₋₁)² / Σeₜ², Льюнг–Бокс
    Q = n(n+2)·Σρₖ²/(n−k) ~ χ²(L − k))."""
    n = len(e)
    rho = acf(e, lags)
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    return {
        "mean_p": stats.ttest_1samp(e, 0.0).pvalue,
        "shapiro_p": stats.shapiro(e).pvalue,
        "dw": np.sum(np.diff(e) ** 2) / (e @ e),
        "ljung_q": q,
        "ljung_p": stats.chi2.sf(q, lags - k),
        "acf": rho,
    }


def empirical_band(errors, level=0.95):
    """Емпіричний інтервал прогнозу з помилок бектесту: квантилі відносної помилки
    для кожного горизонту. ŷ − y = r·y ⇒ y = ŷ / (1 + r)."""
    tail = (1.0 - level) / 2
    low = np.nanquantile(errors, tail, axis=0)
    high = np.nanquantile(errors, 1.0 - tail, axis=0)
    return low, high
