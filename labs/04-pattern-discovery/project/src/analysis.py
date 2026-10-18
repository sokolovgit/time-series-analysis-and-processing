"""Властивості Time Series: статистики, стаціонарність, показник Херста, декомпозиція."""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

ALPHA = 0.05
YEAR = 365.25
HURST_MIN_WINDOW = 8


def describe(x):
    """Вибіркові характеристики: m, медіана, σ, коефіцієнт варіації σ/|m|, асиметрія, ексцес."""
    x = np.asarray(x, dtype=float)
    m, s = x.mean(), x.std(ddof=1)
    return {
        "n": len(x),
        "m": m,
        "медіана": np.median(x),
        "σ": s,
        "min": x.min(),
        "max": x.max(),
        "CV": s / abs(m) if m else np.nan,
        "асиметрія": stats.skew(x),
        "ексцес": stats.kurtosis(x),
    }


def stationarity(x):
    """ADF (H0: одиничний корінь) і KPSS (H0: стаціонарність).

    Ряд стаціонарний, якщо ADF відкидає H0, а KPSS – ні.
    """
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


def rescaled_range(x):
    """R/S одного вікна: розмах кумулятивних відхилень від середнього, поділений на σ."""
    y = np.cumsum(x - x.mean())
    s = x.std()
    return (y.max() - y.min()) / s if s > 0 else np.nan


def expected_rs(n):
    """Очікуване R/S для незалежних нормальних відліків (Anis–Lloyd, поправка Peters):

    E[R/S](n) = (n − ½)/n · Γ((n−1)/2) / (√π·Γ(n/2)) · Σ_{i=1}^{n−1} √((n − i)/i).
    """
    i = np.arange(1, n)
    tail = np.sqrt((n - i) / i).sum()
    if n <= 340:
        ratio = np.exp(gammaln((n - 1) / 2) - gammaln(n / 2)) / np.sqrt(np.pi)
    else:
        ratio = 1 / np.sqrt(n * np.pi / 2)
    return (n - 0.5) / n * ratio * tail


def rs_curve(x, min_window=HURST_MIN_WINDOW):
    """Середнє R/S по неперекривних вікнах для логарифмічної сітки довжин n."""
    x = np.asarray(x, dtype=float)
    sizes = np.unique(np.logspace(np.log10(min_window), np.log10(len(x) // 2), 20).astype(int))
    rs = []
    for n in sizes:
        blocks = x[: len(x) // n * n].reshape(-1, n)
        rs.append(np.nanmean([rescaled_range(b) for b in blocks]))
    rs = np.array(rs)
    ok = np.isfinite(rs) & (rs > 0)
    return sizes[ok], rs[ok]


def hurst(x, corrected=True, min_window=HURST_MIN_WINDOW):
    """Показник Херста за R/S-аналізом: E[R/S](n) = c·n^H, H – нахил у log-log координатах.

    Класична оцінка завищена на коротких вікнах, тому з corrected=True H = 0.5 + нахил
    log(R/S / E[R/S]) – відхилення від очікуваного для незалежних відліків.
    H ≈ 0.5 – незалежні прирости, H > 0.5 – персистентність, H < 0.5 – антиперсистентність.
    """
    sizes, rs = rs_curve(x, min_window)
    if corrected:
        excess = np.log(rs) - np.log([expected_rs(n) for n in sizes])
        h = 0.5 + np.polyfit(np.log(sizes), excess, 1)[0]
        return h, sizes, rs
    h = np.polyfit(np.log(sizes), np.log(rs), 1)[0]
    return h, sizes, rs


def autocorrelation(x, lags=40):
    return acf(np.asarray(x, dtype=float), nlags=lags, fft=True)


def decompose(series, period, robust=True):
    """STL: Y(t) = T(t) + S(t) + R(t), компоненти згладжуються LOESS."""
    return STL(series, period=period, robust=robust).fit()


def decompose_multi(series, periods, windows=None):
    """MSTL: Y(t) = T(t) + Σ S_i(t) + R(t) для кількох сезонностей."""
    return MSTL(series, periods=periods, windows=windows).fit()


def strength(trend, seasonal, resid):
    """Сила тренду і сезонності (Hyndman): F = max(0, 1 − Var(R) / Var(X + R))."""
    r = np.var(resid)
    # кусково-сталий ряд STL відтворює точно, залишку немає – і частка компонент не визначена
    if r <= 1e-9 * np.var(trend + seasonal + resid):
        return np.nan, np.nan
    return tuple(max(0.0, 1.0 - r / np.var(c + resid)) for c in (trend, seasonal))


def harmonic_design(t, k, period=YEAR, trend=True):
    """Матриця регресорів [1, t, cos(2πjt/P), sin(2πjt/P)], j = 1..k."""
    w = 2 * np.pi * t / period
    columns = [np.ones_like(t, dtype=float)]
    if trend:
        columns.append(t.astype(float))
    for j in range(1, k + 1):
        columns += [np.cos(j * w), np.sin(j * w)]
    return np.column_stack(columns)


def seasonal_fit(frame, k=3):
    """МНК-модель для кожного стовпця:

    Y(t) = a + b·t + Σ_{j=1}^{k} (c_j·cos(2πjt/P) + d_j·sin(2πjt/P)).


    Повертає коефіцієнти і аномалії r(t) = Y(t) − Ŷ(t). Амплітуда річного циклу – √(c₁² + d₁²).
    """
    x = harmonic_design(np.arange(len(frame)), k)
    beta, *_ = np.linalg.lstsq(x, frame.values, rcond=None)
    resid = pd.DataFrame(frame.values - x @ beta, index=frame.index, columns=frame.columns)
    return beta, resid


def harmonic_strength(frame, k=3):
    """Сила річної сезонності F_S = 1 − Var(R)/Var(S + R), де S – сума k гармонік.

    На добових даних STL вбирає в сезонну компоненту частину шуму (за 10 років кожен день року
    має лише 10 значень), тож для шумних показників на кшталт опадів F_S виходить завищеним.
    """
    beta, resid = seasonal_fit(frame, k)
    x = harmonic_design(np.arange(len(frame)), k)
    seasonal = x[:, 2:] @ beta[2:]
    total = np.var(seasonal + resid.values, axis=0)
    return pd.Series(1 - np.var(resid.values, axis=0) / total, index=frame.columns)


def table(rows):
    """Словник {назва: {показник: значення}} → DataFrame для друку."""
    return pd.DataFrame(rows).T
