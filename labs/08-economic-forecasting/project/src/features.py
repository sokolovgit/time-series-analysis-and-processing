"""Властивості ряду: статистики, стаціонарність, період сезонності, автокореляція."""

import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

MIN_PERIOD, SEASON_SHARE, LOCAL_RATIO, PAD = 4, 0.1, 12, 16
YEAR, SIGNIFICANCE = 12, 3.0


def describe(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "m": x.mean(),
        "σ": x.std(ddof=1),
        "асиметрія": stats.skew(x),
        "ексцес": stats.kurtosis(x),
    }


def stationarity(x):
    x = np.asarray(x, dtype=float)
    adf_p = adfuller(x, autolag="AIC")[1]
    with warnings.catch_warnings():
        # KPSS обрізає p до меж таблиці [0.01, 0.1] і попереджає про це
        warnings.simplefilter("ignore")
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    return {"p_ADF": adf_p, "p_KPSS": kpss_p, "стаціонарний": adf_p < 0.05 and kpss_p > 0.05}


def poly_r2(y, degree):
    t = np.arange(len(y)) / len(y)
    fit = np.polyval(np.polyfit(t, y, degree), t)
    return 1 - np.var(y - fit) / np.var(y), y - fit


def spectrum(x):
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return power, np.fft.rfftfreq(len(x)), power[1:].sum()


def period(y, known=None):
    """Період сезонності: заданий або домінантна частота періодограми ряду без квадратичного тренду.

    Сезонний пік має бути локальним максимумом спектра рівнів з часткою ≥ SEASON_SHARE і
    виділятися в спектрі перших різниць: потужність на цій частоті більша за медіану сусідніх
    частот у LOCAL_RATIO разів. Друга умова відсіює випадкове блукання: його рівні мають
    «піки» на низьких частотах, а спектр різниць пласкі. Положення піку уточнюється
    періодограмою з доповненням нулями, бо сітка частот 1/n груба для періоду 24 при n = 200.
    Повертає (період, частка піку); 0 – сезонності немає.
    """
    if known:
        return known, 1.0
    y = np.asarray(y, dtype=float)
    _, resid = poly_r2(y, 2)
    power, freqs, total = spectrum(resid)
    peaks = [
        i
        for i in range(3, len(freqs) - 1)
        if freqs[i] <= 1 / MIN_PERIOD and power[i] > power[i - 1] and power[i] > power[i + 1]
    ]
    if not peaks:
        return 0, 0.0
    i = max(peaks, key=lambda k: power[k])
    share = power[i - 1 : i + 2].sum() / total
    d_power, d_freqs, _ = spectrum(np.diff(y))
    j = int(np.argmin(np.abs(d_freqs - freqs[i])))
    ratio = d_power[max(j - 1, 1) : j + 2].max() / np.median(d_power[max(j - 12, 1) : j + 13])
    if share < SEASON_SHARE or ratio < LOCAL_RATIO:
        return 0, share
    fine = np.fft.rfftfreq(PAD * len(resid))
    fine_power = np.abs(np.fft.rfft(resid - resid.mean(), PAD * len(resid))) ** 2
    band = (fine >= freqs[i - 1]) & (fine <= freqs[i + 1])
    return round(1 / fine[band][np.argmax(fine_power[band])]), share


def acf(x, lags):
    x = np.asarray(x, dtype=float) - np.mean(x)
    return np.array([np.sum(x[k:] * x[: len(x) - k]) / np.sum(x * x) for k in lags])


def annual(y, period=YEAR):
    """Річна сезонність місячного ряду: ACF логарифмічних приростів на лагу 12 перевищує
    3/√n – межу значущості нульової автокореляції з запасом. Прирости, а не рівні: у рівнях
    тренд дає високу ACF на будь-якому лагу. Повертає (період або 0, ACF(12))."""
    r = np.diff(np.log(np.asarray(y, dtype=float)))
    rho = float(acf(r, [period])[0])
    return (period if rho > SIGNIFICANCE / np.sqrt(len(r)) else 0), rho
