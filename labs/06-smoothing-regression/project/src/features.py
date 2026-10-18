"""Властивості ряду: статистики, стаціонарність, період сезонності та ознаки для селектора."""

import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

MIN_PERIOD, SEASON_SHARE, LOCAL_RATIO, PAD = 4, 0.1, 12, 16
SELECTOR_FEATURES = (
    "R² тренду",
    "кривизна",
    "сезонність",
    "ACF(1) залишку",
    "ACF(1) приростів",
    "p ADF",
    "шум/рівень",
)


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


def selector_features(y, known_period=None):
    """Ознаки, за якими селектор обирає між експоненційною і регресійною моделлю."""
    y = np.asarray(y, dtype=float)
    r2_lin, _ = poly_r2(y, 1)
    r2_quad, resid = poly_r2(y, 2)
    _, share = period(y, known_period)
    d = np.diff(y)
    with warnings.catch_warnings():
        # на гладких рядах регресія ADF вироджена, p-значення від цього не змінюється
        warnings.simplefilter("ignore")
        adf_p = adfuller(y, autolag="AIC")[1]
    return {
        "R² тренду": r2_quad,
        "кривизна": r2_quad - r2_lin,
        "сезонність": share,
        "ACF(1) залишку": np.corrcoef(resid[1:], resid[:-1])[0, 1],
        "ACF(1) приростів": np.corrcoef(d[1:], d[:-1])[0, 1],
        "p ADF": adf_p,
        "шум/рівень": d.std() / y.std(),
    }
