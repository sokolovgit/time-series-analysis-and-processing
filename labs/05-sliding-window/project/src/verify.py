"""Верифікація алгоритмів на рядах з відомими властивостями."""

import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample

import arima
import learning
import smoothing

RUNS = 50
N = 300
PHI, THETA = 0.5, 0.3
SLOPE, W = 0.2, 11


def arima_recovery(rng):
    phis, thetas, hits = [], [], 0
    for _ in range(RUNS):
        dy = arma_generate_sample([1, -PHI], [1, THETA], N, distrvs=rng.standard_normal)
        y = np.cumsum(dy)
        model = arima.fit(y, (1, 1, 1))
        phis.append(model.params[0])
        thetas.append(model.params[1])
        order, _, _, _ = arima.select(y, 2, (0, 1), 2)
        hits += order == (1, 1, 1)
    print(f"   ARIMA(1,1,1), φ = {PHI}, θ = {THETA}, n = {N}, {RUNS} реалізацій:")
    print(
        f"     оцінки φ = {np.mean(phis):.3f} ± {np.std(phis):.3f}, "
        f"θ = {np.mean(thetas):.3f} ± {np.std(thetas):.3f}"
    )
    print(f"     AIC + Льюнг–Бокс обрали (1,1,1) у {hits} з {RUNS}")


def coverage(rng):
    covered = []
    for _ in range(RUNS):
        y = arma_generate_sample([1, -0.8], [1], N + 50, distrvs=rng.standard_normal)
        model = arima.fit(y[:N], (1, 0, 0))
        _, interval = arima.forecast(model, 50)
        covered.append(np.mean((y[N:] >= interval[:, 0]) & (y[N:] <= interval[:, 1])))
    print(f"   покриття 90 % інтервалу на AR(1), φ = 0,8: {np.mean(covered):.3f}")


def lag_on_trend():
    t = np.arange(200, dtype=float)
    y = SLOPE * t
    sma = smoothing.smooth(y, W)
    alpha = 2 / (W + 1)
    ema = smoothing.ema(y, alpha)
    print(
        f"   лінійний тренд з нахилом {SLOPE}: запізнення SMA (w = {W}) = {(y - sma)[-1]:.3f} "
        f"(теорія (w − 1)/2·b = {(W - 1) / 2 * SLOPE:.3f}), "
        f"EMA (α = {alpha:.3f}) = {(y - ema)[-1]:.3f} "
        f"(теорія (1 − α)/α·b = {(1 - alpha) / alpha * SLOPE:.3f})"
    )
    line = smoothing.forecast_line(y, 10, W)
    ab = learning.forecast_alpha_beta(y, 10, 20)
    truth = SLOPE * (200 + np.arange(10))
    print(
        f"   екстраполяція тренду: max |похибка| ковзного МНК = {np.abs(line - truth).max():.1e}, "
        f"α-β фільтра = {np.abs(ab - truth).max():.1e}"
    )


def seasonal_exact():
    t = np.arange(4 * 365)
    y = 10 * np.cos(2 * np.pi * t / smoothing.YEAR)
    f = smoothing.forecast_seasonal(y[: 2 * 365], 730, 2, 0)
    print(
        f"   сезонне MA на чистому річному циклі: "
        f"max |похибка| = {np.abs(f - y[730:1460]).max():.3f} "
        f"при амплітуді 10 (зсув через 365 замість 365,25 діб)"
    )


def run(rng):
    arima_recovery(rng)
    coverage(rng)
    lag_on_trend()
    seasonal_exact()
