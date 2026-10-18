"""Верифікація методів на рядах з відомими властивостями."""

import numpy as np
import pandas as pd

import analysis
import patterns
import synthesis

RUNS = 100
LENGTHS = (347, 3653)
PHI = 0.8
SHIFT = 2
TREND, AMPLITUDE, PHASE, NOISE = 0.3e-3, 12.0, 0.4, 3.0
DECADE = 3652.5


def hurst_bias(rng):
    print(f"   показник Херста білого шуму (теоретично 0.5), {RUNS} реалізацій:")
    for n in LENGTHS:
        classic = [analysis.hurst(rng.normal(size=n), corrected=False)[0] for _ in range(RUNS)]
        corrected = [analysis.hurst(rng.normal(size=n))[0] for _ in range(RUNS)]
        print(
            f"     n = {n:>4}: класичний {np.mean(classic):.3f} ± {np.std(classic):.3f}, "
            f"скоригований {np.mean(corrected):.3f} ± {np.std(corrected):.3f}"
        )
    x = np.zeros(LENGTHS[1])
    e = rng.normal(size=len(x))
    for t in range(1, len(x)):
        x[t] = PHI * x[t - 1] + e[t]
    print(f"   AR(1) з φ = {PHI}: H = {analysis.hurst(x)[0]:.3f}")


def lag_recovery(rng):
    x = rng.normal(size=LENGTHS[1])
    y = np.r_[rng.normal(size=SHIFT), x[:-SHIFT]] + 0.5 * rng.normal(size=len(x))
    lag, r = patterns.peak_lag(x, y, 5)
    print(f"   крос-кореляція: y(t) = x(t − {SHIFT}) + шум → пік на лагу {lag:+d}, r = {r:.3f}")


def harmonic_recovery(rng):
    t = np.arange(LENGTHS[1])
    w = 2 * np.pi * t / analysis.YEAR
    y = 10 + TREND * t + AMPLITUDE * np.cos(w - PHASE) + NOISE * rng.normal(size=len(t))
    beta, resid = analysis.seasonal_fit(pd.DataFrame({"y": y}))
    amplitude = np.hypot(beta[2, 0], beta[3, 0])
    se = NOISE / np.sqrt(np.sum((t - t.mean()) ** 2)) * DECADE
    print(
        f"   гармонічна регресія: тренд за 10 років {beta[1, 0] * DECADE:.3f} "
        f"(задано {TREND * DECADE:.3f}, СКВ оцінки {se:.3f}), "
        f"амплітуда {amplitude:.3f} (задано {AMPLITUDE}), "
        f"σ залишку {resid['y'].std():.3f} (задано {NOISE})"
    )


def copula_roundtrip(rng):
    z = rng.gamma(2.0, size=(LENGTHS[1], 3)) - 2.0
    pools = synthesis.seasonal_pools(len(z))
    scores = synthesis.normal_scores(z, pools)
    back = synthesis.seasonal_quantiles(scores, [np.sort(z[p], axis=0) for p in pools])
    print(
        f"   копула: max |F⁻¹(Φ(s)) − z| = {np.abs(back - z).max():.2e}, "
        f"нормальні ранги m = {scores.mean():.3f}, σ = {scores.std():.3f}"
    )


def run(rng):
    hurst_bias(rng)
    lag_recovery(rng)
    harmonic_recovery(rng)
    copula_roundtrip(rng)
