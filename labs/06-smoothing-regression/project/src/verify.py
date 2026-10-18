"""Верифікація алгоритмів на рядах з відомими властивостями."""

import numpy as np

import exponential
import features
import regression
import synthetic

RUNS = 30
N = 300
SIGMA = 1.0


def ses_alpha(rng):
    """Для константи з шумом оптимальне α мале, для випадкового блукання – близьке до 1."""
    flat, walk = [], []
    for _ in range(RUNS):
        flat.append(
            exponential.fit(
                5 + rng.normal(0, SIGMA, N), exponential.CANDIDATES["просте (SES)"]
            ).params["smoothing_level"]
        )
        walk.append(
            exponential.fit(
                np.cumsum(rng.normal(0, SIGMA, N)), exponential.CANDIDATES["просте (SES)"]
            ).params["smoothing_level"]
        )
    print(
        f"   SES, {RUNS} реалізацій: α для константи + шум = "
        f"{np.mean(flat):.3f} ± {np.std(flat):.3f}, "
        f"α для блукання = {np.mean(walk):.3f} ± {np.std(walk):.3f}"
    )


def holt_slope(rng):
    """Хольт на лінійному тренді: нахил прогнозу збігається з нахилом тренду."""
    slopes = []
    for _ in range(RUNS):
        y = 10 + 0.2 * np.arange(N) + rng.normal(0, SIGMA, N)
        f = exponential.fit(y, exponential.CANDIDATES["Хольта"]).forecast(10)
        slopes.append(np.mean(np.diff(f)))
    print(
        f"   Хольт, нахил 0,2: оцінений нахил прогнозу = "
        f"{np.mean(slopes):.4f} ± {np.std(slopes):.4f}"
    )


def holt_winters(rng):
    """Хольт–Вінтерс на сезонному ряді: похибка прогнозу на сезон наперед ≈ σ шуму."""
    s, errors = 12, []
    t = np.arange(N + s)
    for _ in range(RUNS):
        y = 20 + 0.05 * t + 5 * np.sin(2 * np.pi * t / s) + rng.normal(0, SIGMA, N + s)
        f = exponential.fit(y[:N], exponential.SEASONAL["Хольта–Вінтерса"], s).forecast(s)
        errors.append(np.sqrt(np.mean((f - y[N:]) ** 2)))
    print(f"   Хольт–Вінтерс, s = {s}, σ = {SIGMA}: RMSE прогнозу на сезон = {np.mean(errors):.3f}")


def polynomial(rng):
    """Поліноміальна регресія відновлює тренд варіанта 8: похибка до істинного тренду ≪ σ."""
    errors = []
    for _ in range(RUNS):
        y, trend = synthetic.quadratic_normal(rng)
        _, fitted = regression.forecast_trend(y, 1, 2)
        errors.append(np.sqrt(np.mean((fitted - trend) ** 2)))
    print(
        f"   поліном 2-го степеня на квадратичному тренді (σ = {synthetic.SIGMA}): "
        f"СКВ до тренду = {np.mean(errors):.3f} "
        f"(теорія σ·√(3/n) = {synthetic.SIGMA * np.sqrt(3 / synthetic.N):.3f})"
    )


def lag_regression(rng):
    """Регресія на лаг AR(1): коефіцієнт ≈ φ, рекурсивний прогноз згасає як φʰ."""
    phi, coefs, gaps = 0.7, [], []
    for _ in range(RUNS):
        y = 10 + synthetic.ar1(rng, N, phi, SIGMA)
        x = regression.lag_matrix(y, 1)
        coefs.append(np.polyfit(x[:, 0], y[1:], 1)[0])
        f, _ = regression.forecast_lags(y, 20, 1, degree=0)
        mu = y.mean()
        theory = mu + phi ** np.arange(1, 21) * (y[-1] - mu)
        gaps.append(np.abs(f - theory).max())
    print(
        f"   регресія на лаг AR(1), φ = {phi}: "
        f"оцінка = {np.mean(coefs):.3f} ± {np.std(coefs):.3f}, "
        f"max |прогноз − теорія μ + φʰ(y − μ)| = {np.mean(gaps):.3f}"
    )


def season_detection(rng):
    """Детектор сезонності на рядах з відомим періодом і на випадкових блуканнях."""
    hits = total = false = walks = 0
    for _ in range(100):
        y, meta = synthetic.benchmark_series(rng)
        found, _ = features.period(y[:200])
        if meta["період"]:
            total += 1
            hits += found == meta["період"]
        else:
            false += found != 0
        walk = np.cumsum(rng.normal(0, 1, 200))
        walks += features.period(walk)[0] != 0
    print(
        f"   детектор сезонності: період знайдено точно у {hits} з {total} сезонних рядів, "
        f"хибний період у {false} з {100 - total} несезонних і у {walks} зі 100 блукань"
    )


def run(rng):
    ses_alpha(rng)
    holt_slope(rng)
    holt_winters(rng)
    polynomial(rng)
    lag_regression(rng)
    season_detection(rng)
