"""Статистичне навчання поліноміальної моделі за МНК. Час нормований: τ = t/n."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEGREES = (1, 2, 3, 4, 5)
HOLDOUT = 0.2


def design(n, degree, start=0, scale=None):
    """Матриця плану F зі стовпцями 1, τ, τ², … , τᵐ."""
    scale = n if scale is None else scale
    t = np.arange(start, start + n, dtype=float) / scale
    return np.vander(t, degree + 1, increasing=True)


def fit(data, degree):
    """МНК у замкненій формі: C = (FᵀF)⁻¹FᵀY."""
    n = len(data)
    f = design(n, degree)
    coeffs = np.linalg.solve(f.T @ f, f.T @ data)
    return f @ coeffs, coeffs


def extrapolate(coeffs, n, horizon):
    """Прогноз на horizon відліків за межі вибірки."""
    return design(horizon, len(coeffs) - 1, start=n, scale=n) @ coeffs


def scores(actual, model):
    """Показники якості навчання."""
    mse = mean_squared_error(actual, model)
    return {
        "R2": r2_score(actual, model),
        "MAE": mean_absolute_error(actual, model),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
    }


def select_degree(data, degrees=DEGREES, holdout=HOLDOUT):
    """Степінь з найменшою RMSE на відкладеному хвості, який модель не бачила."""
    cut = int(len(data) * (1.0 - holdout))
    table = {}
    for m in degrees:
        _, coeffs = fit(data[:cut], m)
        table[m] = {
            "in": scores(data[:cut], design(cut, m) @ coeffs)["RMSE"],
            "out": scores(data[cut:], extrapolate(coeffs, cut, len(data) - cut))["RMSE"],
        }
    return min(table, key=lambda m: table[m]["out"]), table, cut
