"""KPI моделей і прогнозу: метрики, walk-forward перевірка на горизонтах 0,1W, 0,5W і W."""

import numpy as np
import pandas as pd

HORIZONS = (0.1, 0.5, 1.0)


def metrics(actual, predicted):
    """RMSE, MAE, MAPE, зміщення і R² = 1 − Σe²/Σ(y − ȳ)²."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    e = predicted - actual
    return {
        "RMSE": np.sqrt(np.mean(e**2)),
        "MAE": np.mean(np.abs(e)),
        "MAPE": np.mean(np.abs(e / actual)) * 100 if np.all(actual > 0) else np.nan,
        "зміщення": np.mean(e),
        "R²": 1 - np.sum(e**2) / np.sum((actual - actual.mean()) ** 2),
    }


def backtest(series, window, origins, forecaster, horizons=HORIZONS):
    """Для кожного старту s модель бачить y[s : s + W] і прогнозує на max(h)·W кроків;
    KPI на горизонті h – за першими h·W прогнозами, усереднені по стартах."""
    y = np.asarray(series, dtype=float)
    steps = int(max(horizons) * window)
    rows, paths = [], []
    for s in origins:
        train, test = y[s : s + window], y[s + window : s + window + steps]
        mean = np.asarray(forecaster(train, steps, s), dtype=float)
        paths.append((s, mean))
        for h in horizons:
            k = int(h * window)
            rows.append({"старт": s, "горизонт": h} | metrics(test[:k], mean[:k]))
    table = pd.DataFrame(rows)
    return table.groupby("горизонт").mean(numeric_only=True).drop(columns="старт"), table, paths
