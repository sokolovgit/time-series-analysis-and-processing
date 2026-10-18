"""Якість прогнозу: метрики, walk-forward перевірка на горизонтах 0.5–2W, крос-валідація."""

import numpy as np
import pandas as pd

HORIZONS = (0.5, 1.0, 1.5, 2.0)


def metrics(actual, predicted, scale, interval=None):
    """RMSE, MAE, зміщення, MAPE і MASE = MAE / scale, де scale – MAE наївного
    прогнозу на крок у вікні навчання. MASE < 1 – краще за «завтра як сьогодні» на крок."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    e = predicted - actual
    out = {
        "RMSE": np.sqrt(np.mean(e**2)),
        "MAE": np.mean(np.abs(e)),
        "зміщення": np.mean(e),
        "MAPE": np.mean(np.abs(e / actual)) * 100 if np.all(actual > 0) else np.nan,
        "MASE": np.mean(np.abs(e)) / scale,
    }
    if interval is not None:
        out["покриття"] = np.mean((actual >= interval[:, 0]) & (actual <= interval[:, 1]))
    return out


def naive_scale(train):
    return np.mean(np.abs(np.diff(np.asarray(train, dtype=float))))


def backtest(series, window, origins, forecaster, horizons=HORIZONS):
    """Walk-forward: для кожного старту s модель бачить y[s : s + W] і прогнозує на 2W кроків.

    Якість на горизонті h – за першими h·W прогнозами, усереднена по стартах.
    forecaster(train, h, start) повертає прогноз або пару (прогноз, 90 % інтервал).
    """
    y = np.asarray(series, dtype=float)
    steps = int(max(horizons) * window)
    rows, paths = [], []
    for s in origins:
        train, test = y[s : s + window], y[s + window : s + window + steps]
        result = forecaster(train, steps, s)
        mean, interval = result if isinstance(result, tuple) else (result, None)
        mean = np.asarray(mean, dtype=float)
        paths.append((s, mean, interval))
        scale = naive_scale(train)
        for h in horizons:
            k = int(h * window)
            part = None if interval is None else interval[:k]
            rows.append({"старт": s, "горизонт": h} | metrics(test[:k], mean[:k], scale, part))
    table = pd.DataFrame(rows)
    return table.groupby("горизонт").mean(numeric_only=True).drop(columns="старт"), table, paths


def point(result):
    return result[0] if isinstance(result, tuple) else result


def cv_score(train, forecaster, horizon, folds, start=0):
    """RMSE прогнозу на horizon кроків з folds стартів усередині вікна навчання (rolling origin).

    Кожен внутрішній старт бачить train[:cut] – лише минуле відносно перевірочного відрізка.
    """
    train = np.asarray(train, dtype=float)
    cut_points = np.linspace(len(train) // 2, len(train) - horizon, folds).astype(int)
    errors = []
    for cut in cut_points:
        e = point(forecaster(train[:cut], horizon, start)) - train[cut : cut + horizon]
        errors.append(np.mean(e**2))
    return np.sqrt(np.mean(errors))


def tune(train, make_forecaster, candidates, horizon, folds, start=0):
    """Параметр з мінімальною CV-похибкою та вся крива похибки."""
    scores = {c: cv_score(train, make_forecaster(c), horizon, folds, start) for c in candidates}
    return min(scores, key=lambda c: scores[c]), scores
