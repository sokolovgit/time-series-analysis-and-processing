"""Алгоритми ковзного вікна: згладжування і прогноз за останнім вікном спостережень."""

import numpy as np
import pandas as pd

YEAR = 365.25


def window_weights(w, kind):
    """Ваги вікна від найстаршого відліку до найновішого: SMA – рівні, WMA – лінійні 1..w."""
    weights = np.ones(w) if kind == "SMA" else np.arange(1, w + 1, dtype=float)
    return weights / weights.sum()


def smooth(y, w, kind="SMA"):
    """Каузальне згладжування: оцінка в момент t – за вікном y(t−w+1..t), без майбутніх відліків."""
    s = pd.Series(np.asarray(y, dtype=float))
    if kind == "median":
        out = s.rolling(w, min_periods=1).median()
    elif kind == "WMA":
        weights = window_weights(w, "WMA")
        out = s.rolling(w).apply(lambda v: v @ weights, raw=True)
        out[: w - 1] = s[: w - 1].expanding().mean()
    else:
        out = s.rolling(w, min_periods=1).mean()
    return out.to_numpy()


def ema(y, alpha):
    """Експоненційне згладжування: s(t) = α·y(t) + (1 − α)·s(t−1), s(0) = y(0)."""
    y = np.asarray(y, dtype=float)
    s = np.empty_like(y)
    s[0] = y[0]
    for t in range(1, len(y)):
        s[t] = alpha * y[t] + (1 - alpha) * s[t - 1]
    return s


def local_line(y, w):
    """Ковзний МНК: пряма за останніми w відліками, повертає нахил і значення в кінці вікна."""
    tail = np.asarray(y[-w:], dtype=float)
    slope, level = np.polyfit(np.arange(-len(tail) + 1, 1), tail, 1)
    return level, slope


def forecast_flat(train, h, w, kind="SMA"):
    """Прогноз рівнем останнього вікна: SMA, WMA або медіана."""
    tail = np.asarray(train[-w:], dtype=float)
    level = np.median(tail) if kind == "median" else tail @ window_weights(len(tail), kind)
    return np.full(h, level)


def forecast_ema(train, h, alpha):
    return np.full(h, ema(train, alpha)[-1])


def forecast_line(train, h, w):
    """Екстраполяція прямої ковзного МНК: ŷ(t + k) = рівень + нахил·k."""
    level, slope = local_line(train, w)
    return level + slope * np.arange(1, h + 1)


def forecast_seasonal(train, h, years, half_width):
    """Сезонне ковзне середнє: прогноз на день d – середнє за днями d ± half_width
    в останніх years роках, що вже є у вибірці (для горизонту понад рік – у давніших)."""
    train = np.asarray(train, dtype=float)
    n = len(train)
    out = np.empty(h)
    for k in range(h):
        first = int(np.ceil((k + 1 + half_width) / YEAR))
        values = []
        for m in range(first, first + years):
            centre = n + k - round(m * YEAR)
            lo, hi = max(centre - half_width, 0), min(centre + half_width + 1, n)
            if lo < hi:
                values.append(train[lo:hi])
        out[k] = np.concatenate(values).mean() if values else train[-1]
    return out
