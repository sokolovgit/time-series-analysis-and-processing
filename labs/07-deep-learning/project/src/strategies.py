"""Стратегії екстраполяції нейромережею: на рівнях, на приростах і гібрид детермінованої складової з
мережею на залишку."""

import numpy as np

import features
import networks

STRATEGIES = ("рівні", "прирости", "гібрид")
DEGREES = (0, 1, 2)
HARMONICS = 2
CV_FOLDS = 3


def design(t, n, degree, period):
    """Регресори: (t/n)^k, k = 0…p, і гармоніки sin, cos(2πkt/s), k = 1…K."""
    u = np.asarray(t, dtype=float) / n
    cols = [u**k for k in range(degree + 1)]
    if period:
        for k in range(1, HARMONICS + 1):
            arg = 2 * np.pi * k * np.asarray(t, dtype=float) / period
            cols += [np.sin(arg), np.cos(arg)]
    return np.column_stack(cols)


def deterministic(y, degree, period):
    """МНК-оцінка D(t) = Σ βₖ (t/n)^k + Σ (aₖ sin + bₖ cos); повертає функцію D від будь-яких t."""
    n = len(y)
    beta = np.linalg.lstsq(design(np.arange(n), n, degree, period), y, rcond=None)[0]
    return lambda t: design(t, n, degree, period) @ beta


def select_degree(y, period):
    """Степінь тренду за CV з ковзним початком: модель на y[:c] прогнозує h = n/4 кроків,
    c = n − h, n − 1,5h, n − 2h; з кандидатів у межах 1-SE від найкращого береться найменший."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    h = n // 4
    cuts = [n - h - k * h // 2 for k in range(CV_FOLDS)]
    scores = {}
    for degree in DEGREES:
        errors = []
        for c in cuts:
            d = deterministic(y[:c], degree, period)
            errors.append(np.sqrt(np.mean((d(np.arange(c, c + h)) - y[c : c + h]) ** 2)))
        scores[degree] = (np.mean(errors), np.std(errors, ddof=1) / np.sqrt(CV_FOLDS))
    best = min(scores, key=lambda k: scores[k][0])
    limit = scores[best][0] + scores[best][1]
    return min(k for k in DEGREES if scores[k][0] <= limit), {k: v[0] for k, v in scores.items()}


def acf_weight(resid, steps):
    """Вага мережевого прогнозу залишку на кроці h: wₕ = ρ̂(h) до першого лагу,
    де ρ̂ ≤ 0, далі 0. Поза часом кореляції залишок непрогнозований, і зміщення рекурсії
    мережі лише додає похибку."""
    rho = features.acf(resid, range(1, steps + 1))
    stop = np.argmax(rho <= 0) if np.any(rho <= 0) else steps
    rho[stop:] = 0.0
    return rho


def standardize(x):
    m, s = np.mean(x), np.std(x)
    return (x - m) / s, m, s


def validation(model, kind, z, lookback):
    """Два індикатори на останніх VAL_SHARE пар, яких мережа не бачила в градієнті:
    СКВ прогнозу на крок і СКВ рекурсивного прогнозу на всю цю ділянку."""
    pairs = len(z) - lookback
    start = len(z) - round(networks.VAL_SHARE * pairs)
    step = networks.one_step(model, kind, z[start - lookback :], lookback)
    multi = networks.recursive(model, kind, z[:start], lookback, len(z) - start)
    return {
        "СКВ на крок": np.sqrt(np.mean((step - z[start:]) ** 2)),
        "СКВ рекурсії": np.sqrt(np.mean((multi - z[start:]) ** 2)),
    }


def run(strategy, kind, y, steps, spec):
    """Навчити мережу на вікні y за стратегією і повернути підгонку на вікні та прогноз на steps.

    рівні:    z = (y − m)/s, прогноз ŷ = m + s·ẑ;
    прирости: z = (Δy − m)/s, ŷ[n + h] = y[n − 1] + Σ Δŷ;
    гібрид:   r = y − D(t), z = (r − m)/s, ŷ = D(t) + m + s·ẑ.
    """
    y = np.asarray(y, dtype=float)
    n, lookback = len(y), spec["lookback"]
    if strategy == "рівні":
        base = y
    elif strategy == "прирости":
        base = np.diff(y)
    else:
        d = deterministic(y, spec["degree"], spec["period"])
        base = y - d(np.arange(n))
    z, m, s = standardize(base)
    model, info = networks.fit(kind, z, lookback, spec["units"], spec["loss"], spec["seed"])
    info |= validation(model, kind, z, lookback)
    step = m + s * networks.one_step(model, kind, z, lookback)
    future = m + s * networks.recursive(model, kind, z, lookback, steps)
    fitted = np.full(n, np.nan)
    if strategy == "рівні":
        fitted[lookback:] = step
        forecast = future
    elif strategy == "прирости":
        fitted[lookback + 1 :] = y[lookback:-1] + step
        forecast = y[-1] + np.cumsum(future)
    else:
        fitted[lookback:] = d(np.arange(lookback, n)) + step
        forecast = d(np.arange(n, n + steps)) + future
    return {"forecast": forecast, "fitted": fitted} | info
