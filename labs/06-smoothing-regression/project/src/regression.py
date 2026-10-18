"""Регресійні моделі: поліном за часом, тренд + гармоніки, багатовимірна регресія на лаги ряду."""

import numpy as np
from sklearn.linear_model import Ridge

RIDGE_ALPHA = 1e-3
CV_FOLDS = 4


def design(t, degree, season=0, harmonics=0):
    """Регресори: степені нормованого часу 1..degree і гармоніки сезону."""
    t = np.asarray(t, dtype=float)
    columns = [t**k for k in range(1, degree + 1)]
    for j in range(1, harmonics + 1):
        columns += [np.cos(2 * np.pi * j * t / season), np.sin(2 * np.pi * j * t / season)]
    return np.column_stack(columns) if columns else np.empty((len(t), 0))


def scaled(t, n):
    """Час нормується на довжину вікна, щоб степені t не різнилися на порядки."""
    return np.asarray(t, dtype=float) / n


def forecast_trend(train, h, degree, season=0, harmonics=0):
    """МНК-регресія y(t) = β₀ + Σ βₖ·tᵏ + Σ гармонік, екстрапольована на h кроків."""
    n = len(train)
    t = np.arange(n + h)
    x = design(scaled(t, n), degree, season / n if season else 0, harmonics)
    model = Ridge(alpha=RIDGE_ALPHA).fit(x[:n], train)
    return model.predict(x[n:]), model.predict(x[:n])


def lag_matrix(y, lags):
    return np.column_stack([y[lags - k : len(y) - k] for k in range(1, lags + 1)])


def forecast_lags(train, h, lags, degree=1, season=0, harmonics=0):
    """Багатовимірна регресія: y(t) = f(y(t−1), …, y(t−p), t, гармоніки) з рекурсивним прогнозом.

    Модель навчається на один крок, а на h кроків прогноз будується підстановкою власних
    прогнозів замість невідомих майбутніх значень.
    """
    y = np.asarray(train, dtype=float)
    n = len(y)
    t = np.arange(n + h)
    exog = design(scaled(t, n), degree, season / n if season else 0, harmonics)
    x = np.column_stack([lag_matrix(y, lags), exog[lags:n]])
    model = Ridge(alpha=RIDGE_ALPHA).fit(x, y[lags:])
    fitted = np.r_[y[:lags], model.predict(x)]
    history = list(y[-lags:])
    out = np.empty(h)
    for k in range(h):
        row = np.r_[history[::-1][:lags], exog[n + k]]
        out[k] = model.predict(row[None, :])[0]
        history.append(out[k])
    return out, fitted


def candidates(season):
    out = {f"поліном {d}-го степеня": ("trend", {"degree": d}) for d in (1, 2, 3)}
    if season:
        for k in (1, 2, 3):
            out[f"тренд + {k} гарм."] = ("trend", {"degree": 1, "season": season, "harmonics": k})
    for p in (1, 3, 7):
        kw = {"lags": p, "degree": 1}
        if season:
            kw |= {"season": season, "harmonics": 2}
        out[f"лаги 1..{p}" + (" + гарм." if season else "")] = ("lags", kw)
    return out


def run(kind, train, h, kw):
    return forecast_trend(train, h, **kw) if kind == "trend" else forecast_lags(train, h, **kw)


def complexity(kind, kw):
    """Кількість коефіцієнтів моделі – міра складності для правила однієї стандартної похибки."""
    k = 1 + kw.get("degree", 0) + 2 * kw.get("harmonics", 0)
    return k + kw["lags"] if kind == "lags" else k


def cv_folds(train, kind, kw, horizon, folds=CV_FOLDS):
    """RMSE прогнозу на horizon кроків з кожного з folds стартів усередині вікна навчання.

    Перший старт – на третині вікна: короткі навчальні відрізки карають моделі з
    нестійкою екстраполяцією, а прогноз потім іде на горизонт, учетверо довший за CV.
    """
    train = np.asarray(train, dtype=float)
    cuts = np.linspace(len(train) // 3, len(train) - horizon, folds).astype(int)
    return np.array(
        [
            np.sqrt(np.mean((run(kind, train[:c], horizon, kw)[0] - train[c : c + horizon]) ** 2))
            for c in cuts
        ]
    )


def select(train, horizon, season=0):
    """Кандидати з CV-похибкою; обирається найпростіша модель, чия CV-похибка не більша за
    мінімальну плюс її стандартну похибку (правило 1-SE).

    Різниця CV-похибок у межах розкиду між стартами – шум, а складна модель (поліном 3-го
    степеня) екстраполюється значно гірше, ніж показує CV на чверті вікна.
    """
    rows, cv, se, size = {}, {}, {}, {}
    for name, (kind, kw) in candidates(season).items():
        _, fitted = run(kind, train, 1, kw)
        errors = cv_folds(train, kind, kw, horizon)
        cv[name] = float(np.sqrt(np.mean(errors**2)))
        se[name] = float(errors.std(ddof=1) / np.sqrt(len(errors)))
        size[name] = complexity(kind, kw)
        rows[name] = {
            "CV RMSE": cv[name],
            "SE": se[name],
            "параметрів": size[name],
            "СКВ підгонки": float(np.sqrt(np.mean((np.asarray(train) - fitted) ** 2))),
            "spec": (kind, kw),
        }
    best = min(cv, key=lambda k: cv[k])
    simple = [k for k in cv if cv[k] <= cv[best] + se[best]]
    choice = min(simple, key=lambda k: (size[k], cv[k]))
    return choice, rows


def forecast_auto(train, h, season=0, horizon=None):
    best, rows = select(train, horizon or max(len(train) // 5, 2), season)
    kind, kw = rows[best]["spec"]
    return run(kind, train, h, kw)[0]
