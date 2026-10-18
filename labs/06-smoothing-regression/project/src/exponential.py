"""Експоненційне згладжування: просте, Хольта, із затуханням тренду, Хольта–Вінтерса."""

import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CANDIDATES = {
    "просте (SES)": {"trend": None, "damped_trend": False},
    "Хольта": {"trend": "add", "damped_trend": False},
    "Хольта із затуханням": {"trend": "add", "damped_trend": True},
}
SEASONAL = {
    "Хольта–Вінтерса": {"trend": "add", "damped_trend": False},
    "Хольта–Вінтерса із затуханням": {"trend": "add", "damped_trend": True},
    "сезонне без тренду": {"trend": None, "damped_trend": False},
}


def fit(y, spec, season=0, alpha=None):
    """Модель за специфікацією; season > 0 додає адитивну сезонність з цим періодом.

    Параметри α, β, γ, φ і початкові стани оцінюються мінімізацією суми квадратів похибок на крок.
    Заданий alpha фіксує α, решта параметрів оптимізується.
    """
    kwargs = dict(spec)
    if season:
        kwargs |= {"seasonal": "add", "seasonal_periods": season}
    with warnings.catch_warnings():
        # на коротких або зашумлених вікнах оптимізатор попереджає про збіжність – оцінки коректні
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            np.asarray(y, dtype=float), initialization_method="estimated", **kwargs
        )
        return model.fit(optimized=True, smoothing_level=alpha)


def candidates(season):
    out = {name: (spec, 0) for name, spec in CANDIDATES.items()}
    if season:
        out |= {name: (spec, season) for name, spec in SEASONAL.items()}
    return out


def select(y, season=0):
    """Усі кандидати з AIC і похибкою на крок; обирається модель з найменшим AIC."""
    rows = {}
    for name, (spec, s) in candidates(season).items():
        if s and len(y) < 2 * s:
            continue
        result = fit(y, spec, s)
        resid = np.asarray(y) - result.fittedvalues
        rows[name] = {
            "result": result,
            "AIC": result.aic,
            "СКВ на крок": np.sqrt(np.mean(resid**2)),
            "spec": (spec, s),
        }
    best = min(rows, key=lambda k: rows[k]["AIC"])
    return best, rows


def params(result):
    p = result.params
    out = {"α": p["smoothing_level"]}
    if p.get("smoothing_trend") is not None and not np.isnan(p.get("smoothing_trend", np.nan)):
        out["β"] = p["smoothing_trend"]
    if p.get("smoothing_seasonal") is not None and not np.isnan(
        p.get("smoothing_seasonal", np.nan)
    ):
        out["γ"] = p["smoothing_seasonal"]
    if p.get("damping_trend") is not None and not np.isnan(p.get("damping_trend", np.nan)):
        out["φ"] = p["damping_trend"]
    return out


ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, None)
CV_FOLDS = 4


def cv_rmse(train, spec, season, horizon, alpha=None, folds=CV_FOLDS):
    """RMSE прогнозу на horizon кроків з folds стартів усередині вікна навчання.

    Старти ті самі, що в регресії (з третини вікна), але не раніше двох повних сезонів,
    інакше сезонну модель не оцінити.
    """
    train = np.asarray(train, dtype=float)
    first = max(len(train) // 3, 2 * season)
    if first > len(train) - horizon:
        return np.inf
    cuts = np.linspace(first, len(train) - horizon, folds).astype(int)
    errors = []
    for c in cuts:
        f = fit(train[:c], spec, season, alpha).forecast(horizon)
        errors.append(np.mean((f - train[c : c + horizon]) ** 2))
    return float(np.sqrt(np.mean(errors)))


def select_cv(train, season, horizon):
    """Структура і α за похибкою на цільовому горизонті, а не на кроці.

    Спершу серед структур з оптимізованими параметрами обирається найкраща за CV,
    потім для неї перебирається фіксоване α (None – оптимізоване на крок).
    """
    structures = {
        name: cv_rmse(train, spec, s, horizon) for name, (spec, s) in candidates(season).items()
    }
    best = min(structures, key=lambda k: structures[k])
    spec, s = candidates(season)[best]
    alphas = {a: cv_rmse(train, spec, s, horizon, a) for a in ALPHAS}
    alpha = min(alphas, key=lambda a: alphas[a])
    return best, alpha, structures, alphas


def forecast_auto(train, h, season=0):
    best, rows = select(train, season)
    return rows[best]["result"].forecast(h)


def forecast_cv(train, h, season=0, horizon=None):
    best, alpha, _, _ = select_cv(train, season, horizon or max(len(train) // 5, 2))
    spec, s = candidates(season)[best]
    return fit(train, spec, s, alpha).forecast(h)
