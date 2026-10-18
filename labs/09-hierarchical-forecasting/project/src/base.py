"""Базові прогнози вузлів: експоненційне згладжування з вибором моделі за AICc."""

import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CANDIDATES = {
    "SES": {"trend": None, "damped_trend": False},
    "Хольта": {"trend": "add", "damped_trend": False},
    "Хольта із затуханням": {"trend": "add", "damped_trend": True},
}


def fit(y):
    """Модель на ln y з найменшим AICc; повертає назву і результат statsmodels.

    Логарифм тримає прогноз додатним і робить тренд мультиплікативним, що природно для
    кількості народжень, яка змінюється на відсотки за рік.
    """
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, spec in CANDIDATES.items():
            model = ExponentialSmoothing(np.log(y), initialization_method="estimated", **spec)
            result = model.fit(optimized=True)
            if best is None or result.aicc < best[1].aicc:
                best = (name, result)
    return best


def forecast(Y, horizon):
    """Прогнози всіх вузлів на horizon кроків: F розміром n×h, залишки E розміром T×n, моделі."""
    F, E, chosen = [], [], []
    for y in Y.T:
        name, result = fit(y)
        F.append(np.exp(result.forecast(horizon)))
        E.append(y - np.exp(result.fittedvalues))
        chosen.append(name)
    return np.array(F), np.array(E).T, chosen
