"""Метод найменших квадратів: поліноміальні та експоненційна моделі тренду.

Час нормований: τ = t / n, де n – довжина навчальної вибірки, інакше FᵀF при
степені 3–4 погано обумовлена.
"""

import numpy as np
from scipy import stats

MODELS = {
    "лінійна": ("poly", 1),
    "квадратична": ("poly", 2),
    "кубічна": ("poly", 3),
    "поліном 4-го ст.": ("poly", 4),
    "експоненційна": ("exp", 1),
}
LEVEL = 0.95


def design(tau, degree):
    """Матриця плану F зі стовпцями 1, τ, τ², … , τᵐ."""
    return np.vander(np.asarray(tau, dtype=float), degree + 1, increasing=True)


def solve(f, y):
    """МНК у замкненій формі: C = (FᵀF)⁻¹FᵀY."""
    return np.linalg.solve(f.T @ f, f.T @ y)


def fit(y, model):
    """Навчання моделі на ряді y.

    Експоненційна y = a·e^(bτ) зводиться до лінійної: ln y = ln a + bτ.
    Оцінка дисперсії шуму σ² = RSS / (n − k), коваріація коефіцієнтів σ²(FᵀF)⁻¹.
    """
    kind, degree = MODELS[model]
    n = len(y)
    f = design(np.arange(n) / n, degree)
    target = np.log(y) if kind == "exp" else y
    coeffs = solve(f, target)
    residual = target - f @ coeffs
    dof = n - len(coeffs)
    sigma2 = residual @ residual / dof
    inverse = np.linalg.inv(f.T @ f)
    return {
        "model": model,
        "kind": kind,
        "n": n,
        "coeffs": coeffs,
        "fitted": back(kind, f @ coeffs),
        "sigma2": sigma2,
        "dof": dof,
        "cov": sigma2 * inverse,
        "inverse": inverse,
        "cond": np.linalg.cond(f.T @ f),
    }


def back(kind, values):
    return np.exp(values) if kind == "exp" else values


def predict(state, horizon, level=LEVEL):
    """Прогноз на horizon відліків за межі вибірки з прогнозним інтервалом.

    ŷ = f·C, s² = σ²(1 + f(FᵀF)⁻¹fᵀ), межі ŷ ± t(1−α/2, n−k)·s.
    """
    n = state["n"]
    f = design(np.arange(n, n + horizon) / n, len(state["coeffs"]) - 1)
    mean = f @ state["coeffs"]
    spread = np.sqrt(state["sigma2"] * (1.0 + np.einsum("ij,jk,ik->i", f, state["inverse"], f)))
    q = stats.t.ppf(0.5 + level / 2, state["dof"])
    kind = state["kind"]
    return back(kind, mean), back(kind, mean - q * spread), back(kind, mean + q * spread)


def significance(state):
    """t-статистики коефіцієнтів t = c / se(c) і їх p-значення."""
    se = np.sqrt(np.diag(state["cov"]))
    t = state["coeffs"] / se
    return se, t, 2 * stats.t.sf(np.abs(t), state["dof"])


def residuals(state, y):
    """Залишки в тій шкалі, де модель лінійна: y − ŷ або ln y − ln ŷ."""
    if state["kind"] == "exp":
        return np.log(y) - np.log(state["fitted"])
    return y - state["fitted"]


def fit_ar(y, model):
    """Тренд за МНК плюс AR(1) для залишків: eₜ = φ·eₜ₋₁ + uₜ.

    φ теж оцінюється МНК: φ = Σeₜeₜ₋₁ / Σeₜ₋₁². Прогноз тренду коригується
    на φʰ·eₙ – поправка згасає з горизонтом, і модель повертається до тренду.
    """
    state = fit(y, model)
    e = residuals(state, y)
    phi = (e[1:] @ e[:-1]) / (e[:-1] @ e[:-1])
    u = e[1:] - phi * e[:-1]
    return state | {"phi": phi, "last": e[-1], "e": e, "u": u, "sigma2_u": u @ u / (len(u) - 1)}


def predict_ar(state, horizon, level=LEVEL):
    """ŷ(n+h) = тренд(n+h) + φʰ·eₙ.

    Дисперсія помилки: σᵤ²·Σφ²ʲ (j = 0…h−1) від AR(1) плюс σ²·f(FᵀF)⁻¹fᵀ
    від невизначеності коефіцієнтів тренду.
    """
    n = state["n"]
    f = design(np.arange(n, n + horizon) / n, len(state["coeffs"]) - 1)
    h = np.arange(1, horizon + 1)
    mean = f @ state["coeffs"] + state["phi"] ** h * state["last"]
    ar = state["sigma2_u"] * np.cumsum(state["phi"] ** (2 * (h - 1)))
    trend = state["sigma2"] * np.einsum("ij,jk,ik->i", f, state["inverse"], f)
    spread = np.sqrt(ar + trend)
    q = stats.t.ppf(0.5 + level / 2, state["dof"])
    kind = state["kind"]
    return back(kind, mean), back(kind, mean - q * spread), back(kind, mean + q * spread)


def forecaster(model, ar=False):
    """Функція прогнозу для бектесту: навчання на train і екстраполяція на h кроків."""
    if ar:
        return lambda train, h: predict_ar(fit_ar(train, model), h)[0]
    return lambda train, h: predict(fit(train, model), h)[0]


def formula(state):
    """Рівняння тренду в змінній τ = t / n для звіту."""
    c = state["coeffs"]
    if state["kind"] == "exp":
        return f"y = {np.exp(c[0]):.4f}·exp({c[1]:.4f}·τ)"
    terms = [f"{c[0]:.4f}"]
    for power, value in enumerate(c[1:], start=1):
        sign = "−" if value < 0 else "+"
        tail = "τ" if power == 1 else f"τ^{power}"
        terms.append(f"{sign} {abs(value):.4f}·{tail}")
    return "y = " + " ".join(terms)
