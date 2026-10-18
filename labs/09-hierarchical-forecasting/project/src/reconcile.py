"""Методи ієрархічного прогнозування: кожен дає прогноз нижнього рівня b, повний прогноз ỹ = S·b."""

import numpy as np


def projection(S, W):
    """G = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ – узгодження за узагальненим МНК."""
    Wi = np.linalg.inv(W)
    return np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)


def shrink_cov(E):
    """Коваріація залишків зі стисканням до діагоналі, λ за Шефером–Штріммером.

    Вибіркова коваріація 75 рядів за ~30 років вироджена, тому без стискання W⁻¹ не існує.
    """
    T = len(E)
    cov = E.T @ E / T
    sd = np.sqrt(np.diag(cov))
    X = E / sd
    R = cov / np.outer(sd, sd)
    v = (X**2).T @ (X**2) - (X.T @ X) ** 2 / T
    v *= 1 / (T * (T - 1))
    np.fill_diagonal(v, 0.0)
    d = R**2
    np.fill_diagonal(d, 0.0)
    lam = float(np.clip(v.sum() / d.sum(), 0.0, 1.0))
    return lam * np.diag(np.diag(cov)) + (1 - lam) * cov, lam


def bottom_up(h, F, E, Y):
    return F[h["bottom"]]


def ols(h, F, E, Y):
    S = h["S"]
    return projection(S, np.eye(len(S))) @ F


def wls(h, F, E, Y):
    S = h["S"]
    return projection(S, np.diag(S.sum(axis=1))) @ F


def mint(h, F, E, Y):
    W, _ = shrink_cov(E)
    return projection(h["S"], W) @ F


def top_down_ahp(h, F, E, Y):
    """p_j = середнє за роками y_j,t / y_t."""
    p = (Y[:, h["bottom"]] / Y[:, [0]]).mean(axis=0)
    return np.outer(p, F[0])


def top_down_pha(h, F, E, Y):
    """p_j = ȳ_j / ȳ."""
    p = Y[:, h["bottom"]].mean(axis=0) / Y[:, 0].mean()
    return np.outer(p, F[0])


def proportions(h, F, anchor_level):
    """Частки прогнозу: від листа до вузла рівня anchor_level, добуток ŷ_вузла / Σ ŷ_сиблінгів."""
    parent, level = h["parent"], h["level"]
    out = []
    for j in h["bottom"]:
        share, node = np.ones(F.shape[1]), j
        while level[node] > anchor_level:
            siblings = np.flatnonzero(parent == parent[node])
            share = share * F[node] / F[siblings].sum(axis=0)
            node = parent[node]
        out.append(F[node] * share)
    return np.array(out)


def top_down_fp(h, F, E, Y):
    return proportions(h, F, 0)


def middle_out(h, F, E, Y):
    """Прогноз на рівні областей: угору – сумою, униз – частками прогнозу міська/сільська."""
    return proportions(h, F, 1)


METHODS = {
    "BU": bottom_up,
    "TD-AHP": top_down_ahp,
    "TD-PHA": top_down_pha,
    "TD-FP": top_down_fp,
    "MO": middle_out,
    "OLS": ols,
    "WLS": wls,
    "MinT": mint,
}
LINEAR = ("BU", "TD-AHP", "TD-PHA", "OLS", "WLS", "MinT")


def methods(depth):
    """MO має сенс лише за трьох рівнів: у дворівневій ієрархії він збігається з BU."""
    return [m for m in METHODS if depth == 3 or m != "MO"]


def run(h, F, E, Y):
    """Узгоджені прогнози всіма методами; base – незалежні базові прогнози без узгодження."""
    out = {"base": F}
    for name in methods(h["depth"]):
        out[name] = h["S"] @ METHODS[name](h, F, E, Y)
    return out
