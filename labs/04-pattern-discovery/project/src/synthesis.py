"""Генерування модельних Time Series з виявленими закономірностями та перевірка їх адекватності."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

from analysis import YEAR, harmonic_design

HARMONICS = 3
SIGMA_HARMONICS = 2
VAR_MAX_LAGS = 5
BURN = 200
MIN_AR_LENGTH = 10
SEASON_WINDOW = 15
BAND = (5, 95)


def fit_ar1(x):
    """AR(1) навколо середнього: x(t) − μ = φ·(x(t−1) − μ) + σ_e·e(t)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    if len(x) < MIN_AR_LENGTH or x.std() == 0:
        return mu, 0.0, x.std()
    d = x - mu
    phi = np.clip(np.dot(d[1:], d[:-1]) / np.dot(d[:-1], d[:-1]), -0.99, 0.99)
    sigma = np.std(d[1:] - phi * d[:-1])
    return mu, phi, sigma


def simulate_ar1(mu, phi, sigma, n, rng, start=None):
    """Рекурсія AR(1) від попереднього стану start (за замовчуванням – від середнього μ)."""
    x = np.empty(n)
    prev = mu if start is None else start
    for i in range(n):
        prev = mu + phi * (prev - mu) + sigma * rng.normal()
        x[i] = prev
    return x


# --- Група 1: курс валют як послідовність режимів --------------------------------------


def ar1_residuals(x, mu, phi):
    """e(t) = x(t) − μ − φ·(x(t−1) − μ), t = 1..n−1; для коротких ділянок (φ = 0) – x(t) − μ."""
    d = np.asarray(x, dtype=float) - mu
    return d[1:] - phi * d[:-1] if phi else d


def fit_rates(frame, segments):
    """Кусково-стаціонарна модель на виявлених режимах. На ділянці k:

    p(t) = buy − nbu: p(t) − μ_p = φ_p·(p(t−1) − μ_p) + e_p(t) – премія ринку до офіційного курсу;
    s(t) = sell − buy: s(t) − μ_s = φ_s·(s(t−1) − μ_s) + e_s(t) – спред обмінника;
    nbu(t) – зовнішній вхід: фіксований рівень або, якщо НБУ на ділянці рухався, блукання
    з емпіричними приростами. Інновації (e_p, e_s, Δnbu) беруться бутстрепом разом,
    з одного дня ділянки – так зберігаються важкі хвости і зв'язок між рядами.
    """
    nbu_steps = frame["nbu"].diff().fillna(0.0).to_numpy()
    model = []
    for label, start, end in segments:
        part = frame.iloc[start:end]
        premium = (part["buy"] - part["nbu"]).to_numpy()
        spread = (part["sell"] - part["buy"]).to_numpy()
        mu_p, phi_p, _ = fit_ar1(premium)
        mu_s, phi_s, _ = fit_ar1(spread)
        e_p = ar1_residuals(premium, mu_p, phi_p)
        e_s = ar1_residuals(spread, mu_s, phi_s)
        steps = nbu_steps[start + 1 : end] if start + 1 < end else np.zeros(1)
        floating = np.abs(steps).max() > 0 and end - start >= MIN_AR_LENGTH
        m = min(len(e_p), len(e_s))
        model.append(
            {
                "label": label,
                "length": end - start,
                "nbu": part["nbu"].to_numpy(),
                "floating": floating,
                "premium": (mu_p, phi_p),
                "spread": (mu_s, phi_s),
                "innovations": np.column_stack(
                    [e_p[-m:], e_s[-m:], steps[-m:] if floating else np.zeros(m)]
                ),
            }
        )
    return model


def synth_rates(model, rng):
    """Зміна режиму – це стрибок ринку до нової рівноваги: премія кожної ділянки стартує з μ_p,
    спред – зі значення наприкінці попередньої ділянки."""
    rows = []
    s_prev = None
    for seg in model:
        n = seg["length"]
        (mu_p, phi_p), (mu_s, phi_s) = seg["premium"], seg["spread"]
        draws = seg["innovations"][rng.integers(0, len(seg["innovations"]), n)]
        if seg["floating"]:
            nbu = seg["nbu"][0] + np.r_[0.0, np.cumsum(draws[1:, 2])]
        else:
            nbu = np.full(n, seg["nbu"].mean())
        p_prev = mu_p
        s_prev = mu_s if s_prev is None else s_prev
        out = np.empty((n, 3))
        for i in range(n):
            p_prev = mu_p + phi_p * (p_prev - mu_p) + draws[i, 0]
            s_prev = max(mu_s + phi_s * (s_prev - mu_s) + draws[i, 1], 0.0)
            out[i] = nbu[i] + p_prev, nbu[i] + p_prev + s_prev, nbu[i]
        rows.append(out)
    return pd.DataFrame(np.vstack(rows), columns=["buy", "sell", "nbu"])


# --- Група 2: погода як сезонний процес з просторово-часовою залежністю ------------------


def day_of_year(n, offset=0):
    return ((np.arange(n) + offset) % YEAR).astype(int)


def seasonal_pools(n, offset=0, window=SEASON_WINDOW):
    """Для кожного дня року – індекси відліків, чий день року відстоїть не далі ніж на window."""
    doy = (np.arange(n) + offset) % YEAR
    pools = []
    for day in range(int(np.ceil(YEAR))):
        gap = np.abs(doy - day)
        pools.append(np.flatnonzero(np.minimum(gap, YEAR - gap) <= window))
    return pools


def normal_scores(z, pools):
    """Сезонне перетворення до нормальних рангів: s = Φ⁻¹(F̂_d(z)), F̂_d – емпірична функція
    розподілу аномалій у вікні навколо дня року d."""
    scores = np.empty_like(z)
    doy = day_of_year(len(z))
    for day, pool in enumerate(pools):
        rows = np.flatnonzero(doy == day)
        for c in range(z.shape[1]):
            ref = np.sort(z[pool, c])
            scores[rows, c] = (np.searchsorted(ref, z[rows, c]) + 0.5) / len(ref)
    return stats.norm.ppf(scores)


def seasonal_quantiles(scores, sorted_pools):
    """Зворотне перетворення z = F̂_d⁻¹(Φ(s)) лінійною інтерполяцією емпіричних квантилів.

    Ранг r відповідає u = (r + ½)/L, тому позиція в упорядкованій вибірці – u·L − ½.
    """
    z = np.empty_like(scores)
    doy = day_of_year(len(scores))
    columns = np.arange(scores.shape[1])
    for day, ref in enumerate(sorted_pools):
        rows = np.flatnonzero(doy == day)
        position = np.clip(stats.norm.cdf(scores[rows]) * len(ref) - 0.5, 0, len(ref) - 1)
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, len(ref) - 1)
        frac = position - low
        z[rows] = ref[low, columns] * (1 - frac) + ref[high, columns] * frac
    return z


def fit_var(data):
    var = VAR(data).fit(maxlags=VAR_MAX_LAGS, ic="bic")
    return {"coefs": var.coefs, "sigma_u": var.sigma_u, "residuals": var.resid, "order": var.k_ar}


def fit_climate(frame, k=HARMONICS, k_sigma=SIGMA_HARMONICS):
    """Для кожного міста Y(t) = a + b·t + Σ гармонік + σ(t)·z(t), де σ(t) – сезонна СКВ.

    σ²(t) оцінюється регресією r²(t) на гармоніки – це не залежить від форми розподілу r.
    Стандартизовані аномалії z(t) усіх міст разом описуються VAR(p):
    z(t) = Σ_{i=1}^{p} A_i·z(t − i) + u(t), u ~ (0, Σ_u).
    Для копули окремий VAR будується на нормальних рангах s(t) замість z(t).
    """
    t = np.arange(len(frame))
    x_mean = harmonic_design(t, k)
    x_sigma = harmonic_design(t, k_sigma, trend=False)
    beta, *_ = np.linalg.lstsq(x_mean, frame.values, rcond=None)
    resid = frame.values - x_mean @ beta
    gamma, *_ = np.linalg.lstsq(x_sigma, resid**2, rcond=None)
    z = resid / np.sqrt(x_sigma @ gamma)
    pools = seasonal_pools(len(z))
    scores = normal_scores(z, pools)
    return {
        "columns": list(frame.columns),
        "index": frame.index,
        "beta": beta,
        "gamma": gamma,
        "k": k,
        "k_sigma": k_sigma,
        "z": z,
        "scores": scores,
        "sorted_pools": [np.sort(z[pool], axis=0) for pool in pools],
        "var_z": fit_var(z),
        "var_scores": fit_var(scores),
    }


def mean_curve(model, n):
    return harmonic_design(np.arange(n), model["k"]) @ model["beta"]


def sigma_curve(model, n):
    x = harmonic_design(np.arange(n), model["k_sigma"], trend=False)
    return np.sqrt(x @ model["gamma"])


def simulate_var(var, n, rng, innovations="gauss"):
    """Генерування VAR(p). Інновації – нормальні з Σ_u або сезонний бутстреп залишків моделі.

    Бутстреп бере вектор u(t) цілком, тому зберігає просторову кореляцію між містами,
    а вибір лише з того самого сезону – ще й різну взимку і влітку форму розподілу.
    """
    coefs, residuals = var["coefs"], var["residuals"]
    p, m, _ = coefs.shape
    total = n + BURN
    if innovations == "gauss":
        u = rng.multivariate_normal(np.zeros(m), var["sigma_u"], total)
    else:
        pools = seasonal_pools(len(residuals), p)
        days = day_of_year(total, -BURN)
        u = residuals[[rng.choice(pools[d]) for d in days]]
    z = np.zeros((total, m))
    for t in range(p, total):
        z[t] = u[t] + sum(coefs[i] @ z[t - 1 - i] for i in range(p))
    return z[BURN:]


def synth_climate(model, rng, method="bootstrap"):
    """method: gauss – VAR з нормальними інноваціями, bootstrap – VAR із сезонним бутстрепом
    залишків, copula – копула: VAR на нормальних рангах із бутстрепом залишків, після чого
    ранги переводяться в аномалії через сезонні емпіричні квантилі."""
    n = len(model["index"])
    if method == "copula":
        scores = simulate_var(model["var_scores"], n, rng, "bootstrap")
        z = seasonal_quantiles(scores, model["sorted_pools"])
    else:
        z = simulate_var(model["var_z"], n, rng, method)
    values = mean_curve(model, n) + sigma_curve(model, n) * z
    return pd.DataFrame(values, index=model["index"], columns=model["columns"])


# --- Адекватність -------------------------------------------------------------------------


def monte_carlo(metrics, real, generate, runs, rng):
    """Порівняння показників реального ряду з розподілом тих самих показників на runs модельних.

    Модель адекватна за показником, якщо реальне значення лежить у центральному 90 % інтервалі.
    """
    reference = metrics(real)
    samples = pd.DataFrame([metrics(generate(rng)) for _ in range(runs)])
    low, high = np.percentile(samples.values, BAND, axis=0)
    report = pd.DataFrame(
        {
            "реальний": pd.Series(reference),
            "модель m": samples.mean(),
            "P5": low,
            "P95": high,
        }
    )
    real = report["реальний"]
    report["в межах"] = (real >= report["P5"]) & (real <= report["P95"])
    return report, samples
