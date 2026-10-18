"""Подолання розходження фільтра: навчена обмежена пам'ять і власний адаптивний алгоритм."""

import numpy as np
from scipy import optimize, special, stats

import detect
import filters

DRIFT = 0.5
ALPHA = 0.05
MAD_EFFICIENCY = 0.3675


def siegmund_arl(h, drift=DRIFT):
    """Середня довжина серії до хибної тривоги однобічного CUSUM за наближенням Зігмунда.

    ARL₀ = (e^{2δb} − 2δb − 1) / 2δ²,  b = h + 1.166.
    """
    b = h + 1.166
    return (np.exp(2 * drift * b) - 2 * drift * b - 1) / (2 * drift**2)


def cusum_threshold(n, alpha=ALPHA, drift=DRIFT):
    """Поріг h, за якого на вибірці обсягу n очікується α хибних тривог із двох боків.

    ARL₀(h) = 2n / α.
    """
    target = 2 * n / alpha
    return float(optimize.brentq(lambda h: siegmund_arl(h, drift) - target, 0.0, 100.0))


def resolution(data):
    """Крок квантування вимірів – найменша ненульова зміна сусідніх відліків."""
    steps = np.abs(np.diff(data))
    steps = steps[steps > 0]
    return float(steps.min()) if len(steps) else 0.0


def noise_sigma(diffs, order, step=0.0):
    """СКВ шуму з різниць порядку order: Var Δᵐε = C(2m, m)·σ², σ – через MAD.

    Знизу обмежена шумом квантування q/√12: якщо ряд довго стоїть на місці, MAD = 0,
    і без цієї межі будь-яка зміна давала б нескінченну нормовану нев'язку.
    """
    sigma = detect.mad_sigma(diffs) / np.sqrt(special.comb(2 * order, order))
    return max(sigma, step / np.sqrt(12))


def effective_dof(order):
    """Ступені свободи σ̂ на один використаний відлік різниць.

    Різниці Δᵐε корельовані: ρⱼ = (−1)ʲ C(2m, m − j) / C(2m, m), тому дисперсія оцінки
    дисперсії більша в 1 + 2Σρⱼ² разів, а MAD має асимптотичну ефективність 0.3675.
    """
    rho = [
        special.comb(2 * order, order - j) / special.comb(2 * order, order)
        for j in range(1, order + 1)
    ]
    return MAD_EFFICIENCY / (1 + 2 * sum(r * r for r in rho))


def to_normal(t, df):
    """Статистика з оціненою σ̂ має розподіл Стьюдента – переводимо її в N(0, 1).

    z = sign(t)·Φ⁻¹(1 − S_t(|t|; df)), де S_t – функція виживання розподілу Стьюдента.
    """
    return float(np.sign(t) * stats.norm.isf(stats.t.sf(abs(t), df)))


def min_memory(order):
    """Найкоротша пам'ять, за якої дисперсія прогнозу не більша за дисперсію виміру: ρ(n) ≤ 1."""
    n = order
    while filters.prediction_factor(n, order) > 1.0:
        n += 1
    return n


def run(data, order, alpha=ALPHA, drift=DRIFT):
    """Власний алгоритм: зростаюча пам'ять + CUSUM нормованих нев'язок + перезапуск.

    z_k = r_k / σ̂·√(1 + ρ(n)),  g⁺ = max(0, g⁺ + z − δ),  g⁻ = max(0, g⁻ − z − δ).
    Коли g > h, початок розладки τ – останній крок, де g дорівнювала нулю. Пам'ять
    скорочується до k − τ, а стан перераховується МНК саме за цими вимірами.
    """
    n_total = len(data)
    h = cusum_threshold(n_total, alpha, drift)
    diffs = np.diff(data, order)
    step = resolution(data)
    floor = min_memory(order)
    dof = effective_dof(order)
    estimate = np.empty(n_total)
    innovation = np.full(n_total, np.nan)
    memory = np.empty(n_total, dtype=int)
    alarms = []

    x = v = a = 0.0
    n = 0
    up = down = 0.0
    up_start = down_start = 0
    for k in range(n_total):
        n += 1
        x_pred, v_pred = filters.predict(x, v, a)
        r = data[k] - x_pred
        x, v, a = filters.correct(x_pred, v_pred, a, r, filters.gains(n, order))

        # σ̂ – лише з уже отриманих вимірів, тест вмикається, коли їх достатньо
        seen = k + 1 - order
        if k >= order:
            innovation[k] = r
        if n > order and seen > 2 * order:
            window = diffs[max(0, k - n - order) : seen]
            t = r / (
                noise_sigma(window, order, step)
                * np.sqrt(1 + filters.prediction_factor(n - 1, order))
            )
            z = to_normal(t, max(len(window) * dof, 1.0))
            up, down = max(0.0, up + z - drift), max(0.0, down - z - drift)
            up_start = k if up == 0.0 else up_start
            down_start = k if down == 0.0 else down_start
            if max(up, down) > h:
                start = up_start if up > h else down_start
                n = min(max(k - start, floor), k + 1)
                x, v, a = filters.fit_state(data[k - n + 1 : k + 1], order)
                alarms.append((k, start))
                up = down = 0.0
                up_start = down_start = k
        estimate[k] = x
        memory[k] = n
    return estimate, innovation, memory, alarms, h
