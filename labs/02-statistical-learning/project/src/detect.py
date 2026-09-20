"""Виявлення аномальних вимірів: базовий детектор і власний адаптивний."""

import numpy as np
from scipy import stats

MAD_TO_SIGMA = 1.4826
ALPHA = 0.05
WINDOWS = (3, 5, 9, 15, 25, 41)


def residual(data, window):
    """Залишок після ковзної медіани непарної ширини window."""
    pad = window // 2
    view = np.lib.stride_tricks.sliding_window_view(np.pad(data, pad, mode="edge"), window)
    return data - np.median(view, axis=1)


def clean(data, mask):
    """Замінити позначені відліки лінійною інтерполяцією сусідів."""
    out = data.copy()
    if mask.any() and not mask.all():
        i = np.arange(len(data), dtype=float)
        out[mask] = np.interp(i[mask], i[~mask], out[~mask])
    return out


def fixed(data, window=5, k=3.0):
    """Базовий детектор: вікно й поріг k·СКВ задані наперед."""
    r = residual(data, window)
    return np.abs(r) > k * r.std()


def mad_sigma(r):
    """Оцінка СКВ через медіанне абсолютне відхилення: 1.4826 · median(|r − median r|)."""
    return MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r))))


def threshold(n, alpha=ALPHA):
    """Поріг з обсягу вибірки за поправкою Бонферроні: k = Φ⁻¹(1 − α/2n)."""
    return float(stats.norm.ppf(1.0 - alpha / (2 * n)))


def lag1(r):
    """Автокореляція залишку на лагу 1."""
    c = r - r.mean()
    return float(c[:-1] @ c[1:] / (c @ c))


def best_window(data, windows=WINDOWS):
    """Ширина вікна, за якої |ACF(1)| залишку найменша."""
    return min(windows, key=lambda w: abs(lag1(residual(data, w))))


def adaptive(data, alpha=ALPHA, rounds=5, windows=WINDOWS):
    """Власний детектор: вікно з ACF, поріг з обсягу вибірки, σ через MAD."""
    window = best_window(data, windows)
    k = threshold(len(data), alpha)
    mask = np.zeros(len(data), dtype=bool)
    for _ in range(rounds):
        r = residual(clean(data, mask), window)
        found = np.abs(r) > k * mad_sigma(r)
        if np.array_equal(found, mask):
            break
        mask = found
    return mask, window, k
