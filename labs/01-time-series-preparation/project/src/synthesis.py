"""Завдання III рівня, п.5: синтез моделі, подібної реальним даним."""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

import stats

Series = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Synthetic:
    """Результат синтезу моделі за реальними даними."""

    series: Series  # синтезована вибірка
    trend: Series  # тренд реальних даних, відновлений за МНК
    residual: Series  # залишок реальних даних після зняття тренду
    coeffs: Series  # коефіцієнти полінома тренду


def like(real: Series, rng: Generator, degree: int = 2) -> Synthetic:
    """Синтезувати вибірку з тим самим трендом і тими ж характеристиками похибки."""
    trend, coeffs = stats.mnk_fit(real, degree)
    residual = stats.detrended(real, trend)
    noise = rng.normal(float(np.mean(residual)), float(np.std(residual)), len(real))
    return Synthetic(series=trend + noise, trend=trend, residual=residual, coeffs=coeffs)
