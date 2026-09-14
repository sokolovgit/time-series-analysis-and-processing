"""Етап 4: статистичні (числові) характеристики вибірок."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Stats:
    """Числові характеристики вибірки."""

    mean: float
    variance: float
    std: float

    def __str__(self) -> str:
        return f"M = {self.mean:.4f}, D = {self.variance:.4f}, СКВ = {self.std:.4f}"


def describe(sample: Series) -> Stats:
    """Математичне сподівання, дисперсія та середньоквадратичне відхилення."""
    variance = float(np.var(sample))
    return Stats(mean=float(np.mean(sample)), variance=variance, std=float(np.sqrt(variance)))


def mnk_fit(series: Series, degree: int = 2) -> tuple[Series, Series]:
    """Згладжування за МНК: C = (F^T F)^-1 F^T Y.

    Повертає згладжений ряд і коефіцієнти полінома степеня degree.
    """
    t = np.arange(len(series), dtype=float)
    f = np.vander(t, degree + 1, increasing=True)
    coeffs = np.linalg.solve(f.T @ f, f.T @ series)
    return f @ coeffs, coeffs


def detrended(sample: Series, trend: Series) -> Series:
    """Залишок після зняття тренду - саме він несе характеристики похибки."""
    return sample - trend


def r_squared(actual: Series, fitted: Series) -> float:
    """Достовірність апроксимації R**2 (coefficient of determination)."""
    residual = float(np.sum((actual - fitted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - residual / total
