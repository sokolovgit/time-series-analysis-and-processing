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
        return (
            f"математичне сподівання = {self.mean:.4f}, "
            f"дисперсія = {self.variance:.4f}, "
            f"СКВ = {self.std:.4f}"
        )


def describe(sample: Series) -> Stats:
    """Порахувати математичне сподівання, дисперсію та середньоквадратичне відхилення."""
    raise NotImplementedError


def detrended(sample: Series, trend: Series) -> Series:
    """Відняти тренд, щоб оцінити характеристики самої лише похибки."""
    raise NotImplementedError
