"""Етап 3: адитивна модель статистичної вибірки."""

import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]


def additive(trend: Series, noise: Series) -> Series:
    """Адитивна модель: невипадкова складова (тренд) + стохастична складова (похибка)."""
    raise NotImplementedError


def with_anomalies(
    sample: Series,
    rng: np.random.Generator,
    fraction: float,
    magnitude: float,
) -> Series:
    """Внести аномальні викиди у задану частку відліків вибірки."""
    raise NotImplementedError
