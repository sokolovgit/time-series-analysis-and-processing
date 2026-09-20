"""Етап 3: адитивна модель статистичної вибірки."""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

Series = NDArray[np.float64]


def additive(trend: Series, noise: Series) -> Series:
    """Адитивна модель: невипадкова складова (тренд) + стохастична складова (похибка)."""
    return trend + noise


def with_anomalies(
    sample: Series,
    rng: Generator,
    fraction: float = 0.1,
    magnitude: float = 3.0,
) -> Series:
    """Внести аномальні виміри у задану частку відліків: похибка більша в magnitude разів."""
    result = sample.copy()
    count = int(len(sample) * fraction)
    indices = rng.choice(len(sample), size=count, replace=False)
    result[indices] += rng.normal(0.0, magnitude * float(np.std(sample)), count)
    return result
