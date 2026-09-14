"""Етап 1: моделі генерації випадкової величини за заданими законами розподілу."""

from collections.abc import Callable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

Sample = NDArray[np.float64]


def uniform(rng: Generator, size: int, low: float, high: float) -> Sample:
    """Рівномірний закон розподілу на відрізку [low, high]."""
    raise NotImplementedError


def normal(rng: Generator, size: int, mean: float, std: float) -> Sample:
    """Нормальний закон розподілу з параметрами (mean, std)."""
    raise NotImplementedError


def exponential(rng: Generator, size: int, scale: float) -> Sample:
    """Експоненційний закон розподілу з масштабом scale."""
    raise NotImplementedError


def chi_square(rng: Generator, size: int, df: float) -> Sample:
    """Розподіл хі-квадрат з df ступенями свободи."""
    raise NotImplementedError


_REGISTRY: dict[str, Callable[..., Sample]] = {
    "uniform": uniform,
    "normal": normal,
    "exponential": exponential,
    "chi_square": chi_square,
}

# Параметри законів обираються самостійно (п. 3 завдання).
DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "uniform": {"low": -10.0, "high": 10.0},
    "normal": {"mean": 0.0, "std": 5.0},
    "exponential": {"scale": 5.0},
    "chi_square": {"df": 3.0},
}


def generate(rng: Generator, law: str, size: int, **params: float) -> Sample:
    """Згенерувати похибку за назвою закону розподілу."""
    if law not in _REGISTRY:
        raise ValueError(f"невідомий закон розподілу: {law}")
    return _REGISTRY[law](rng, size, **{**DEFAULT_PARAMS[law], **params})
