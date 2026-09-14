"""Етап 2: моделі зміни досліджуваного процесу (ідеальний тренд)."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]


def constant(size: int, level: float) -> Series:
    """Постійна величина: S(t) = level."""
    raise NotImplementedError


def linear(size: int, intercept: float, slope: float) -> Series:
    """Лінійний тренд: S(t) = intercept + slope * t."""
    raise NotImplementedError


def quadratic(size: int, a: float, b: float, c: float) -> Series:
    """Квадратичний тренд: S(t) = a * t**2 + b * t + c."""
    raise NotImplementedError


def cubic(size: int, a: float, b: float, c: float, d: float) -> Series:
    """Кубічний тренд: S(t) = a * t**3 + b * t**2 + c * t + d."""
    raise NotImplementedError


_REGISTRY: dict[str, Callable[..., Series]] = {
    "constant": constant,
    "linear": linear,
    "quadratic": quadratic,
    "cubic": cubic,
}

# Параметри законів зміни обираються самостійно (п. 3 завдання).
DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "constant": {"level": 10.0},
    "linear": {"intercept": 10.0, "slope": 0.05},
    "quadratic": {"a": 0.0001, "b": 0.01, "c": 10.0},
    "cubic": {"a": 1e-7, "b": 0.0001, "c": 0.01, "d": 10.0},
}


def build(law: str, size: int, **params: float) -> Series:
    """Побудувати ідеальний тренд за назвою закону зміни."""
    if law not in _REGISTRY:
        raise ValueError(f"невідомий закон зміни тренду: {law}")
    return _REGISTRY[law](size, **{**DEFAULT_PARAMS[law], **params})
