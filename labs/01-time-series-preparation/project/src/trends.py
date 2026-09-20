"""Етап 2: моделі зміни досліджуваного процесу (ідеальний тренд)."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]


def _time(size: int) -> Series:
    return np.arange(size, dtype=float)


def constant(size: int, level: float) -> Series:
    """Постійна величина: S(t) = level."""
    return np.full(size, level, dtype=float)


def linear(size: int, intercept: float, slope: float) -> Series:
    """Лінійний тренд: S(t) = intercept + slope * t."""
    return intercept + slope * _time(size)


def quadratic(size: int, a: float, b: float, c: float) -> Series:
    """Квадратичний тренд: S(t) = a * t**2 + b * t + c."""
    t = _time(size)
    return a * t**2 + b * t + c


def cubic(size: int, a: float, b: float, c: float, d: float) -> Series:
    """Кубічний тренд: S(t) = a * t**3 + b * t**2 + c * t + d."""
    t = _time(size)
    return a * t**3 + b * t**2 + c * t + d


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

# Ступінь полінома, яким тренд відновлюється за МНК.
DEGREE: dict[str, int] = {"constant": 0, "linear": 1, "quadratic": 2, "cubic": 3}


def build(law: str, size: int, **params: float) -> Series:
    """Побудувати ідеальний тренд за назвою закону зміни."""
    if law not in _REGISTRY:
        raise ValueError(f"невідомий закон зміни тренду: {law}")
    return _REGISTRY[law](size, **{**DEFAULT_PARAMS[law], **params})
