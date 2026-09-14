"""Етап 1: моделі генерації випадкової величини за заданими законами розподілу."""

from collections.abc import Callable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

Sample = NDArray[np.float64]


def uniform(rng: Generator, size: int, low: float, high: float) -> Sample:
    """Рівномірний закон розподілу на відрізку [low, high]. M = (a+b)/2, D = (b-a)**2/12."""
    return rng.uniform(low, high, size)


def normal(rng: Generator, size: int, mean: float, std: float) -> Sample:
    """Нормальний закон розподілу. M = mean, D = std**2."""
    return rng.normal(mean, std, size)


def exponential(rng: Generator, size: int, scale: float) -> Sample:
    """Експоненційний закон розподілу. M = scale, D = scale**2."""
    return rng.exponential(scale, size)


def chi_square(rng: Generator, size: int, df: float) -> Sample:
    """Розподіл хі-квадрат з df ступенями свободи. M = df, D = 2*df."""
    return rng.chisquare(df, size)


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


def theoretical(law: str, **params: float) -> tuple[float, float]:
    """Теоретичні математичне сподівання та дисперсія закону - потрібні для верифікації."""
    p = {**DEFAULT_PARAMS[law], **params}
    match law:
        case "uniform":
            return (p["low"] + p["high"]) / 2, (p["high"] - p["low"]) ** 2 / 12
        case "normal":
            return p["mean"], p["std"] ** 2
        case "exponential":
            return p["scale"], p["scale"] ** 2
        case "chi_square":
            return p["df"], 2 * p["df"]
        case _:
            raise ValueError(f"невідомий закон розподілу: {law}")


def generate(rng: Generator, law: str, size: int, **params: float) -> Sample:
    """Згенерувати похибку за назвою закону розподілу."""
    if law not in _REGISTRY:
        raise ValueError(f"невідомий закон розподілу: {law}")
    return _REGISTRY[law](rng, size, **{**DEFAULT_PARAMS[law], **params})
