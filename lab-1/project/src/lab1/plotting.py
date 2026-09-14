"""Етап 6: візуалізація результатів розрахунків."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]


def plot_series(series: dict[str, Series], title: str, file: Path) -> Path:
    """Побудувати графік одного або кількох рядів і зберегти у файл."""
    raise NotImplementedError


def plot_histogram(sample: Series, title: str, file: Path, *, bins: int = 20) -> Path:
    """Побудувати гістограму закону розподілу вибірки і зберегти у файл."""
    raise NotImplementedError
