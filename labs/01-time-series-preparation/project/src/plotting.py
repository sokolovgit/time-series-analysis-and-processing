"""Візуалізація результатів розрахунків."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # графіки зберігаються у файли, вікна не відкриваються

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]


def plot_series(
    series: dict[str, Series],
    title: str,
    file: Path,
    *,
    x: NDArray | None = None,
    xlabel: str = "час t, номер відліку",
    ylabel: str = "значення S(t)",
) -> Path:
    """Побудувати графік одного або кількох рядів і зберегти у файл.

    Якщо передано x (дати або роки), він використовується як шкала абсцис.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in series.items():
        if x is None:
            ax.plot(values, label=label, linewidth=1.0)
        else:
            ax.plot(x, values, label=label, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    if x is not None and np.issubdtype(np.asarray(x).dtype, np.datetime64):
        fig.autofmt_xdate()
    return _save(fig, file)


def plot_histogram(
    sample: Series,
    title: str,
    file: Path,
    *,
    bins: int = 20,
    xlabel: str = "значення випадкової величини",
    ylabel: str = "частота",
) -> Path:
    """Побудувати гістограму закону розподілу вибірки і зберегти у файл."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample, bins=bins, facecolor="steelblue", alpha=0.7, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _save(fig, file)


def plot_histogram_pair(
    samples: dict[str, Series],
    title: str,
    file: Path,
    *,
    bins: int = 20,
    xlabel: str = "значення",
) -> Path:
    """Дві гістограми поруч: вибірка та виділена з неї похибка."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (label, values) in zip(axes, samples.items(), strict=True):
        ax.hist(values, bins=bins, facecolor="steelblue", alpha=0.7, edgecolor="white")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("частота")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, file)


def _save(fig: plt.Figure, file: Path) -> Path:
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file, dpi=120)
    plt.close(fig)
    return file
