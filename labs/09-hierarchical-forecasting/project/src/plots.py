"""Спільне оформлення графіків: палітра, теплові карти, збереження."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
REAL = "#0b0b0b"
BAD = "#d03b3b"

DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1f1f"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential", ["#f0f6fd", "#86b6ef", "#2a78d6", "#0d366b"]
)

DPI = 140


def setup():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, out, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)


def heatmap(ax, matrix, rows, cols, cmap=DIVERGING, vmin=-1.0, vmax=1.0, fmt="{:.2f}"):
    """Теплова карта з підписаними значеннями; колір тексту обирається за світлістю клітинки."""
    matrix = np.asarray(matrix, dtype=float)
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            r, g, b, _ = image.cmap(image.norm(matrix[i, j]))
            light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55
            ax.text(
                j,
                i,
                fmt.format(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color=INK if light else "white",
            )
    return image
