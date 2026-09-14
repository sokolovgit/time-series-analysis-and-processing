"""Етап 5 та завдання III рівня: реальні дані з файлу і з веб-сайту."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Series = NDArray[np.float64]


def read_excel_column(file: Path, column: str) -> Series:
    """Прочитати один показник із .xls/.xlsx як числовий ряд."""
    raise NotImplementedError


def fetch_page(url: str, *, timeout: float = 10.0) -> str:
    """Завантажити HTML сторінки для подальшого парсингу."""
    raise NotImplementedError


def parse_table(html: str) -> pd.DataFrame:
    """Витягти таблицю зі сторінки у DataFrame (bs4 / pandas.read_html)."""
    raise NotImplementedError


def save(frame: pd.DataFrame, file: Path) -> Path:
    """Зберегти результат парсингу у файл. Формат — за розширенням."""
    raise NotImplementedError
