"""Етап 5 та завдання III рівня: реальні дані з файлу і з веб-сайту."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from numpy.typing import NDArray

Series = NDArray[np.float64]

USER_AGENT = "Mozilla/5.0 (compatible; TimeSeriesLab/1.0)"
_YEAR = re.compile(r"^\d{4}$")


def read_rates(file: Path, *, date_column: str = "Дата") -> pd.DataFrame:
    """Прочитати файл реальних даних, перетворивши колонку дат у datetime."""
    frame = pd.read_excel(file)
    frame[date_column] = pd.to_datetime(frame[date_column], format="%d.%m.%Y")
    return frame


def fill_gaps(series: Series, *, missing: float = 0.0) -> tuple[Series, int]:
    """Заповнити пропуски лінійною інтерполяцією сусідніх відліків.

    В Oschadbank (USD).xls пропуски закодовані нулем: з 24.02.2022 банк певний час
    не котирував валюту. Без обробки такі нулі різко завищують дисперсію.
    Повертає очищений ряд і кількість заповнених відліків.
    """
    values = series.copy()
    gaps = values == missing
    count = int(gaps.sum())
    if count and not gaps.all():
        index = np.arange(len(values), dtype=float)
        values[gaps] = np.interp(index[gaps], index[~gaps], values[~gaps])
    return values, count


def fetch_page(url: str, *, timeout: float = 20.0) -> str:
    """Завантажити HTML сторінки для подальшого парсингу."""
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_emissions(html: str) -> pd.DataFrame:
    """Витягти таблицю річних викидів CO2 з worldometers.info.

    Колонки сторінки: Year, CO2 emissions (tons), 1 Year Change, Per Capita.
    Числа записані з роздільником тисяч (39,632,664,219).
    """
    table = BeautifulSoup(html, "lxml").find("table")
    if table is None:
        raise ValueError("на сторінці немає таблиці з викидами")

    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 4 or not _YEAR.match(cells[0]):
            continue
        rows.append(
            {
                "Рік": int(cells[0]),
                "Викиди": float(cells[1].replace(",", "")),
                "НаДушу": float(cells[3]),
            }
        )

    if not rows:
        raise ValueError("не вдалося розпізнати жодного рядка таблиці")
    return pd.DataFrame(rows).sort_values("Рік").reset_index(drop=True)


def save(frame: pd.DataFrame, file: Path) -> Path:
    """Зберегти результат парсингу у файл. Формат - за розширенням."""
    file.parent.mkdir(parents=True, exist_ok=True)
    match file.suffix:
        case ".csv":
            frame.to_csv(file, index=False)
        case ".json":
            frame.to_json(file, orient="records", date_format="iso", indent=2)
        case ".xlsx":
            frame.to_excel(file, index=False)
        case _:
            raise ValueError(f"непідтримуваний формат файлу: {file.suffix}")
    return file
