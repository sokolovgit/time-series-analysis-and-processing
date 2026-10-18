"""Вхідні дані: курс USD Ощадбанку (група 1) і погодний архів Open-Meteo по містах (група 2)."""

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2024-12-31"
HOURLY_START = START

CITIES = {
    "Ужгород": (48.62, 22.30),
    "Львів": (49.84, 24.03),
    "Івано-Франківськ": (48.92, 24.71),
    "Луцьк": (50.75, 25.34),
    "Вінниця": (49.23, 28.47),
    "Київ": (50.45, 30.52),
    "Чернігів": (51.49, 31.29),
    "Полтава": (49.59, 34.55),
    "Харків": (49.99, 36.23),
    "Дніпро": (48.46, 35.05),
    "Запоріжжя": (47.84, 35.14),
    "Одеса": (46.48, 30.72),
}

VARIABLES = {
    "temperature_2m_mean": "temp",
    "relative_humidity_2m_mean": "humidity",
    "pressure_msl_mean": "pressure",
    "wind_speed_10m_mean": "wind",
    "precipitation_sum": "precip",
    "shortwave_radiation_sum": "radiation",
}

RATE_COLUMNS = {"Купівля": "buy", "Продаж": "sell", "КурсНбу": "nbu"}


def rates(path):
    """Курси купівлі, продажу і НБУ. Нулі у файлі – пропуски, заповнюємо лінійною інтерполяцією."""
    raw = pd.read_excel(path)
    frame = raw[list(RATE_COLUMNS)].rename(columns=RATE_COLUMNS).astype(float)
    frame.index = pd.to_datetime(raw["Дата"], dayfirst=True)
    gaps = int((frame == 0.0).sum().sum())
    frame = frame.replace(0.0, np.nan).interpolate(limit_direction="both")
    return frame, gaps


def _fetch(frequency, variables, start, cities):
    lat, lon = zip(*cities.values(), strict=True)
    params = {
        "latitude": ",".join(map(str, lat)),
        "longitude": ",".join(map(str, lon)),
        "start_date": start,
        "end_date": END,
        frequency: ",".join(variables),
        "timezone": "Europe/Kyiv",
    }
    response = requests.get(ARCHIVE, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = [payload]
    frames = []
    for city, block in zip(cities, payload, strict=True):
        part = pd.DataFrame(block[frequency]).rename(columns={"time": "date", **VARIABLES})
        part.insert(1, "city", city)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def weather(path):
    """Добові показники по містах у довгому форматі (date, city, temp, ...). Кешується у CSV."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _fetch("daily", VARIABLES, START, CITIES).to_csv(path, index=False)
    frame = pd.read_csv(path, parse_dates=["date"])
    columns = list(VARIABLES.values())
    by_city = frame.groupby("city")[columns]
    frame[columns] = by_city.transform(lambda s: s.interpolate(limit_direction="both"))
    return frame


def hourly_temperature(path, city="Київ"):
    """Погодинна температура одного міста – для декомпозиції з двома сезонностями."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = _fetch("hourly", ["temperature_2m"], HOURLY_START, {city: CITIES[city]})
        part.rename(columns={"temperature_2m": "temp"}).to_csv(path, index=False)
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date")["temp"].interpolate(limit_direction="both")


def wide(frame, variable):
    """Таблиця дата × місто для одного показника."""
    return frame.pivot(index="date", columns="city", values=variable)[list(CITIES)].asfreq("D")
