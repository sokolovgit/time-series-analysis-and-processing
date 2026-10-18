"""Вхідні дані: курс купівлі USD в Ощадбанку і добова температура Києва з архіву Open-Meteo."""

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
KYIV = (50.45, 30.52)
START, END = "2015-01-01", "2024-12-31"


def rates(path, column="Купівля"):
    """Курс з файлу Ощадбанку за номером запису. Нулі – пропуски, їх заповнює інтерполяція."""
    raw = pd.read_excel(path)
    values = raw[column].replace(0.0, np.nan).interpolate(limit_direction="both")
    gaps = int((raw[column] == 0.0).sum())
    dates = pd.to_datetime(raw["Дата"], dayfirst=True)
    return pd.Series(values.to_numpy(dtype=float), name="buy"), dates, gaps


def temperature(path):
    """Середньодобова температура Києва 2015–2024. Кешується у CSV."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "latitude": KYIV[0],
            "longitude": KYIV[1],
            "start_date": START,
            "end_date": END,
            "daily": "temperature_2m_mean",
            "timezone": "Europe/Kyiv",
        }
        response = requests.get(ARCHIVE, params=params, timeout=120)
        response.raise_for_status()
        daily = response.json()["daily"]
        frame = pd.DataFrame({"date": daily["time"], "temp": daily["temperature_2m_mean"]})
        frame.to_csv(path, index=False)
    frame = pd.read_csv(path, parse_dates=["date"])
    series = frame.set_index("date")["temp"].asfreq("D").interpolate(limit_direction="both")
    return series.rename("temp")
