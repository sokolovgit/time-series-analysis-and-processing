"""Вхідні дані: річні світові викиди CO₂ з worldometers.info (веб-скрапінг з кешем у CSV)."""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://www.worldometers.info/co2-emissions/co2-emissions-by-year/"
USER_AGENT = "Mozilla/5.0 (compatible; TimeSeriesStudy/1.0)"
TIMEOUT = 20
YEAR = re.compile(r"^\d{4}$")


def parse(html):
    """Таблиця сторінки: Year, CO2 emissions (tons), 1 Year Change, Per Capita.

    Числа записані з роздільником тисяч (39,632,664,219).
    """
    table = BeautifulSoup(html, "lxml").find("table")
    if table is None:
        raise ValueError("на сторінці немає таблиці з викидами")
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 4 or not YEAR.match(cells[0]):
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


def fetch():
    response = requests.get(SOURCE_URL, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return parse(response.text)


def emissions(path, refresh=False):
    """Роки і викиди в млрд т. Сайт опитується лише без кешу або з refresh.

    Повертає також джерело ряду: «сайт» або «кеш».
    """
    source = "кеш"
    if refresh or not path.exists():
        frame = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        source = "сайт"
    frame = pd.read_csv(path).sort_values("Рік")
    return frame["Рік"].to_numpy(dtype=int), frame["Викиди"].to_numpy(dtype=float) / 1e9, source
