"""Вхідні дані: продажі мережі магазинів (Pr_1.xls) і чотири економічні показники з НБУ та FRED."""

import io

import pandas as pd
import requests

NBU_RATE = "https://bank.gov.ua/NBU_Exchange/exchange_site"
NBU_CPI = "https://bank.gov.ua/NBUStatService/v1/statdirectory/inflation"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
RATE_START = "1996-09-02"
END = "2026-09-30"
TIMEOUT = 180
REGION_NAMES = {"Днепро-Слобідський": "Дніпро-Слобідський"}

SALES_COLUMNS = {
    "Код магазину": "store",
    "Дата": "date",
    "Місяц": "month",
    "КільКість реалізацій": "qty",
    "Собівартість одиниці": "cost",
    "Ціна реализації": "price",
}


def sales(path):
    """Продажі з довідником «магазин → регіон», який лежить у тому ж аркуші праворуч.

    Повертає таблицю продажів з регіоном і кількість відкинутих службових рядків.
    """
    raw = pd.read_excel(path)
    regions = raw[["Код магазину.1", "Регіон"]].dropna()
    regions.columns = ["store", "region"]
    regions["store"] = regions["store"].astype(int)
    # у файлі назва регіону записана російською транслітерацією
    regions["region"] = regions["region"].replace(REGION_NAMES)
    frame = raw[list(SALES_COLUMNS)].rename(columns=SALES_COLUMNS)
    # останній рядок файлу – маркер «Стоп» без продажу
    service = frame["month"] == "Стоп"
    frame = frame[~service].merge(regions, on="store", how="left", validate="many_to_one")
    return frame, int(service.sum()), regions


def cached(path, download):
    """CSV з кешу; при першому зверненні завантажується і зберігається."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        download().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"])


def monthly(frame, column):
    return frame.set_index("date")[column].resample("MS").mean().loc[:END]


def rate(data):
    """Офіційний курс гривні до долара НБУ, середній за місяць."""

    def download():
        params = {
            "start": RATE_START.replace("-", ""),
            "end": END.replace("-", ""),
            "valcode": "usd",
            "sort": "exchangedate",
            "order": "asc",
            "json": "",
        }
        response = requests.get(NBU_RATE, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        frame = pd.DataFrame(response.json())
        dates = pd.to_datetime(frame["exchangedate"], format="%d.%m.%Y")
        return pd.DataFrame({"date": dates, "value": frame["rate_per_unit"]})

    return monthly(cached(data / "nbu_usd.csv", download), "value").rename("rate")


def cpi(data):
    """Ланцюговий індекс споживчих цін України, грудень 2006 = 100.

    НБУ публікує місячну зміну ІСЦ (PCPM, %) з датою наступного місяця: значення з датою
    2022-04-01 – це інфляція березня 2022. Індекс: Iₜ = 100 · Π (1 + pₖ/100).
    """

    def download():
        params = {"period": "m", "mcrd081": "Total", "json": ""}
        response = requests.get(NBU_CPI, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        frame = pd.DataFrame(response.json())
        frame = frame[
            frame["ku"].isna()
            & (frame["mcrk110"] == "NULL")
            & (frame["tzep"] == "PCPM_")
            & (frame["id_api"] == "prices_price_cpi_")
        ]
        dates = pd.to_datetime(frame["dt"], format="%Y%m%d") - pd.DateOffset(months=1)
        return pd.DataFrame({"date": dates, "value": frame["value"]}).sort_values("date")

    change = monthly(cached(data / "nbu_cpi.csv", download), "value")
    return (100 * (1 + change / 100).cumprod()).rename("cpi")


def fred(data, series_id):
    def download():
        response = requests.get(FRED, params={"id": series_id}, timeout=TIMEOUT)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), na_values=".")
        frame.columns = ["date", "value"]
        return frame.dropna()

    return monthly(cached(data / f"fred_{series_id.lower()}.csv", download), "value")


def retail(data):
    """Обсяг роздрібної торгівлі США без сезонного коригування, млрд дол. на місяць."""
    return (fred(data, "RSXFSN") / 1000).rename("retail")


def brent(data):
    """Ціна нафти Brent, середня за місяць, дол./барель."""
    return fred(data, "DCOILBRENTEU").rename("brent")
