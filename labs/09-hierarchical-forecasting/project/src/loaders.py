"""Парсинг датасету Держстату: кількість живонароджених за регіонами і типом місцевості."""

import warnings

import pandas as pd

INDICATOR = "Кількість живонароджених"
TOTALS = {
    "Вік": "Усього",
    "Вік матері": "Усього",
    "Вік батька": "Усього",
    "Стать": "Усього",
    "Сімейний cтан": "Не застосовується",
    "Місяць реєстрації події": "Загалом",
    "Громадянство матері": "Усього",
    "Громадянство батька": "Усього",
    "Періодичність": "Річна",
}
COUNTRY = "Україна"
AREAS = {"Загалом": "усього", "Міська": "міська", "Сільська": "сільська"}


def births(path, cache):
    """Річні ряди у довгому форматі: region, area, year, births.

    Файл містить 18 показників у розрізах віку, статі, громадянства тощо; лишаються тільки
    підсумкові рядки «Кількість живонароджених» за територією і типом місцевості.
    """
    if cache.exists():
        return pd.read_csv(cache)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(path)
    mask = raw["Показник"] == INDICATOR
    for column, value in TOTALS.items():
        mask &= raw[column] == value
    rows = raw[mask]
    years = [c for c in raw.columns if str(c).isdigit()]
    frame = rows.melt(
        id_vars=["Територіальний розріз", "Тип місцевості"],
        value_vars=years,
        var_name="year",
        value_name="births",
    ).dropna(subset=["births"])
    frame.columns = ["region", "area", "year", "births"]
    frame["area"] = frame["area"].map(AREAS)
    frame["year"] = frame["year"].astype(int)
    frame["births"] = frame["births"].astype(int)
    frame = frame.sort_values(["region", "area", "year"], ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame
