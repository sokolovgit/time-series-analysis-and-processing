"""Рівень II: аналіз множини економічних процесів з різними властивостями."""

import numpy as np
import pandas as pd

import detect
import features
import learn

FORECAST = 12
STRONG_SEASON = 0.5
SMOOTH_R2, SMOOTH_SIGMA = 0.9, 3.0


def configs(rate, cpi, retail, brent):
    return [
        {
            "key": "rate",
            "short": "курс",
            "title": "офіційний курс UAH/USD, НБУ",
            "unit": "грн за 1 USD",
            "series": rate,
        },
        {
            "key": "cpi",
            "short": "ІСЦ",
            "title": "індекс споживчих цін України (грудень 2006 = 100)",
            "unit": "%",
            "series": cpi,
        },
        {
            "key": "retail",
            "short": "роздріб США",
            "title": "роздрібна торгівля США, без сезонного коригування",
            "unit": "млрд дол.",
            "series": retail,
        },
        {
            "key": "brent",
            "short": "Brent",
            "title": "ціна нафти Brent",
            "unit": "дол./бар.",
            "series": brent,
        },
    ]


def returns(y):
    """Логарифмічні прирости rₜ = ln yₜ − ln yₜ₋₁ – відносна зміна показника за місяць."""
    return np.diff(np.log(y))


def kind(props):
    """Тип процесу за властивостями: сила сезонності, гладкість тренду, волатильність."""
    if props["ACF(12) приростів"] > STRONG_SEASON:
        return "тренд + сильна сезонність"
    if props["R² тренду"] > SMOOTH_R2 and props["σ приростів, %"] < SMOOTH_SIGMA:
        return "гладкий тренд" + (", слабка сезонність" if props["період"] else "")
    if props["R² тренду"] < SMOOTH_R2 - 0.1:
        return "волатильний, без стійкого тренду"
    return "випадкове блукання зі стрибками"


def analyze(cfg):
    """Статистики, стаціонарність рівнів і приростів, сезонність, тренд, аномальні місяці, МНК."""
    s = cfg["series"]
    y = s.to_numpy(dtype=float)
    r = returns(y)
    period, rho = features.annual(y)
    r2, _ = features.poly_r2(y, 2)
    level = features.stationarity(y)
    growth = features.stationarity(r)
    k = detect.threshold(len(r))
    mask = np.abs(r - np.median(r)) > k * r.std(ddof=1)
    degree, table, cut = learn.select_degree(y)
    fitted, coeffs = learn.fit(y, degree)
    forecast = learn.extrapolate(coeffs, len(y), FORECAST)
    props = {
        "n": len(y),
        "початок": s.index[0].strftime("%Y-%m"),
        "кінець": s.index[-1].strftime("%Y-%m"),
        "мінімум": y.min(),
        "максимум": y.max(),
        "R² тренду": r2,
        "період": period,
        "ACF(12) приростів": rho,
        "p ADF рівнів": level["p_ADF"],
        "p ADF приростів": growth["p_ADF"],
        "середній приріст, %": 100 * r.mean(),
        "σ приростів, %": 100 * r.std(ddof=1),
        "ACF(1) приростів": float(features.acf(r, [1])[0]),
        "ексцес приростів": features.describe(r)["ексцес"],
        "шоків": int(mask.sum()),
    }
    props["тип"] = kind(props)
    shocks = s.index[1:][mask]
    return {
        "props": props,
        "returns": r,
        "shocks": shocks,
        "shock_values": 100 * r[mask],
        "k": k,
        "degree": degree,
        "table": table,
        "cut": cut,
        "fitted": fitted,
        "coeffs": coeffs,
        "scores": learn.scores(y, fitted),
        "forecast": forecast,
        "future": pd.date_range(s.index[-1], periods=FORECAST + 1, freq="MS")[1:],
    }


def run(cfgs):
    results = {}
    for cfg in cfgs:
        res = analyze(cfg)
        p = res["props"]
        results[cfg["key"]] = res
        print(f"\n   {cfg['title']}: {p['початок']} – {p['кінець']}, n = {p['n']}")
        print(
            f"   приріст {p['середній приріст, %']:.2f} ± {p['σ приростів, %']:.2f} % за місяць, "
            f"ACF(1) = {p['ACF(1) приростів']:.3f}, ексцес = {p['ексцес приростів']:.1f}"
        )
        print(
            f"   p ADF: рівні {p['p ADF рівнів']:.3f}, прирости {p['p ADF приростів']:.3g}; "
            f"річна сезонність {'є' if p['період'] else 'немає'} "
            f"(ACF(12) = {p['ACF(12) приростів']:.3f})"
        )
        shocks = ", ".join(
            f"{d.strftime('%Y-%m')} ({v:+.1f} %)"
            for d, v in zip(res["shocks"], res["shock_values"], strict=True)
        )
        print(f"   шоки приростів (k = {res['k']:.2f}): {shocks or 'немає'}")
        rmse = ", ".join(f"{m}: {v['out']:.2f}" for m, v in res["table"].items())
        print(
            f"   МНК: RMSE на хвості {rmse}; обрано степінь {res['degree']}, "
            f"R² = {res['scores']['R2']:.3f}"
        )
        print(f"   тип: {p['тип']}")
    table = pd.DataFrame({cfg["short"]: results[cfg["key"]]["props"] for cfg in cfgs})
    print("\n   Порівняння властивостей рядів:")
    print(table.to_string())
    return {"results": results, "table": table}
