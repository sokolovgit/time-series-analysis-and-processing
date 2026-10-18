"""Рівень I, варіант 8: показники ефективності, МНК-модель і динаміка прибутку за регіонами."""

import numpy as np
import pandas as pd

import detect
import learn

MONTHS = (
    "Січень",
    "Лютий",
    "Березень",
    "Квітень",
    "Травень",
    "Червень",
    "Липень",
    "Серпень",
    "Вересень",
    "Жовтень",
    "Листопад",
    "Грудень",
)


def preliminary(frame, service):
    """Обсяг, пропуски, повернення, продажі нижче собівартості, покриття календаря."""
    days = frame["date"].dt.normalize()
    out = {
        "записів": len(frame),
        "службових рядків відкинуто": service,
        "пропусків": int(frame[["qty", "cost", "price"]].isna().sum().sum()),
        "магазинів": frame["store"].nunique(),
        "регіонів": frame["region"].nunique(),
        "без регіону": int(frame["region"].isna().sum()),
        "днів з продажами": days.nunique(),
        "повернень (кількість < 0)": int((frame["qty"] < 0).sum()),
        "продажів нижче собівартості": int((frame["price"] < frame["cost"]).sum()),
    }
    stats = frame[["qty", "cost", "price"]].describe().T
    stats.index = ["кількість", "собівартість", "ціна"]
    return out, stats


def indicators(frame):
    """Виручка V = q·p, витрати C = q·c, прибуток P = V − C = q(p − c), рентабельність P/V."""
    frame = frame.copy()
    frame["revenue"] = frame["qty"] * frame["price"]
    frame["costs"] = frame["qty"] * frame["cost"]
    frame["profit"] = frame["revenue"] - frame["costs"]
    frame["month_no"] = frame["date"].dt.month
    return frame


def summary(frame, by):
    table = frame.groupby(by)[["qty", "revenue", "costs", "profit"]].sum()
    table["рентабельність, %"] = 100 * table["profit"] / table["revenue"]
    return table.rename(
        columns={
            "qty": "продано, од.",
            "revenue": "виручка",
            "costs": "витрати",
            "profit": "прибуток",
        }
    )


def dynamics(frame):
    """Прибуток регіону за місяцями, місячна зміна ΔPₘ = (Pₘ − Pₘ₋₁)/|Pₘ₋₁|·100 і частка регіону."""
    profit = frame.pivot_table("profit", "month_no", "region", aggfunc="sum").fillna(0.0)
    profit.index = [MONTHS[m - 1] for m in profit.index]
    change = 100 * profit.diff() / profit.shift().abs()
    share = 100 * profit.div(profit.sum(axis=1), axis=0)
    return profit, change, share


def daily(frame, days, column="profit"):
    """Добовий ряд показника на повному календарі року, дні без продажів – нулі."""
    series = frame.groupby(frame["date"].dt.normalize())[column].sum()
    return series.reindex(days, fill_value=0.0)


def trend(y):
    """МНК-модель ряду: аномалії замінюються інтерполяцією, степінь – за відкладеним хвостом."""
    y = np.asarray(y, dtype=float)
    mask, window, k = detect.adaptive(y)
    clean = detect.clean(y, mask)
    degree, table, _ = learn.select_degree(clean)
    fitted, coeffs = learn.fit(clean, degree)
    return {
        "y": y,
        "clean": clean,
        "anomalies": mask,
        "window": window,
        "k": k,
        "degree": degree,
        "table": table,
        "fitted": fitted,
        "coeffs": coeffs,
        "scores": learn.scores(clean, fitted),
    }


def run(path, out):
    import loaders

    raw, service, regions = loaders.sales(path)
    pre, stats = preliminary(raw, service)
    print("   " + "; ".join(f"{k}: {v}" for k, v in pre.items()))
    print(stats.round(2).to_string())
    print("   магазинів за регіонами: " + str(regions["region"].value_counts().to_dict()))

    frame = indicators(raw)
    total = summary(frame.assign(all="усього"), "all")
    by_region = summary(frame, "region").sort_values("прибуток", ascending=False)
    by_month = summary(frame, "month_no")
    by_month.index = [MONTHS[m - 1] for m in by_month.index]
    print("\n   Показники ефективності за регіонами:")
    print(pd.concat([by_region, total]).round(1).to_string())
    print("\n   Показники ефективності за місяцями:")
    print(by_month.round(1).to_string())

    profit, change, share = dynamics(frame)
    print("\n   Динаміка прибутку за регіонами, грн:")
    print(profit.round(0).to_string())
    print("\n   Місячна зміна прибутку, %:")
    print(change.round(1).to_string())
    gap = abs(profit.to_numpy().sum() - frame["profit"].sum())
    print(f"\n   перевірка: Σ регіонів − загальний прибуток = {gap:.2e}")

    days = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    series = {"усього": daily(frame, days)}
    for region in by_region.index:
        series[region] = daily(frame[frame["region"] == region], days)
    models = {name: trend(y) for name, y in series.items()}
    print("\n   МНК-моделі добового прибутку:")
    for name, m in models.items():
        c = ", ".join(f"{v:.1f}" for v in m["coeffs"])
        print(
            f"   {name}: аномалій {m['anomalies'].sum()} (вікно {m['window']}, k = {m['k']:.2f}), "
            f"степінь {m['degree']}, C = [{c}], R² = {m['scores']['R2']:.3f}"
        )

    with pd.ExcelWriter(out / "profit_by_region.xlsx") as writer:
        profit.round(2).to_excel(writer, sheet_name="Прибуток, грн")
        change.round(2).to_excel(writer, sheet_name="Зміна, %")
        share.round(2).to_excel(writer, sheet_name="Частка, %")
        pd.concat([by_region, total]).round(2).to_excel(writer, sheet_name="Показники")
        pd.DataFrame(
            {"дата": days}
            | {f"{name}, факт": m["y"] for name, m in models.items()}
            | {f"{name}, МНК": m["fitted"] for name, m in models.items()}
        ).round(2).to_excel(writer, sheet_name="МНК-моделі", index=False)
    print(f"   збережено {out / 'profit_by_region.xlsx'}")
    return {
        "frame": frame,
        "pre": pre,
        "stats": stats,
        "by_region": by_region,
        "by_month": by_month,
        "total": total,
        "profit": profit,
        "change": change,
        "share": share,
        "dates": days,
        "models": models,
    }
