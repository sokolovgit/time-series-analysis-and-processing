"""Попередній аналіз: повнота рядів, узгодженість сум, тренди, злами, міська частка."""

import numpy as np

import hierarchy
import loaders

BREAK = 2014


def run(births):
    table = births.pivot_table(index="year", columns=["region", "area"], values="births")
    years = table.index.to_numpy()
    regions = sorted(births["region"].unique())
    print(f"Період: {years[0]}–{years[-1]}, {len(years)} років; територій: {len(regions)}")
    lengths = {r: int(table[(r, "усього")].notna().sum()) for r in regions if r != loaders.COUNTRY}
    short = {r: n for r, n in lengths.items() if n < len(years)}
    print(f"Неповні ряди областей: {short}")
    print(f"Виключено з ієрархії (рядів немає після 2013): {', '.join(hierarchy.EXCLUDED)}")

    local = []
    for r in regions:
        if r == loaders.COUNTRY or (r, "міська") not in table:
            continue
        parts = table[(r, "міська")] + table.get((r, "сільська"), 0)
        local.append(np.nanmax(np.abs(parts - table[(r, "усього")])))
    print(f"max |міська + сільська − область| = {max(local):.0f}")

    h = hierarchy.build(births, 3)
    total = h["Y"][:, 0]
    official = table[(loaders.COUNTRY, "усього")].reindex(years).to_numpy()
    lost = sum(table[(r, "усього")].fillna(0).to_numpy() for r in hierarchy.EXCLUDED)
    known = ~np.isnan(official)
    gap = official[known] - total[known]
    print(
        f"Офіційний ряд «Україна» є з {years[known][0]} р.; "
        f"Україна − Σ 25 = Крим + Севастополь до {BREAK - 1}: "
        f"max |різниця| = {np.max(np.abs(gap - lost[known])):.0f}; "
        f"з {BREAK}: max |різниця| = {np.max(np.abs(gap[years[known] >= BREAK])):.0f}"
    )

    peak = years[np.argmax(np.where(years > 2005, total, 0))]
    low = years[np.argmin(np.where(years < 2005, total, np.inf))]
    print(
        f"Україна (25 регіонів): {years[0]} – {total[0]:.0f}, мінімум {low} – "
        f"{total[years == low][0]:.0f}, пік {peak} – {total[years == peak][0]:.0f}, "
        f"{years[-1]} – {total[-1]:.0f} ({100 * (total[-1] / total[years == peak][0] - 1):+.1f}%)"
    )
    urban = h["Y"][:, [n.endswith("міська") for n in h["names"]]].sum(axis=1)
    share = 100 * urban / total
    print(f"Частка міських народжень: {years[0]} – {share[0]:.1f}%, {years[-1]} – {share[-1]:.1f}%")

    regional = h["Y"][:, h["level"] == 1]
    names = np.array(h["names"])[h["level"] == 1]
    growth = np.diff(np.log(regional), axis=0)
    shocks = []
    for j, name in enumerate(names):
        k = np.argmin(growth[:, j])
        shocks.append((name, int(years[k + 1]), 100 * (np.exp(growth[k, j]) - 1)))
    shocks.sort(key=lambda s: s[2])
    print("Найбільші річні падіння:")
    for name, year, drop in shocks[:4]:
        print(f"  {name:18s} {year}: {drop:+.1f}%")
    change = 100 * (regional[-1] / regional[years == 2013][0] - 1)
    order = np.argsort(change)
    print(
        f"Зміна 2013→2021 за областями: від {change[order[0]]:+.1f}% ({names[order[0]]}) "
        f"до {change[order[-1]]:+.1f}% ({names[order[-1]]})"
    )
    return {
        "years": years,
        "total": total,
        "official": official,
        "share": share,
        "h": h,
        "change": change,
    }
