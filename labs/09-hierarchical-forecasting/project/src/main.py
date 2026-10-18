"""ЛР 9: ієрархічне прогнозування народжуваності, рівень III, варіант 8."""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import evaluate
import figures
import loaders
import plots
import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"
DATASET = (
    ROOT.parent
    / "docs/samples/Lecture_17_Lab9,Variants7-12/datasets/2_dataset_POPULATION_BIRTH.xlsx"
)

SEED = 42


def banner(text):
    print("=" * 80)
    print(text)


def show(table, title):
    print(f"\n{title}")
    print(table.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 160)
    OUT.mkdir(exist_ok=True)
    plots.setup()

    banner("ЕТАП 1. Парсинг 2_dataset_POPULATION_BIRTH.xlsx і попередній аналіз")
    births = loaders.births(DATASET, DATA / "births.csv")
    print(f"Рядків після відбору: {len(births)}")
    data = analysis.run(births)

    banner("ЕТАП 2. Ретроспективна перевірка методів, старти 2015–2018, горизонт 3 роки")
    errors = pd.concat([evaluate.backtest(births, depth) for depth in (2, 3)], ignore_index=True)
    tables = {}
    for depth in (2, 3):
        part = errors[errors["depth"] == depth]
        tables[depth] = evaluate.summary(part)
        show(tables[depth], f"MAPE, %, {depth} рівні")
        show(evaluate.summary(part, "ase"), f"MASE, {depth} рівні")
    best = evaluate.best(tables[3])
    print(f"\nНайменша середня MAPE за трьома рівнями: {best}")

    banner(f"ЕТАП 3. Прогноз на {evaluate.FINAL_HORIZON} років з повних даних")
    finals = {depth: evaluate.final(births, depth) for depth in (2, 3)}
    result = finals[3]
    print(f"Базові моделі: {pd.Series(result['models']).value_counts().to_dict()}")
    print(f"λ стискання коваріації в MinT: {result['lambda']:.3f}")
    gap = evaluate.incoherence(result["h"], result["F"])
    print(f"Розбіжність базових прогнозів країна vs Σ нижнього рівня, %: {np.round(gap, 2)}")
    country = pd.DataFrame(
        {m: f[0] for m, f in result["forecasts"].items()}, index=result["years"]
    ).T
    show(country.round(0), "Прогноз по Україні, осіб")
    nodes = figures.all_figures(data, errors, tables, finals, best, OUT)
    show(
        nodes[nodes["рівень"] != "місцевість"].drop(columns="рівень"),
        f"Прогноз {best} за областями",
    )

    banner("ЕТАП 4. Верифікація")
    for depth in (2, 3):
        print(f"{depth} рівні:")
        verify.run(finals[depth], np.random.default_rng(SEED))


if __name__ == "__main__":
    main()
