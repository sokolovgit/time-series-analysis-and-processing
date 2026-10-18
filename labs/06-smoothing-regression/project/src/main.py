"""ЛР 6: експоненційне згладжування і регресійні моделі. Рівень ІІ – групи вимог 1, 2 і 3."""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import figures
import loaders
import plots
import selector
import study
import synthetic
import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"
RATES = ROOT.parent / "docs/samples/Lab_work_6/exponential_models/data/Oschadbank (USD).xls"

SEED = 42
BENCH_SIZE = 450


def benchmark(rng, fresh):
    """Набір рядів для селектора; розрахунок довгий, тому результат кешується в out/."""
    path = OUT / "benchmark.csv"
    if path.exists() and not fresh:
        table = pd.read_csv(path)
        if len(table) == BENCH_SIZE:
            print(f"   набір з {BENCH_SIZE} рядів прочитано з {path.name}")
            return table
    table = selector.build_benchmark(rng, BENCH_SIZE)
    table.to_csv(path, index=False)
    return table


def stage_selector(rng, fresh):
    table = benchmark(rng, fresh)
    print(
        f"   рядів {len(table)}; краща сім'я: "
        + ", ".join(f"{k} {v}" for k, v in table["краще"].value_counts().items())
    )
    print(pd.crosstab(table["тренд"], table["краще"]).to_string())
    runs = selector.cross_validate(table)
    frame = pd.DataFrame(runs)
    print("   крос-валідація стратегій (RMSE / оракул на рядах поза навчанням):")
    print(frame.round(3).to_string())
    print(
        "   середнє: "
        + ", ".join(f"{k} {v:.3f}" for k, v in frame.drop(columns="k").mean().items())
    )
    tree, rules = selector.train(table)
    k, scores = selector.tune_threshold(table)
    print("   правила дерева на всьому наборі:")
    print("\n".join("     " + line for line in rules.splitlines()))
    print(
        "   поріг гібриду k за всім набором: "
        + ", ".join(f"{a:g} → {v:.3f}" for a, v in scores.items())
        + f"; обрано k = {k:g}"
    )
    return table, runs, tree, k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["verify", "study", "all"], default="all")
    parser.add_argument("--fresh", action="store_true", help="перерахувати набір рядів селектора")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    plots.setup()
    # окремий генератор на кожен етап, щоб кеш набору рядів не зсував синтетичні ряди
    rng_verify, rng_bench, rng_series = (np.random.default_rng([SEED, i]) for i in range(3))

    if args.part in ("verify", "all"):
        print("=" * 80)
        print("ВЕРИФІКАЦІЯ АЛГОРИТМІВ на рядах з відомими властивостями")
        verify.run(rng_verify)
    if args.part == "verify":
        return

    print("\n" + "=" * 80)
    print("СЕЛЕКТОР: набір синтетичних рядів з випадковими властивостями")
    table, cv_runs, tree, k = stage_selector(rng_bench, args.fresh)

    rates, _, gaps = loaders.rates(RATES)
    temperature = loaders.temperature(DATA / "kyiv_temperature.csv")
    cfgs = study.configs(
        synthetic.quadratic_normal(rng_series),
        synthetic.linear_exponential(rng_series),
        rates,
        temperature,
    )
    runs = []
    for i, cfg in enumerate(cfgs, 1):
        print("\n" + "=" * 80)
        extra = f", пропусків {gaps}" if cfg["key"] == "rates" else ""
        print(f"РЯД {i}. {cfg['title']}{extra}")
        runs.append(study.run(cfg, tree, k))

    figures.series(runs, OUT)
    figures.exponential_fit(runs, OUT)
    figures.alpha_curves(runs, OUT)
    figures.regression_fit(runs, OUT)
    figures.forecasts(runs, OUT)
    figures.kpi_heatmap(runs, OUT)
    figures.benchmark(table, cv_runs, OUT)


if __name__ == "__main__":
    main()
