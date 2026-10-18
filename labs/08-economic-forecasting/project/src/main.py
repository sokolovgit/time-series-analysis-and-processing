"""ЛР 8: проєктний практикум з прогнозування економічних показників. Рівні I–III."""

import argparse
import warnings
from pathlib import Path

import numpy as np

import figures
import level1
import level2
import level3
import loaders
import plots
import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"
SALES = ROOT.parent / "docs/samples/Lab_work_8/Pr_1.xls"

SEED = 42
PARTS = ("level1", "level2", "level3", "verify", "all")


def banner(text):
    print("=" * 80)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=PARTS, default="all")
    parser.add_argument("--fresh", action="store_true", help="перенавчити всі мережі")
    args = parser.parse_args()
    chosen = PARTS[:-1] if args.part == "all" else (args.part,)

    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    plots.setup()

    if "level1" in chosen:
        banner("РІВЕНЬ I. Продажі мережі магазинів, Pr_1.xls, варіант 8 – прибуток за регіонами")
        sales = level1.run(SALES, OUT)
        figures.sales(sales, OUT)

    cfgs = level2.configs(
        loaders.rate(DATA), loaders.cpi(DATA), loaders.retail(DATA), loaders.brent(DATA)
    )
    if "level2" in chosen:
        banner("РІВЕНЬ II. Множина економічних процесів з різними властивостями")
        processes = level2.run(cfgs)
        figures.processes(cfgs, processes, OUT)

    if "level3" in chosen:
        banner("РІВЕНЬ III. Три класи методів на тих самих рядах, інтегрована оцінка")
        series = [
            cfg | {"series": cfg["series"].to_numpy(dtype=float), "window": level3.WINDOW}
            for cfg in cfgs
        ]
        study = level3.run(series, OUT / "level3.pkl", args.fresh)
        figures.study(cfgs, study, OUT)

    if "verify" in chosen:
        banner("ВЕРИФІКАЦІЯ інтегрованої оцінки на сигналах з відомим кращим класом")
        checks = verify.run(np.random.default_rng(SEED), OUT / "verify.pkl", args.fresh)
        figures.verification(checks, OUT)


if __name__ == "__main__":
    main()
