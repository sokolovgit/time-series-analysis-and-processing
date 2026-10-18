"""ЛР 4: пошук закономірностей в структурах Time Series. Рівень ІІІ – групи вимог 1 і 2."""

import argparse
import warnings
from pathlib import Path

import numpy as np

import plots
import study_rates
import study_weather
import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"
RATES = ROOT.parent / "docs/samples/Lab_work_4/Time_Series_Decomposition/data/Oschadbank (USD).xls"

SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["0", "1", "2", "all"], default="all")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(exist_ok=True)
    plots.setup()
    rng = np.random.default_rng(SEED)

    if args.group in ("0", "all"):
        print("=" * 80)
        print("ВЕРИФІКАЦІЯ МЕТОДІВ на рядах з відомими властивостями")
        verify.run(np.random.default_rng(SEED))
    if args.group in ("1", "all"):
        print("=" * 80)
        print("ГРУПА ВИМОГ 1. Курс USD в Ощадбанку (дані ЛР 1–3)")
        study_rates.run(RATES, OUT, rng)
    if args.group in ("2", "all"):
        print("\n" + "=" * 80)
        print("ГРУПА ВИМОГ 2. Погодний архів Open-Meteo, 12 міст України, 2015–2024")
        study_weather.run(DATA, OUT, rng)


if __name__ == "__main__":
    main()
