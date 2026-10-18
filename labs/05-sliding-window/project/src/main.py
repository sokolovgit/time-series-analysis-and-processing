"""ЛР 5: алгоритми ковзного вікна для обробки Time Series. Рівень ІІ – групи вимог 1 і 2."""

import argparse
import warnings
from pathlib import Path

import numpy as np

import figures
import loaders
import plots
import study
import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"
RATES = ROOT.parent / "docs/samples/Lab_work_5/ARMA_ARIMA/data/Oschadbank (USD).xls"

SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["verify", "study", "all"], default="all")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(exist_ok=True)
    plots.setup()

    if args.part in ("verify", "all"):
        print("=" * 80)
        print("ВЕРИФІКАЦІЯ АЛГОРИТМІВ на рядах з відомими властивостями")
        verify.run(np.random.default_rng(SEED))
    if args.part == "verify":
        return

    rates, _, gaps = loaders.rates(RATES)
    temperature = loaders.temperature(DATA / "kyiv_temperature.csv")
    runs = []
    for cfg, series, header in [
        (
            study.rates_config(),
            rates,
            f"РЯД 1. Курс купівлі USD в Ощадбанку, {len(rates)} записів, пропусків {gaps}",
        ),
        (
            study.temperature_config(),
            temperature,
            f"РЯД 2. Середньодобова температура Києва, {len(temperature)} діб",
        ),
    ]:
        print("\n" + "=" * 80)
        print(header)
        runs.append(study.run(cfg, series, OUT))

    print("\n" + "=" * 80)
    print("ПОХИБКА ЗА КРОКОМ ПРОГНОЗУ (RMSE)")
    figures.horizon_errors(runs, OUT)
    figures.arima_diagnostics(runs, OUT)
    figures.tuning_curves(runs, OUT)
    figures.optimization_gain(runs, OUT)
    figures.kpi_heatmap(runs, OUT)
    for key, (table, starts) in figures.lead_sweep(runs, OUT).items():
        print(f"   {key}, {starts} стартів:")
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
