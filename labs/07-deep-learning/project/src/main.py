"""ЛР 7: методи глибинного навчання для Time Series. Рівень IV – три типи мереж і довготривала
екстраполяція на інтервал, рівний інтервалу спостереження."""

import argparse
import warnings
from pathlib import Path

import numpy as np

import figures
import loaders
import networks
import plots
import study
import synthetic
import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"
RATES = ROOT.parent / "docs/samples/Lab_work_7/example_1/Oschadbank (USD).xls"

SEED = 42


def architectures(chosen, lookback):
    """Обрані структури: зведення шарів у текстовий файл і моделі для схеми."""
    models = {k: networks.build(k, lookback, c["units"]) for k, c in chosen.items()}
    lines = []
    for model in models.values():
        model.summary(print_fn=lambda s, line_break=None: lines.append(s))
    (OUT / "architectures.txt").write_text("\n".join(lines), encoding="utf-8")
    for kind, model in models.items():
        print(f"   {kind}: параметрів {model.count_params()}")
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["verify", "study", "all"], default="all")
    parser.add_argument("--fresh", action="store_true", help="перенавчити всі мережі")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    plots.setup()
    rng_verify, rng_series = (np.random.default_rng([SEED, i]) for i in range(2))

    if args.part in ("verify", "all"):
        print("=" * 80)
        print("ВЕРИФІКАЦІЯ мереж і стратегій на сигналах з відомою поведінкою")
        checks = verify.run(rng_verify, OUT / "verify.pkl", args.fresh)
        figures.verification(checks, OUT)
    if args.part == "verify":
        return

    rates, _, gaps = loaders.rates(RATES)
    temperature = loaders.temperature(DATA / "kyiv_temperature.csv")
    cfgs = study.configs(
        synthetic.quadratic_normal(rng_series),
        synthetic.linear_exponential(rng_series),
        rates,
        temperature,
    )
    props = {}
    for i, cfg in enumerate(cfgs, 1):
        print("\n" + "=" * 80)
        extra = f", пропусків {gaps}" if cfg["key"] == "rates" else ""
        print(f"РЯД {i}. {cfg['title']}{extra}")
        props[cfg["key"]] = study.properties(cfg)

    print("\n" + "=" * 80)
    print("СИНТЕЗ СТРУКТУРИ: розмір і функція втрат кожного типу мережі за трьома індикаторами")
    struct = study.structure(cfgs, props, OUT / "structure.pkl", args.fresh)
    models = architectures(struct["chosen"], cfgs[0]["lookback"])

    print("\n" + "=" * 80)
    print("ПРОГНОЗ: три мережі × три стратегії, walk-forward на горизонтах 0,1W, 0,5W і W")
    runs = study.forecasts(cfgs, props, struct["chosen"], OUT / "forecast.pkl", args.fresh)
    kpi, fits = {}, {}
    for i, cfg in enumerate(cfgs, 1):
        print("\n" + "=" * 80)
        print(f"РЯД {i}. {cfg['title']}")
        fits[cfg["key"]] = study.in_sample(cfg, runs[cfg["key"]])
        kpi[cfg["key"]] = study.evaluate(cfg, runs[cfg["key"]])

    figures.series(cfgs, OUT)
    figures.decomposition(cfgs, props, OUT)
    figures.architecture(models, struct["chosen"], OUT)
    figures.structure(struct, cfgs, OUT)
    figures.learning(cfgs, runs, OUT)
    figures.fit(cfgs, runs, OUT)
    figures.strategies_plot(cfgs, runs, OUT)
    figures.forecasts(cfgs, runs, OUT)
    figures.kpi_heatmap(cfgs, kpi, OUT)
    figures.horizon_curves(cfgs, runs, OUT)


if __name__ == "__main__":
    main()
