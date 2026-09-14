"""Точка входу: послідовний прогін етапів лабораторної роботи."""

import argparse
from collections.abc import Callable
from itertools import product

import numpy as np
from numpy.random import Generator

from lab1 import distributions, models, parsing, plotting, stats, trends
from lab1.config import Settings
from lab1.paths import OSCHADBANK_USD, ensure_output


def stage_random(s: Settings, rng: Generator) -> None:
    """Етап 1. Моделі генерації випадкової величини за заданими законами розподілу."""
    for law in s.noise_laws:
        noise = distributions.generate(rng, law, s.sample_size)
        print(f"{law}: {stats.describe(noise)}")
        plotting.plot_histogram(
            noise,
            f"Закон розподілу похибки: {law}",
            ensure_output() / f"hist_{law}.png",
        )


def stage_trend(s: Settings, rng: Generator) -> None:
    """Етап 2. Моделі зміни (ідеальний тренд) досліджуваного процесу."""
    series = {law: trends.build(law, s.sample_size) for law in s.trend_laws}
    plotting.plot_series(series, "Ідеальний тренд", ensure_output() / "trends.png")


def stage_additive(s: Settings, rng: Generator) -> None:
    """Етап 3. Адитивна модель вибірки: тренд + похибка, довільна комбінаторика."""
    for noise_law, trend_law in product(s.noise_laws, s.trend_laws):
        ideal = trends.build(trend_law, s.sample_size)
        noise = distributions.generate(rng, noise_law, s.sample_size)
        sample = models.additive(ideal, noise)
        plotting.plot_series(
            {"тренд": ideal, "вибірка": sample},
            f"Адитивна модель: {trend_law} + {noise_law}",
            ensure_output() / f"additive_{trend_law}_{noise_law}.png",
        )


def stage_statistics(s: Settings, rng: Generator) -> None:
    """Етап 4. Числові характеристики синтезованих вибірок."""
    for noise_law, trend_law in product(s.noise_laws, s.trend_laws):
        ideal = trends.build(trend_law, s.sample_size)
        noise = distributions.generate(rng, noise_law, s.sample_size)
        sample = models.additive(ideal, noise)
        print(f"{trend_law} + {noise_law}")
        print(f"  вибірка: {stats.describe(sample)}")
        print(f"  похибка: {stats.describe(stats.detrended(sample, ideal))}")


def stage_real_data(s: Settings, rng: Generator) -> None:
    """Етап 5. Статистичні характеристики реальних даних з Oschadbank (USD).xls."""
    # На II та III рівні складності потрібні 3 показники замість одного.
    columns = ("Купівля", "Продаж", "НБУ") if s.level > 1 else ("Купівля",)
    for column in columns:
        real = parsing.read_excel_column(OSCHADBANK_USD, column=column)
        print(f"{column}: {stats.describe(real)}")


def stage_web_parsing(s: Settings, rng: Generator) -> None:
    """Завдання III рівня. Парсинг сайту, збереження у файл, синтез подібної моделі."""
    if s.level < 3:
        print("пропущено: потрібен III рівень складності")
        return
    html = parsing.fetch_page("https://www.oschadbank.ua/rates-archive")
    frame = parsing.parse_table(html)
    parsing.save(frame, ensure_output() / "parsed.csv")


Stage = Callable[[Settings, Generator], None]

STAGES: list[tuple[str, Stage]] = [
    ("1. Генерація випадкової величини", stage_random),
    ("2. Ідеальний тренд", stage_trend),
    ("3. Адитивна модель вибірки", stage_additive),
    ("4. Статистичні характеристики", stage_statistics),
    ("5. Реальні дані (Oschadbank USD)", stage_real_data),
    ("6. Парсинг сайту (III рівень)", stage_web_parsing),
]


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(prog="lab1", description="ЛР 1: підготовка Time Series")
    parser.add_argument("--variant", type=int, default=defaults.variant, help="місяць народження")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=defaults.level)
    parser.add_argument("--n", type=int, default=defaults.sample_size, help="обсяг вибірки")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()
    return Settings(variant=args.variant, level=args.level, sample_size=args.n, seed=args.seed)


def main() -> None:
    s = parse_args()
    rng = np.random.default_rng(s.seed)

    print(f"Варіант {s.variant}, рівень {s.level}, обсяг вибірки {s.sample_size}, seed {s.seed}")
    print(f"Похибка: {', '.join(s.noise_laws)}. Тренд: {', '.join(s.trend_laws)}.")

    for title, stage in STAGES:
        print(f"\n--- {title} ---")
        try:
            stage(s, rng)
        except NotImplementedError:
            print("не реалізовано")


if __name__ == "__main__":
    main()
