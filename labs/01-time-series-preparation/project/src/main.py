"""Точка входу: послідовний прогін етапів лабораторної роботи."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import requests
from numpy.random import Generator
from numpy.typing import NDArray

import distributions
import models
import parsing
import plotting
import stats
import synthesis
import trends
from config import Settings
from paths import OSCHADBANK_USD, ensure_assets

Series = NDArray[np.float64]

# Показники з Oschadbank (USD).xls та латинські назви для імен файлів.
REAL_COLUMNS = {"Купівля": "buy", "Продаж": "sell", "КурсНбу": "nbu"}

# Джерело для завдання III рівня: річні світові викиди CO2, звичайна HTML-таблиця.
SOURCE_URL = "https://www.worldometers.info/co2-emissions/co2-emissions-by-year/"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def stage_random(s: Settings, rng: Generator, out: Path) -> dict[str, Series]:
    """Етап 1. Моделі генерації випадкової величини за заданими законами розподілу."""
    section("Етап 1. Генерація випадкової величини")
    noises: dict[str, Series] = {}
    for law in s.noise_laws:
        noise = distributions.generate(rng, law, s.sample_size)
        mean, variance = distributions.theoretical(law)
        print(f"{law}:")
        print(f"  розраховано: {stats.describe(noise)}")
        print(f"  теоретично:  M = {mean:.4f}, D = {variance:.4f}")
        plotting.plot_histogram(
            noise, f"Закон розподілу похибки: {law}", out / f"01_hist_{law}.png"
        )
        noises[law] = noise
    return noises


def stage_trends(s: Settings, out: Path) -> dict[str, Series]:
    """Етап 2. Моделі зміни (ідеальний тренд) досліджуваного процесу."""
    section("Етап 2. Ідеальний тренд")
    ideals = {law: trends.build(law, s.sample_size) for law in s.trend_laws}
    for law, ideal in ideals.items():
        print(
            f"{law}: параметри {trends.DEFAULT_PARAMS[law]}, S(0) = {ideal[0]:.4f}, "
            f"S(n-1) = {ideal[-1]:.4f}"
        )
    plotting.plot_series(
        ideals,
        "Ідеальний тренд досліджуваного процесу",
        out / "02_trends.png",
        ylabel="ідеальний тренд S(t)",
    )
    return ideals


def stage_additive(
    noises: dict[str, Series],
    ideals: dict[str, Series],
    out: Path,
) -> dict[tuple[str, str], Series]:
    """Етап 3. Адитивна модель вибірки: тренд + похибка, комбінаторика довільна."""
    section("Етап 3. Адитивна модель статистичної вибірки")
    samples: dict[tuple[str, str], Series] = {}
    for trend_law, noise_law in product(ideals, noises):
        sample = models.additive(ideals[trend_law], noises[noise_law])
        samples[trend_law, noise_law] = sample
        print(f"{trend_law} + {noise_law}: {stats.describe(sample)}")
        plotting.plot_series(
            {"ідеальний тренд": ideals[trend_law], "вибірка": sample},
            f"Адитивна модель: {trend_law} + {noise_law}",
            out / f"03_additive_{trend_law}_{noise_law}.png",
            ylabel="значення вибірки Y(t)",
        )
    return samples


def stage_statistics(samples: dict[tuple[str, str], Series], out: Path) -> None:
    """Етап 4. Числові характеристики вибірок із виділенням тренду за МНК."""
    section("Етап 4. Статистичні характеристики вибірок")
    for (trend_law, noise_law), sample in samples.items():
        fitted, _ = stats.mnk_fit(sample, trends.DEGREE[trend_law])
        residual = stats.detrended(sample, fitted)
        print(f"{trend_law} + {noise_law}:")
        print(f"  вибірка:            {stats.describe(sample)}")
        print(f"  похибка (МНК):      {stats.describe(residual)}")
        print(f"  теоретично похибка: D = {distributions.theoretical(noise_law)[1]:.4f}")
        print(f"  R**2 = {stats.r_squared(sample, fitted):.6f}")

        # Гістограми вимагає етап 4 завдання. Для вибірки з трендом закон розподілу
        # не проглядається, тому поруч показуємо залишок після зняття тренду.
        if trend_law == "quadratic":
            plotting.plot_histogram_pair(
                {"вибірка Y(t)": sample, "похибка після зняття тренду": residual},
                f"Закон розподілу: {trend_law} + {noise_law}",
                out / f"04_hist_{trend_law}_{noise_law}.png",
            )


def stage_anomalies(
    samples: dict[tuple[str, str], Series],
    rng: Generator,
    out: Path,
) -> None:
    """Аномальні виміри: вплив викидів на статистичні характеристики."""
    section("Аномальні виміри")
    key = next(iter(samples))
    sample = samples[key]
    polluted = models.with_anomalies(sample, rng, fraction=0.1, magnitude=3.0)
    print(f"{key[0]} + {key[1]}, 10% аномальних вимірів:")
    print(f"  без АВ: {stats.describe(sample)}")
    print(f"  з АВ:   {stats.describe(polluted)}")
    plotting.plot_series(
        {"вибірка": sample, "вибірка з АВ": polluted},
        f"Аномальні виміри: {key[0]} + {key[1]}",
        out / "05_anomalies.png",
        ylabel="значення вибірки Y(t)",
    )


def stage_real_data(s: Settings, out: Path) -> None:
    """Етап 5. Статистичні характеристики реальних даних з Oschadbank (USD).xls."""
    section("Етап 5. Реальні дані: Oschadbank (USD).xls")
    frame = parsing.read_rates(OSCHADBANK_USD)
    dates = frame["Дата"].to_numpy()
    columns = list(REAL_COLUMNS) if s.level > 1 else list(REAL_COLUMNS)[:1]
    for column in columns:
        raw = frame[column].to_numpy(dtype=float)
        real, gaps = parsing.fill_gaps(raw)
        fitted, coeffs = stats.mnk_fit(real, degree=2)
        slug = REAL_COLUMNS[column]
        print(f"{column} ({len(real)} відліків, пропусків заповнено: {gaps}):")
        if gaps:
            print(f"  до обробки:    {stats.describe(raw)}")
        print(f"  ряд:           {stats.describe(real)}")
        print(f"  похибка (МНК): {stats.describe(stats.detrended(real, fitted))}")
        print(f"  тренд: {coeffs[2]:.3e}*t**2 + {coeffs[1]:.3e}*t + {coeffs[0]:.4f}")
        print(f"  R**2 = {stats.r_squared(real, fitted):.6f}")
        plotting.plot_series(
            {column: real, "тренд за МНК": fitted},
            f"Курс USD, Ощадбанк: {column}",
            out / f"06_real_{slug}.png",
            x=dates,
            xlabel="дата",
            ylabel="курс, грн за 1 USD",
        )
        plotting.plot_histogram(
            stats.detrended(real, fitted),
            f"Розподіл похибки реальних даних: {column}",
            out / f"06_hist_{slug}.png",
            xlabel="відхилення від тренду, грн",
        )


def stage_web(rng: Generator, out: Path) -> None:
    """Завдання III рівня: парсинг сайту, збереження у файл і синтез подібної моделі."""
    section("Завдання III рівня. Парсинг сайту та синтез подібної моделі")
    try:
        html = parsing.fetch_page(SOURCE_URL)
    except requests.RequestException as error:
        print(f"сайт недоступний ({error}), етап пропущено")
        return

    frame = parsing.parse_emissions(html)
    file = parsing.save(frame, out / "07_parsed.csv")
    years = frame["Рік"].to_numpy()
    print(f"Джерело: {SOURCE_URL}")
    print(f"Отримано {len(frame)} записів за {years[0]}-{years[-1]}, збережено у {file.name}")

    # Ряд у мільярдах тонн – інакше коефіцієнти тренду нечитні.
    real = frame["Викиди"].to_numpy(dtype=float) / 1e9
    result = synthesis.like(real, rng, degree=2)

    print(f"  реальні дані:  {stats.describe(real)} (млрд тонн)")
    print(f"  похибка (МНК): {stats.describe(result.residual)}")
    print(
        f"  тренд: {result.coeffs[2]:.3e}*t**2 + {result.coeffs[1]:.3e}*t + {result.coeffs[0]:.4f}"
    )
    print(f"  синтезована:   {stats.describe(result.series)}")
    print(f"  R**2 тренду = {stats.r_squared(real, result.trend):.6f}")

    worst = int(np.argmin(result.residual))
    print(
        f"  найбільше відхилення вниз від тренду: {years[worst]} рік,"
        f" {result.residual[worst]:.3f} млрд тонн"
    )

    plotting.plot_series(
        {"викиди CO$_2$": real, "тренд за МНК": result.trend},
        "Світові викиди CO$_2$ (worldometers.info)",
        out / "07_parsed.png",
        x=years,
        xlabel="рік",
        ylabel="викиди CO$_2$, млрд тонн",
    )
    plotting.plot_series(
        {"реальні дані": real, "синтезована модель": result.series},
        "Верифікація: реальні дані та синтезована модель",
        out / "08_synthetic.png",
        x=years,
        xlabel="рік",
        ylabel="викиди CO$_2$, млрд тонн",
    )


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
    out = ensure_assets()

    print(f"Варіант {s.variant}, рівень {s.level}, обсяг вибірки {s.sample_size}, seed {s.seed}")
    print(f"Похибка: {', '.join(s.noise_laws)}. Тренд: {', '.join(s.trend_laws)}.")

    noises = stage_random(s, rng, out)
    ideals = stage_trends(s, out)
    samples = stage_additive(noises, ideals, out)
    stage_statistics(samples, out)
    stage_anomalies(samples, rng, out)
    stage_real_data(s, out)
    if s.level >= 3:
        stage_web(rng, out)

    print(f"\nГрафіки та експорти збережено у {out}")


if __name__ == "__main__":
    main()
