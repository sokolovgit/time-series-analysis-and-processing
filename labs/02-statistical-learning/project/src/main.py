"""ЛР 2: статистичне навчання за Big Data Time Series. Варіант 8, рівень складності III."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import detect
import learn
import series

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "out"
RATES = ROOT / "docs/samples/data/Oschadbank (USD).xls"

N = 10_000
SEED = 42
FRACTION = 0.1
HORIZON = 0.5
FRAGMENT = 300


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def axes(title, ylabel="Значення", height=4.5):
    fig, ax = plt.subplots(figsize=(10, height))
    ax.set_title(title)
    ax.set_xlabel("Номер відліку")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig, ax


def confusion(mask, truth, n):
    """Виявлені, хибні та пропущені аномалії відносно внесених."""
    real = np.zeros(n, dtype=bool)
    real[truth] = True
    return int((mask & real).sum()), int((mask & ~real).sum()), int((~mask & real).sum())


def stage_input(rng):
    print("\n1. Синтез вибірки")
    ideal = series.trend(N)
    clean_data = series.sample(N, rng)
    data, index = series.add_anomalies(clean_data, rng, FRACTION)
    print(f"   обсяг {N}, квадратичний тренд, похибка N(0, {series.SIGMA}²)")
    print(f"   внесено {len(index)} аномальних вимірів ({FRACTION:.0%}), зсув від 5σ")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7))
    top.plot(data, lw=0.5, color="#9dc3e6", label="вибірка з АВ")
    top.plot(ideal, lw=2.0, color="#c00000", label="ідеальний тренд")
    top.set_title("Синтезована вибірка")
    head = slice(0, FRAGMENT)
    bottom.plot(np.arange(FRAGMENT), data[head], lw=0.8, color="#9dc3e6", label="вибірка з АВ")
    bottom.plot(np.arange(FRAGMENT), ideal[head], lw=2.0, color="#c00000", label="ідеальний тренд")
    inside = index[index < FRAGMENT]
    bottom.scatter(inside, data[inside], s=22, color="#e08214", zorder=3, label="внесені АВ")
    bottom.set_title(f"Фрагмент: перші {FRAGMENT} відліків")
    for ax in (top, bottom):
        ax.set_xlabel("Номер відліку")
        ax.set_ylabel("Значення")
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, "01_input.png")
    return clean_data, data, index


def stage_detect(clean_data, data, index):
    print("\n2. Виявлення аномальних вимірів")
    false_fixed = int(detect.fixed(clean_data).sum())
    mask_clean, window, k = detect.adaptive(clean_data)
    false_adaptive = int(mask_clean.sum())
    print(
        f"   на чистій вибірці: фіксований 3σ – {false_fixed} хибних, адаптивний – {false_adaptive}"
    )
    print(f"   підібрані параметри: вікно {window}, поріг {k:.2f}σ")

    rows = {"фіксований 3σ": detect.fixed(data), "адаптивний": detect.adaptive(data)[0]}
    print(f"   {'детектор':<16}{'виявлено':>10}{'хибних':>9}{'пропущено':>12}")
    for name, mask in rows.items():
        tp, fp, fn = confusion(mask, index, N)
        print(f"   {name:<16}{tp:>10}{fp:>9}{fn:>12}")

    fig, ax = axes(f"Виявлення АВ, фрагмент з перших {FRAGMENT} відліків")
    head = np.arange(FRAGMENT)
    ax.plot(head, data[head], lw=0.8, color="#9dc3e6", label="вибірка з АВ")
    inside = index[index < FRAGMENT]
    ax.scatter(inside, data[inside], s=40, color="#e08214", label="внесені АВ")
    found = np.flatnonzero(rows["адаптивний"][head])
    ax.scatter(found, data[found], s=40, marker="x", color="#0070c0", label="виявлені")
    ax.legend()
    save(fig, "02_detect.png")

    base = detect.residual(clean_data, 5)
    adapted = detect.residual(clean_data, window)
    panels = (
        (base, 3.0 * base.std(), f"Базовий: вікно 5, поріг 3σ – {false_fixed} хибних"),
        (
            adapted,
            k * detect.mad_sigma(adapted),
            f"Адаптивний: вікно {window}, поріг {k:.2f}σ – {false_adaptive} хибних",
        ),
    )
    fig, stacked = plt.subplots(2, 1, figsize=(10, 7), sharey=True)
    for ax, (r, level, title) in zip(stacked, panels, strict=True):
        ax.plot(r, lw=0.4, color="#9dc3e6")
        ax.axhline(level, color="#c00000", label=f"поріг ±{level:.1f}")
        ax.axhline(-level, color="#c00000")
        ax.set_title(title)
        ax.set_xlabel("Номер відліку")
        ax.set_ylabel("Залишок")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
    fig.suptitle("Пороги детекторів на чистій вибірці, аномалій не внесено")
    save(fig, "03_thresholds.png")
    print(
        f"   пороги на чистій вибірці: фіксований ±{panels[0][1]:.1f}, "
        f"адаптивний ±{panels[1][1]:.1f}"
    )
    return detect.clean(data, rows["адаптивний"])


def stage_degree(data, name, figure):
    best, table, cut = learn.select_degree(data)
    print(f"   {'степінь':<10}{'RMSE навчання':>16}{'RMSE прогнозу':>16}")
    for m, row in table.items():
        mark = " <" if m == best else ""
        print(f"   {m:<10}{row['in']:>16.3f}{row['out']:>16.3f}{mark}")
    print(f"   обрано степінь {best}: навчання на {cut} відліках, перевірка на {len(data) - cut}")

    degrees = list(table)
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(10, 4.2))
    full.set_yscale("log")
    full.set_title("Усі степені")
    zoom.set_title(f"Без степеня {degrees[0]}")
    for ax, shown in ((full, degrees), (zoom, degrees[1:])):
        ax.plot(shown, [table[m]["in"] for m in shown], "o-", label="на навчальній частині")
        ax.plot(shown, [table[m]["out"] for m in shown], "s-", label="на відкладеному хвості")
        ax.set_xlabel("Степінь полінома")
        ax.set_ylabel("RMSE")
        ax.set_xticks(shown)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Вибір степеня полінома: {name}")
    save(fig, figure)
    return best


def stage_learn(data, degree, name, prefix, ylabel="Значення", ideal=None):
    model, coeffs = learn.fit(data, degree)
    quality = learn.scores(data, model)
    n = len(data)
    print("   " + "  ".join(f"{key} = {value:.4f}" for key, value in quality.items()))
    real = [c / n**i for i, c in enumerate(coeffs)]
    print("   коефіцієнти при t: " + ", ".join(f"c{i} = {c:.4g}" for i, c in enumerate(real)))

    horizon = int(n * HORIZON)
    forecast = learn.extrapolate(coeffs, n, horizon)

    fig, ax = axes(f"Результат навчання: {name}", ylabel)
    ax.plot(data, lw=0.5, color="#9dc3e6", label="очищена вибірка")
    if ideal is not None:
        ax.plot(ideal, lw=1.5, color="#7f7f7f", ls="--", label="ідеальний тренд")
    ax.plot(model, lw=2.0, color="#c00000", label=f"МНК, степінь {degree}")
    ax.legend()
    save(fig, f"{prefix}_fit.png")

    fig, ax = axes(f"Екстраполяція на {horizon} відліків: {name}", ylabel)
    ax.plot(data, lw=0.5, color="#9dc3e6", label="очищена вибірка")
    ax.plot(model, lw=1.5, color="#c00000", label="навчена модель")
    ax.plot(np.arange(n, n + horizon), forecast, lw=2.0, color="#0070c0", label="прогноз")
    ax.axvline(n, color="#7f7f7f", ls=":")
    ax.legend()
    save(fig, f"{prefix}_forecast.png")
    return quality


def stage_real():
    print("\n5. Реальні дані: курс USD, Ощадбанк")
    data, gaps = series.real_rates(RATES)
    print(f"   {len(data)} відліків, заповнено {gaps} пропусків інтерполяцією")

    mask, window, k = detect.adaptive(data)
    cleaned = detect.clean(data, mask)
    print(f"   вікно {window}, поріг {k:.2f}σ, виявлено {int(mask.sum())} аномальних вимірів")

    fig, ax = axes("Курс USD: виявлені аномальні виміри", "Курс, грн")
    ax.plot(data, lw=0.8, color="#9dc3e6", label="вихідний ряд")
    found = np.flatnonzero(mask)
    ax.scatter(found, data[found], s=18, marker="x", color="#c00000", label="виявлені АВ")
    ax.legend()
    save(fig, "06_real_detect.png")

    degree = stage_degree(cleaned, "курс USD", "07_real_degree.png")
    stage_learn(cleaned, degree, "курс USD", "08_real", ylabel="Курс, грн")


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"ЛР 2. Варіант 8, рівень складності III, обсяг вибірки {N}, seed {SEED}")

    clean_data, data, index = stage_input(rng)
    cleaned = stage_detect(clean_data, data, index)

    print("\n3. Вибір степеня моделі")
    degree = stage_degree(cleaned, "синтезована вибірка", "04_degree.png")

    print("\n4. Навчання та екстраполяція")
    stage_learn(cleaned, degree, "синтезована вибірка", "05", ideal=series.trend(N))

    stage_real()
    print(f"\nГрафіки збережено в {OUT}")


if __name__ == "__main__":
    main()
