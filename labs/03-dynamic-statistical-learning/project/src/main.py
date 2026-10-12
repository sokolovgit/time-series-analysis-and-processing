"""ЛР 3: статистичне навчання за динамічним Time Series. Варіант 8, групи вимог 1 і 2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import adaptive
import detect
import filters
import series

OUT = Path(__file__).resolve().parents[1] / "out"
RATES = Path(__file__).resolve().parents[2] / "docs/samples/Lab_work_3/data/Oschadbank (USD).xls"

N = 2000
SEED = 42
FRACTION = 0.1
RUNS = 100
FRAGMENT = 300

NAMES = {2: "α-β", 3: "α-β-γ"}
COLORS = {2: "#e08214", 3: "#0070c0"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def decorate(ax, title, ylabel="Значення"):
    ax.set_title(title)
    ax.set_xlabel("Номер відліку")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def errors(estimate, ideal, order):
    """Зміщення, СКВ і максимум похибки оцінки без перших order відліків розгону."""
    e = (estimate - ideal)[order:]
    return e.mean(), np.sqrt((e**2).mean()), np.abs(e).max()


def print_errors(rows):
    print(f"   {'фільтр':<28}{'зміщення':>10}{'СКВ':>9}{'max |e|':>10}")
    for name, (bias, rmse, worst) in rows.items():
        print(f"   {name:<28}{bias:>10.3f}{rmse:>9.3f}{worst:>10.3f}")


def confusion(mask, truth):
    real = np.zeros(len(mask), dtype=bool)
    real[truth] = True
    return int((mask & real).sum()), int((mask & ~real).sum()), int((~mask & real).sum())


def stage_input(rng, ideal, name, figure):
    clean_data = series.noisy(ideal, rng)
    data, index = series.add_anomalies(clean_data, rng, FRACTION)
    mask, window, k = detect.adaptive(data)
    tp, fp, fn = confusion(mask, index)
    print(f"   обсяг {N}, похибка N(0, {series.SIGMA}²), внесено {len(index)} АВ ({FRACTION:.0%})")
    print(f"   детектор ЛР 2: вікно {window}, поріг {k:.2f}σ")
    print(f"   виявлено {tp}, хибних {fp}, пропущено {fn}")
    cleaned = detect.clean(data, mask)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7))
    top.plot(data, lw=0.5, color="#9dc3e6", label="вибірка з АВ")
    top.plot(ideal, lw=2.0, color="#c00000", label="ідеальний тренд")
    decorate(top, f"Вхідні дані: {name}")
    head = np.arange(FRAGMENT)
    bottom.plot(head, data[head], lw=0.8, color="#9dc3e6", label="вибірка з АВ")
    bottom.plot(head, cleaned[head], lw=1.0, color="#2e7d32", label="після очищення")
    inside = index[index < FRAGMENT]
    bottom.scatter(inside, data[inside], s=22, color="#e08214", zorder=3, label="внесені АВ")
    decorate(bottom, f"Фрагмент: перші {FRAGMENT} відліків")
    save(fig, figure)
    return cleaned


def stage_verify(data):
    print("\n   Верифікація: фільтр зі зростаючою пам'яттю проти пакетного МНК")
    for order in (2, 3):
        estimate, _ = filters.run(data[:200], order)
        batch = [
            np.polyval(np.polyfit(np.arange(n), data[:n], order - 1), n - 1)
            for n in range(order + 1, 201)
        ]
        gap = np.max(np.abs(estimate[order:] - batch))
        print(f"   {NAMES[order]:<6} max |рекурентний − пакетний| = {gap:.1e}")


def stage_growing(data, ideal):
    print("\n2. Згладжування фільтрами зі зростаючою пам'яттю")
    estimates = {order: filters.run(data, order)[0] for order in (2, 3)}
    print_errors({NAMES[o]: errors(e, ideal, o) for o, e in estimates.items()})

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7))
    top.plot(data, lw=0.5, color="#9dc3e6", label="очищена вибірка")
    top.plot(ideal, lw=1.5, color="#7f7f7f", ls="--", label="ідеальний тренд")
    for order, estimate in estimates.items():
        top.plot(estimate, lw=1.5, color=COLORS[order], label=NAMES[order])
    decorate(top, "Оцінки фільтрів зі зростаючою пам'яттю")
    for order, estimate in estimates.items():
        bottom.plot(estimate - ideal, lw=0.8, color=COLORS[order], label=NAMES[order])
    bottom.axhline(0, color="#7f7f7f", lw=0.8)
    bottom.set_ylim(-60, 20)
    decorate(bottom, "Похибка оцінки відносно ідеального тренду", "Похибка")
    save(fig, "03_growing.png")


def stage_bias(rng, ideal):
    print(f"\n3. Перевірка незміщеності: {RUNS} реалізацій шуму, середнє похибки по ансамблю")
    ensemble = ideal[:, None] + rng.normal(0.0, series.SIGMA, (N, RUNS))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for order in (2, 3):
        estimate, _ = filters.run(ensemble, order)
        mean = (estimate - ideal[:, None]).mean(axis=1)
        tail = mean[N // 2 :]
        band = series.SIGMA / np.sqrt(RUNS)
        print(
            f"   {NAMES[order]:<6} середня похибка на другій половині {tail.mean():8.3f}, "
            f"масштаб випадкового розкиду ±{band:.2f}"
        )
        ax.plot(mean, lw=1.0, color=COLORS[order], label=NAMES[order])
    ax.axhline(0, color="#7f7f7f", lw=0.8)
    ax.set_ylim(-60, 10)
    decorate(ax, "Середня похибка оцінки по ансамблю реалізацій", "Зміщення")
    save(fig, "04_bias.png")


def stage_divergence(data, ideal):
    print("\n5. Розходження фільтрів зі зростаючою пам'яттю")
    t1, t2 = (int(N * b) for b in series.BREAKS)
    estimates = {order: filters.run(data, order)[0] for order in (2, 3)}
    print_errors({NAMES[o]: errors(e, ideal, o) for o, e in estimates.items()})

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(data, lw=0.5, color="#9dc3e6", label="очищена вибірка")
    ax.plot(ideal, lw=1.5, color="#7f7f7f", ls="--", label="ідеальний тренд")
    for order, estimate in estimates.items():
        ax.plot(estimate, lw=1.5, color=COLORS[order], label=NAMES[order])
    for t in (t1, t2):
        ax.axvline(t, color="#c00000", ls=":", lw=1.0)
    decorate(ax, "Розходження фільтрів зі зростаючою пам'яттю, злами тренду позначено")
    save(fig, "06_divergence.png")
    return estimates


def stage_memory(data, ideal):
    print("\n6. Модифікація 1: обмежена пам'ять N*, навчена за нев'язками прогнозу")
    learned = {}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for order in (2, 3):
        best, grid, rmse = adaptive.learn_memory(data, order)
        learned[order] = filters.run(data, order, best)[0]
        print(f"   {NAMES[order]:<6} N* = {best}, СКВ нев'язки {rmse.min():.3f}")
        ax.plot(grid, rmse, "o-", ms=3, color=COLORS[order], label=NAMES[order])
        ax.scatter([best], [rmse.min()], s=80, color="#c00000", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Довжина пам'яті N")
    ax.set_ylabel("СКВ нев'язки прогнозу")
    ax.set_title("Вибір пам'яті фільтра, N* позначено червоним")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save(fig, "07_memory.png")
    return learned


def stage_adaptive(data, ideal):
    print("\n7. Модифікація 3: власний адаптивний алгоритм (CUSUM + перезапуск пам'яті)")
    t1, t2 = (int(N * b) for b in series.BREAKS)
    runs = {order: adaptive.run(data, order) for order in (2, 3)}
    h = runs[3][4]
    print(f"   поріг CUSUM h = {h:.2f}, дрейф δ = {adaptive.DRIFT}, α = {adaptive.ALPHA}")
    for order, (_, _, _, alarms, _) in runs.items():
        shown = ", ".join(f"{k} (початок {s})" for k, s in alarms[:6])
        more = f" і ще {len(alarms) - 6}" if len(alarms) > 6 else ""
        print(f"   {NAMES[order]:<6} тривог {len(alarms)}: {shown}{more}")
    print(f"   справжні злами тренду: {t1}, {t2}")

    estimate, _, memory, alarms, _ = runs[3]
    fig, (top, mid, bottom) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    top.plot(data, lw=0.5, color="#9dc3e6", label="очищена вибірка")
    top.plot(ideal, lw=1.5, color="#7f7f7f", ls="--", label="ідеальний тренд")
    top.plot(estimate, lw=1.2, color=COLORS[3], label="адаптивний α-β-γ")
    decorate(top, "Адаптивний α-β-γ фільтр")
    mid.plot(filters.run(data, 3)[0] - ideal, lw=0.8, color="#7f7f7f", label="зростаюча пам'ять")
    mid.plot(estimate - ideal, lw=0.8, color=COLORS[3], label="адаптивний")
    mid.set_ylim(-40, 40)
    decorate(mid, "Похибка оцінки", "Похибка")
    bottom.plot(memory, lw=1.0, color=COLORS[3], label="пам'ять n")
    decorate(bottom, "Поточна пам'ять фільтра", "n")
    for ax in (top, mid, bottom):
        for t in (t1, t2):
            ax.axvline(t, color="#c00000", ls=":", lw=1.0)
        for k, _ in alarms:
            ax.axvline(k, color="#2e7d32", ls="--", lw=1.0)
    save(fig, "08_adaptive.png")
    return {order: run[0] for order, run in runs.items()}


def stage_compare(ideal, growing, learned, tuned):
    print("\n8. Порівняння на кусковому тренді (модифікація 2 – перехід α-β → α-β-γ)")
    rows = {}
    for order in (2, 3):
        rows[f"{NAMES[order]} зростаюча пам'ять"] = errors(growing[order], ideal, order)
        rows[f"{NAMES[order]} пам'ять N*"] = errors(learned[order], ideal, order)
        rows[f"{NAMES[order]} адаптивний"] = errors(tuned[order], ideal, order)
    print_errors(rows)


def stage_monte_carlo(rng):
    print(f"\n9. Монте-Карло, {RUNS} реалізацій, адаптивний α-β-γ")
    quad, piece = series.quadratic(N), series.piecewise(N)
    t1, t2 = (int(N * b) for b in series.BREAKS)
    false_runs = 0
    delays = {t1: [], t2: []}
    misses = 0
    piece_errors = []
    for _ in range(RUNS):
        false_runs += bool(adaptive.run(series.noisy(quad, rng), 3)[3])
        estimate, _, _, alarms, _ = adaptive.run(series.noisy(piece, rng), 3)
        piece_errors.append(estimate - piece)
        for t in delays:
            after = [k - t for k, _ in alarms if k >= t]
            if after and after[0] < t1:
                delays[t].append(after[0])
            else:
                misses += 1
    print(
        f"   квадратичний тренд (злам відсутній): реалізацій з хибною тривогою "
        f"{false_runs}/{RUNS}, очікується близько α = {adaptive.ALPHA:.0%}"
    )
    for t, d in delays.items():
        print(f"   злам у {t}: виявлено {len(d)}/{RUNS}, затримка медіана {np.median(d):.0f}")
    mean = np.mean(piece_errors, axis=0)
    print(f"   середня по ансамблю похибка на кусковому тренді {mean[3:].mean():.3f}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(mean, lw=1.0, color=COLORS[3], label="адаптивний α-β-γ")
    ax.axhline(0, color="#7f7f7f", lw=0.8)
    for t in (t1, t2):
        ax.axvline(t, color="#c00000", ls=":", lw=1.0)
    ax.set_ylim(-10, 10)
    decorate(ax, "Середня похибка адаптивного фільтра по ансамблю, кусковий тренд", "Зміщення")
    save(fig, "09_adaptive_bias.png")


def stage_real():
    print("\n10. Реальні дані: курс USD, Ощадбанк")
    raw, gaps = series.real_rates(RATES)
    mask, window, k = detect.adaptive(raw)
    data = detect.clean(raw, mask)
    print(f"   {len(raw)} відліків, пропусків {gaps}")
    print(f"   вікно {window}, поріг {k:.2f}σ, виявлено {int(mask.sum())} аномальних вимірів")

    results = {}
    for order in (2, 3):
        results[f"{NAMES[order]} зростаюча пам'ять"] = filters.run(data, order)
        best, _, _ = adaptive.learn_memory(data, order)
        results[f"{NAMES[order]} пам'ять N* = {best}"] = filters.run(data, order, best)
        estimate, innovation, memory, alarms, _ = adaptive.run(data, order)
        results[f"{NAMES[order]} адаптивний"] = (estimate, innovation)
        print(
            f"   {NAMES[order]:<6} адаптивний: тривог {len(alarms)}, "
            f"середня пам'ять {memory.mean():.1f}"
        )
    print(f"   {'фільтр':<28}{'зміщення':>10}{'СКВ':>9}")
    for name, (_, innovation) in results.items():
        r = innovation[~np.isnan(innovation)]
        print(f"   {name:<28}{r.mean():>10.3f}{np.sqrt((r**2).mean()):>9.3f}")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8))
    top.plot(raw, lw=1.0, color="#9dc3e6", label="курс купівлі")
    found = np.flatnonzero(mask)
    top.scatter(found, raw[found], s=22, marker="x", color="#c00000", zorder=3, label="виявлені АВ")
    for name, color in (("α-β-γ зростаюча пам'ять", "#7f7f7f"), ("α-β-γ адаптивний", COLORS[3])):
        top.plot(results[name][0], lw=1.3, color=color, label=name)
    decorate(top, "Курс USD: згладжування α-β-γ фільтрами", "Курс, грн")
    for name, color in (("α-β-γ зростаюча пам'ять", "#7f7f7f"), ("α-β-γ адаптивний", COLORS[3])):
        bottom.plot(results[name][1], lw=0.8, color=color, label=name)
    bottom.axhline(0, color="#7f7f7f", lw=0.8)
    decorate(bottom, "Нев'язка прогнозу на крок", "Нев'язка, грн")
    save(fig, "10_real.png")


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"ЛР 3. Варіант 8, групи вимог 1 і 2, обсяг вибірки {N}, seed {SEED}")

    print("\n1. Квадратичний тренд: синтез і очищення від аномальних вимірів")
    quad = series.quadratic(N)
    data = stage_input(rng, quad, "квадратичний тренд", "01_input.png")
    stage_verify(data)
    stage_growing(data, quad)
    stage_bias(rng, quad)

    print("\n4. Кусковий тренд (лінійний – квадратичний – лінійний): синтез і очищення")
    piece = series.piecewise(N)
    data = stage_input(rng, piece, "кусковий тренд", "05_input.png")
    growing = stage_divergence(data, piece)
    learned = stage_memory(data, piece)
    tuned = stage_adaptive(data, piece)
    stage_compare(piece, growing, learned, tuned)
    stage_monte_carlo(rng)

    stage_real()
    print(f"\nГрафіки збережено в {OUT}")


if __name__ == "__main__":
    main()
