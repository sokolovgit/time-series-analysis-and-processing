"""МКР, варіант 8: модель і прогноз часового ряду викидів CO₂ методом найменших квадратів."""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

import evaluate
import loaders
import lsm
import plots

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "out"
DATA = PROJECT / "data/co2.csv"

HORIZON = 10
START = 25
DM_HORIZONS = (1, 3, 5, 8, 10)
SHOW_ORIGINS = (25, 35, 45)
REFINED = "{} + AR(1)"


def section(title):
    print(f"\n{title}\n" + "-" * len(title))


def table(curves):
    header = "".join(f"{h:>7}" for h in range(1, HORIZON + 1))
    print(f"MAPE, % за горизонтом h:\n{'':<26}{header}{'сер.':>8}")
    for name, mape in curves.items():
        print(f"{name:<26}" + "".join(f"{v:>7.2f}" for v in mape) + f"{mape.mean():>8.2f}")


def stage_data(refresh):
    section("1. Вхідні дані та статистичні характеристики")
    years, y, source = loaders.emissions(DATA, refresh)
    print(f"Джерело: {loaders.SOURCE_URL} ({source})")
    print(f"{len(y)} річних значень за {years[0]}–{years[-1]}, млрд т CO₂")
    growth = 100 * np.diff(y) / y[:-1]
    print(
        f"  середнє {y.mean():.3f}, СКВ {y.std(ddof=1):.3f}, мін {y.min():.3f}, макс {y.max():.3f}"
    )
    print(f"  асиметрія {stats.skew(y):.3f}, ексцес {stats.kurtosis(y):.3f}")
    print(f"  середній приріст {growth.mean():.2f}% на рік, СКВ приросту {growth.std(ddof=1):.2f}%")
    falls = years[1:][growth < 0]
    print(f"  роки зі спадом: {', '.join(map(str, falls))}")
    print(f"  найбільший спад {growth.min():.2f}% у {years[1:][growth.argmin()]}")
    plots.series(OUT, years, y)
    return years, y


def stage_candidates(years, y):
    section("2. Моделі тренду за МНК на всьому інтервалі спостереження")
    print(
        f"{'модель':<18}{'k':>3}{'R²':>9}{'R²adj':>9}{'RMSE':>8}{'MAPE,%':>8}"
        f"{'AIC':>9}{'BIC':>9}{'cond':>10}"
    )
    fits = []
    for name in lsm.MODELS:
        state = lsm.fit(y, name)
        k = len(state["coeffs"])
        q = evaluate.metrics(y, state["fitted"]) | evaluate.criteria(y, state["fitted"], k)
        print(
            f"{name:<18}{k:>3}{q['R2']:>9.4f}{q['R2adj']:>9.4f}{q['RMSE']:>8.3f}{q['MAPE']:>8.2f}"
            f"{q['AIC']:>9.1f}{q['BIC']:>9.1f}{state['cond']:>10.1e}"
        )
        fits.append((name, state, lsm.predict(state, HORIZON)[0]))
    for name, state, forecast in fits:
        print(f"  {name}: {lsm.formula(state)}; {years[-1] + HORIZON}: {forecast[-1]:.2f}")
    plots.candidates(OUT, years, y, fits, HORIZON)


def stage_backtest(years, y):
    section("3. Вибір моделі тренду за точністю прогнозу поза вибіркою")
    print(
        f"Rolling origin: навчання на {years[0]}…t−1, прогноз на 1…{HORIZON} років,"
        f" t = {years[START]}…{years[-1]} ({len(y) - START} точок старту)"
    )
    errors = {
        name: evaluate.backtest(y, lsm.forecaster(name), START, HORIZON) for name in lsm.MODELS
    }
    curves = {name: evaluate.mape_by_horizon(e) for name, e in errors.items()}
    table(curves)
    best = min(lsm.MODELS, key=lambda name: curves[name].mean())
    print(f"Найменша середня MAPE: {best}")
    return best, errors


def stage_refine(years, y, best, errors):
    section(f"4. Аналіз залишків тренду «{best}» і уточнення моделі")
    state = lsm.fit_ar(y, best)
    se, t, p = lsm.significance(state)
    print(lsm.formula(state) + f", τ = (рік − {years[0]}) / {len(y)}")
    for i, (c, s, tv, pv) in enumerate(zip(state["coeffs"], se, t, p, strict=True)):
        print(f"  c{i} = {c:+.4f}, se = {s:.4f}, t = {tv:+.2f}, p = {pv:.1e}")

    tests = {}
    for label, e, k in (("eₜ тренду", state["e"], 0), ("uₜ після AR(1)", state["u"], 1)):
        r = evaluate.residual_tests(e, k)
        tests[label] = r
        print(f"  {label}:")
        print(f"    середнє {e.mean():+.2e} (t-тест p = {r['mean_p']:.3f})")
        print(f"    Шапіро–Вілк p = {r['shapiro_p']:.3f}")
        print(f"    Дарбін–Уотсон DW = {r['dw']:.3f}")
        print(f"    Льюнг–Бокс Q({evaluate.LAGS}) = {r['ljung_q']:.2f}, p = {r['ljung_p']:.4f}")
        print("    ACF: " + " ".join(f"{v:+.2f}" for v in r["acf"]))
    print(f"  AR(1) за МНК: φ = {state['phi']:.4f}, σᵤ = {np.sqrt(state['sigma2_u']):.4f}")
    plots.residuals(
        OUT,
        years,
        state["e"],
        state["u"],
        tests["eₜ тренду"]["acf"],
        tests["uₜ після AR(1)"]["acf"],
    )

    name = REFINED.format(best)
    errors = errors | {name: evaluate.backtest(y, lsm.forecaster(best, ar=True), START, HORIZON)}
    errors |= {n: evaluate.backtest(y, f, START, HORIZON) for n, f in evaluate.BASELINES.items()}
    return name, state, errors


def stage_effectiveness(years, y, best, refined, errors):
    section("5. Ефективність: МНК проти базових прогнозів")
    curves = {name: evaluate.mape_by_horizon(e) for name, e in errors.items()}
    table(curves)
    for model in (best, refined):
        for name in evaluate.BASELINES:
            gain = 100 * (1 - curves[model] / curves[name])
            better = [h + 1 for h in np.flatnonzero(gain > 0)]
            where = ", ".join(map(str, better)) if better else "–"
            print(
                f"  {model} проти «{name}»: зниження MAPE {gain.mean():+.1f}% у середньому,"
                f" точніший при h = {where}"
            )
    print("Тест Дібольда–Маріано (H0: однакова точність; DM < 0 – МНК точніший):")
    for rival in ("наївний", "дрейф"):
        for h in DM_HORIZONS:
            dm, p, n = evaluate.diebold_mariano(
                errors[refined][:, h - 1], errors[rival][:, h - 1], h
            )
            verdict = (
                "МНК точніший"
                if p < 0.05
                else (f"«{rival}» точніший" if 1 - p < 0.05 else "різниця незначуща")
            )
            print(
                f"  проти «{rival}», h = {h:>2}: DM = {dm:+.3f}, p = {p:.4f}, n = {n} – {verdict}"
            )
    plots.backtest(OUT, curves, refined)
    plots.origins(
        OUT,
        years,
        y,
        SHOW_ORIGINS,
        HORIZON,
        lsm.forecaster(best, ar=True),
        refined,
        evaluate.naive,
    )


def stage_forecast(years, y, state, errors, refined):
    section(f"6. Прогноз поза інтервалом спостереження на {HORIZON} років")
    mean, low, high = lsm.predict_ar(state, HORIZON)
    trend = lsm.predict(state, HORIZON)[0]
    r_low, r_high = evaluate.empirical_band(errors[refined])
    e_low, e_high = mean / (1 + r_high), mean / (1 + r_low)
    future = np.arange(years[-1] + 1, years[-1] + HORIZON + 1)
    print(f"{'рік':>6}{'тренд':>9}{'прогноз':>10}{'95% модель':>18}{'95% емпіричний':>20}")
    for row in zip(future, trend, mean, low, high, e_low, e_high, strict=True):
        band = f"{row[3]:.2f}…{row[4]:.2f}"
        empirical = f"{row[5]:.2f}…{row[6]:.2f}"
        print(f"{row[0]:>6}{row[1]:>9.2f}{row[2]:>10.2f}{band:>18}{empirical:>20}")
    plots.forecast(
        OUT, years, y, state["fitted"], future, mean, (low, high), (e_low, e_high), refined
    )


def stage_verify(y):
    section("7. Верифікація реалізації МНК")
    n = len(y)
    tau = np.arange(n) / n
    for name, (kind, degree) in lsm.MODELS.items():
        state = lsm.fit(y, name)
        target = np.log(y) if kind == "exp" else y
        reference = np.polyfit(tau, target, degree)[::-1]
        lstsq = np.linalg.lstsq(lsm.design(tau, degree), target, rcond=None)[0]
        print(
            f"  {name:<18} |C − polyfit| = {np.abs(state['coeffs'] - reference).max():.1e},"
            f" |C − lstsq| = {np.abs(state['coeffs'] - lstsq).max():.1e}"
        )
    state = lsm.fit_ar(y, "експоненційна")
    e = state["e"]
    reference = np.linalg.lstsq(e[:-1, None], e[1:], rcond=None)[0][0]
    print(f"  φ AR(1)            |φ − lstsq| = {abs(state['phi'] - reference):.1e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="заново отримати дані з сайту")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    plots.setup()

    years, y = stage_data(args.refresh)
    stage_candidates(years, y)
    best, errors = stage_backtest(years, y)
    refined, state, errors = stage_refine(years, y, best, errors)
    stage_effectiveness(years, y, best, refined, errors)
    stage_forecast(years, y, state, errors, refined)
    stage_verify(y)
    print(f"\nГрафіки збережено в {OUT}")


if __name__ == "__main__":
    main()
