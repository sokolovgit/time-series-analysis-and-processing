"""Група 2: погода 12 міст України – декомпозиція, властивості, кореляції, кластери, синтез."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram

import analysis
import loaders
import patterns
import plots
import synthesis

CITY = "Київ"
WEST, EAST = "Львів", "Харків"
DECADE = 3652.5
LAGS = 40
MAX_LAG = 5
K_CITIES = 4
RUNS = 100
WINTER, SUMMER = (12, 1, 2), (6, 7, 8)
HOURS_PER_STEP = 3
MSTL_WINDOWS = (11, 11)
AMPLITUDE_OFFSETS = {"Полтава": (-40, 3), "Запоріжжя": (3, -9), "Львів": (-22, -10)}
TREND_OFFSETS = {"Запоріжжя": (3, -9)}
DAYS = 365
METHODS = {
    "gauss": "VAR + нормальні інновації",
    "bootstrap": "VAR + сезонний бутстреп",
    "copula": "копула + бутстреп",
}
MAIN_METHOD = "bootstrap"

LABELS = {
    "temp": "температура, °C",
    "humidity": "вологість, %",
    "pressure": "тиск, гПа",
    "wind": "вітер, км/год",
    "precip": "опади, мм",
    "radiation": "радіація, МДж/м²",
}
SHORT = {
    "temp": "T",
    "humidity": "вологість",
    "pressure": "тиск",
    "wind": "вітер",
    "precip": "опади",
    "radiation": "радіація",
}
CLUSTERS = ["захід", "центр і північ", "південь", "схід"]
CITY_METRICS = [
    "m",
    "σ",
    "b",
    "A",
    "σ зима/літо",
    "acf1",
    "acf7",
    "H",
    "асиметрія",
    "асиметрія зима",
    "ексцес",
]
GLOBAL_METRICS = ["r середня", "r Л→Х +1", "ARI кластерів"]
SHOWN = [("асиметрія зима", CITY), ("r Л→Х +1", None), ("H", CITY)]


def season(index, months):
    return np.isin(index.month, months)


def climate_features(frame):
    """Ознаки клімату міста: середні показники, річна амплітуда температури, річна сума опадів."""
    rows = {}
    for city, part in frame.groupby("city"):
        part = part.set_index("date")
        beta, _ = analysis.seasonal_fit(part[["temp"]], 1)
        rows[city] = {
            "T, °C": part["temp"].mean(),
            "амплітуда, °C": np.hypot(beta[2, 0], beta[3, 0]),
            "вологість, %": part["humidity"].mean(),
            "опади, мм/рік": part["precip"].sum() / (len(part) / analysis.YEAR),
            "вітер, км/год": part["wind"].mean(),
            "радіація, МДж/м²": part["radiation"].mean(),
        }
    return pd.DataFrame(rows).T.loc[list(loaders.CITIES)]


def correlation_clusters(anomalies, k=K_CITIES):
    labels = patterns.cut(patterns.correlation_linkage(anomalies), k)
    return patterns.relabel(labels, [loaders.CITIES[c][1] for c in anomalies.columns])


def metrics(frame, reference_labels):
    """Показники температурного поля: для кожного міста і спільні для всієї мережі."""
    beta, resid = analysis.seasonal_fit(frame)
    winter = season(frame.index, WINTER)
    out = {}
    for j, city in enumerate(frame.columns):
        r = resid[city]
        monthly = r.groupby(frame.index.month).std()
        values = {
            "m": frame[city].mean(),
            "σ": frame[city].std(),
            "b": beta[1, j] * DECADE,
            "A": np.hypot(beta[2, j], beta[3, j]),
            "σ зима/літо": monthly[list(WINTER)].mean() / monthly[list(SUMMER)].mean(),
            "acf1": r.autocorr(1),
            "acf7": r.autocorr(7),
            "H": analysis.hurst(r.to_numpy())[0],
            "асиметрія": stats.skew(r),
            "асиметрія зима": stats.skew(r[winter]),
            "ексцес": stats.kurtosis(r),
        }
        out |= {f"{city}|{key}": v for key, v in values.items()}
    corr = resid.corr().values
    out["r середня"] = corr[np.triu_indices_from(corr, 1)].mean()
    out["r Л→Х +1"] = resid[WEST].corr(resid[EAST].shift(-1))
    out["ARI кластерів"] = patterns.agreement(reference_labels, correlation_clusters(resid))
    return out


def stage_input(frame, hourly, out):
    cities = frame["city"].nunique()
    days = frame["date"].nunique()
    print(
        f"   {cities} міст × {days} днів "
        f"({frame['date'].min():%d.%m.%Y} – {frame['date'].max():%d.%m.%Y}), "
        f"{len(loaders.VARIABLES)} показників"
    )
    print(
        f"   погодинна температура ({CITY}): {len(hourly)} відліків, "
        f"для MSTL усереднена до {HOURS_PER_STEP} год"
    )
    kyiv = frame[frame["city"] == CITY].set_index("date")
    rows = {SHORT[v]: analysis.describe(kyiv[v]) for v in LABELS}
    print(f"   {CITY}, добові значення:")
    print(analysis.table(rows).round(2).to_string())
    print("   середні по містах:")
    print(frame.groupby("city")[list(LABELS)].mean().loc[list(loaders.CITIES)].round(2).to_string())

    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True)
    for ax, (v, label), color in zip(axes.flat, LABELS.items(), plots.COLORS, strict=False):
        ax.plot(kyiv.index, kyiv[v], lw=0.4, color=color, alpha=0.8)
        ax.plot(kyiv.index, kyiv[v].rolling(30, center=True).mean(), lw=1.2, color=plots.INK)
        ax.set_title(label, fontsize=9, loc="left")
    fig.suptitle(f"{CITY}: добові показники 2015–2024 і ковзне середнє за 30 днів", fontsize=11)
    plots.save(fig, out, "08_weather_input.png")

    temp = kyiv["temp"]
    years = sorted(temp.index.year.unique())
    grid = np.full((len(years), 366), np.nan)
    for i, year in enumerate(years):
        part = temp[temp.index.year == year]
        grid[i, part.index.dayofyear - 1] = part.values
    fig, ax = plt.subplots(figsize=(11, 3.6))
    image = ax.imshow(
        grid,
        aspect="auto",
        cmap=plots.DIVERGING,
        norm=TwoSlopeNorm(0, -20, 30),
        interpolation="nearest",
    )
    ax.set_yticks(range(len(years)), years)
    starts = pd.date_range("2021-01-01", periods=12, freq="MS")
    ax.set_xticks(
        [day.dayofyear - 1 for day in starts],
        ["січ", "лют", "бер", "кві", "тра", "чер", "лип", "сер", "вер", "жов", "лис", "гру"],
    )
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="°C", pad=0.01)
    ax.set_title(f"{CITY}: середньодобова температура, рік × день року")
    plots.save(fig, out, "09_weather_heatmap.png")


def stage_decomposition(frame, hourly, out):
    kyiv = loaders.wide(frame, "temp")[CITY]
    res = analysis.decompose(kyiv, DAYS)
    f_trend, f_season = analysis.strength(res.trend.values, res.seasonal.values, res.resid.values)
    print(
        f"   STL ({CITY}, T, період 365): F тренду = {f_trend:.3f}, F сезонності = {f_season:.3f}, "
        f"σ залишку = {res.resid.std():.2f} °C"
    )
    print(
        f"   тренд STL: {res.trend.iloc[:DAYS].mean():.2f} °C у 2015 → "
        f"{res.trend.iloc[-DAYS:].mean():.2f} °C у 2024"
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
    parts = [
        (res.observed, "Ряд"),
        (res.trend, "Тренд"),
        (res.seasonal, "Сезонна компонента"),
        (res.resid, "Залишок"),
    ]
    for ax, (values, title) in zip(axes, parts, strict=True):
        ax.plot(
            values.index,
            values.values,
            color=plots.COLORS[0],
            lw=0.7 if title in ("Ряд", "Залишок") else 1.2,
        )
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_title(f"STL-декомпозиція температури, {CITY}", loc="center")
    plots.save(fig, out, "10_weather_stl.png")

    steps = hourly.resample(f"{HOURS_PER_STEP}h").mean()
    day = 24 // HOURS_PER_STEP
    year = round(analysis.YEAR * day)
    multi = analysis.decompose_multi(steps, (day, year), MSTL_WINDOWS)
    seasonal = multi.seasonal
    print(f"   MSTL ({CITY}, крок {HOURS_PER_STEP} год, періоди {day} і {year}):")
    print(
        f"     σ добової сезонності = {seasonal.iloc[:, 0].std():.2f} °C, "
        f"σ річної = {seasonal.iloc[:, 1].std():.2f} °C, σ залишку = {multi.resid.std():.2f} °C"
    )
    by_month = (
        seasonal.iloc[:, 0].groupby(seasonal.index.month).max()
        - seasonal.iloc[:, 0].groupby(seasonal.index.month).min()
    )
    print(
        f"     розмах добового циклу: {by_month.min():.1f} °C (міс. {by_month.idxmin()}) – "
        f"{by_month.max():.1f} °C (міс. {by_month.idxmax()})"
    )

    fig, axes = plt.subplots(5, 1, figsize=(10, 9))
    axes[0].plot(steps.index, steps.values, lw=0.4, color=plots.COLORS[0])
    axes[0].set_title(f"Температура з кроком {HOURS_PER_STEP} год", fontsize=9, loc="left")
    axes[1].plot(multi.trend.index, multi.trend.values, color=plots.COLORS[0])
    axes[1].set_title("Тренд", fontsize=9, loc="left")
    axes[2].plot(seasonal.index, seasonal.iloc[:, 1].values, color=plots.COLORS[1])
    axes[2].set_title("Річна сезонність", fontsize=9, loc="left")
    axes[3].plot(seasonal.index, seasonal.iloc[:, 0].values, lw=0.3, color=plots.COLORS[2])
    axes[3].set_title(
        "Добова сезонність (амплітуда змінюється протягом року)", fontsize=9, loc="left"
    )
    week = seasonal.loc["2024-07-01":"2024-07-07"].iloc[:, 0]
    winter_week = seasonal.loc["2024-01-08":"2024-01-14"].iloc[:, 0]
    axes[4].plot(
        np.arange(len(week)) * HOURS_PER_STEP,
        week.values,
        marker="o",
        ms=3,
        color=plots.COLORS[1],
        label="липень",
    )
    axes[4].plot(
        np.arange(len(winter_week)) * HOURS_PER_STEP,
        winter_week.values,
        marker="o",
        ms=3,
        color=plots.COLORS[0],
        label="січень",
    )
    axes[4].set_title("Добова сезонність, тиждень", fontsize=9, loc="left")
    axes[4].set_xlabel("години від початку тижня")
    axes[4].legend()
    fig.suptitle(f"MSTL-декомпозиція погодинної температури, {CITY} 2015–2024", fontsize=11)
    plots.save(fig, out, "11_weather_mstl.png")

    seasonality = pd.DataFrame(
        {v: analysis.harmonic_strength(loaders.wide(frame, v)) for v in LABELS}
    )
    print("   сила річної сезонності F_S (3 гармоніки) за містами і показниками:")
    print(seasonality.rename(columns=SHORT).round(2).to_string())
    return seasonality


def stage_gradients(frame, seasonality, out):
    temp = loaders.wide(frame, "temp")
    beta, _ = analysis.seasonal_fit(temp)
    lon = np.array([loaders.CITIES[c][1] for c in temp.columns])
    trend = beta[1] * DECADE
    amplitude = np.hypot(beta[2], beta[3])
    r_amp = stats.pearsonr(lon, amplitude)
    r_trend = stats.pearsonr(lon, trend)
    print("   тренд і річна амплітуда температури (МНК, лінійний тренд + 3 гармоніки):")
    for city, b, a in zip(temp.columns, trend, amplitude, strict=True):
        print(f"     {city:<17} тренд {b:+.2f} °C/10 років   амплітуда {a:5.2f} °C")
    print(
        f"   кореляція з довготою: амплітуда r = {r_amp.statistic:.2f} (p = {r_amp.pvalue:.3f}), "
        f"тренд r = {r_trend.statistic:.2f} (p = {r_trend.pvalue:.3f})"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), width_ratios=(1, 1, 1.3))
    for ax, values, title, offsets in [
        (axes[0], amplitude, "Річна амплітуда, °C", AMPLITUDE_OFFSETS),
        (axes[1], trend, "Тренд, °C за 10 років", TREND_OFFSETS),
    ]:
        ax.scatter(lon, values, s=36, color=plots.COLORS[0], zorder=3)
        fit = np.polyfit(lon, values, 1)
        ax.plot(
            lon[[lon.argmin(), lon.argmax()]],
            np.polyval(fit, lon[[lon.argmin(), lon.argmax()]]),
            color=plots.MUTED,
            ls="--",
        )
        for x, y, city in zip(lon, values, temp.columns, strict=True):
            ax.annotate(
                city,
                (x, y),
                fontsize=6.5,
                color=plots.INK_SECONDARY,
                xytext=offsets.get(city, (3, 3)),
                textcoords="offset points",
            )
        ax.set_xlabel("довгота, °E")
        ax.set_title(title)
    image = plots.heatmap(
        axes[2],
        seasonality.values,
        list(seasonality.index),
        [SHORT[v] for v in seasonality.columns],
        cmap=plots.SEQUENTIAL,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Сила річної сезонності F_S")
    fig.colorbar(image, ax=axes[2], pad=0.01)
    plots.save(fig, out, "12_weather_gradients.png")
    return trend, amplitude


def stage_properties(frame, out):
    temp = loaders.wide(frame, "temp")
    _, resid = analysis.seasonal_fit(temp)
    kyiv = temp[CITY]
    rows = {"T": analysis.stationarity(kyiv), "аномалія T": analysis.stationarity(resid[CITY])}
    print(f"   стаціонарність ({CITY}):")
    print(analysis.table(rows).to_string(float_format=lambda v: f"{v:.4f}"))
    props = {}
    winter = season(temp.index, WINTER)
    for city in temp.columns:
        r = resid[city].to_numpy()
        props[city] = {
            "σ": r.std(),
            "асиметрія": stats.skew(r),
            "асиметрія зима": stats.skew(r[winter]),
            "ексцес": stats.kurtosis(r),
            "acf1": resid[city].autocorr(1),
            "H": analysis.hurst(r)[0],
            "стаціонарна": analysis.stationarity(r)["стаціонарний"],
        }
    print("   властивості температурних аномалій за містами:")
    print(analysis.table(props).to_string(float_format=lambda v: f"{v:.3f}"))

    r = resid[CITY]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    bins = np.linspace(-20, 15, 50)
    axes[0].hist(
        r[winter],
        bins=bins,
        density=True,
        alpha=0.6,
        color=plots.COLORS[0],
        label=f"зима, γ = {stats.skew(r[winter]):.2f}",
    )
    summer = season(temp.index, SUMMER)
    axes[0].hist(
        r[summer],
        bins=bins,
        density=True,
        alpha=0.6,
        color=plots.COLORS[1],
        label=f"літо, γ = {stats.skew(r[summer]):.2f}",
    )
    axes[0].set_title("Аномалії температури")
    axes[0].set_xlabel("°C")
    axes[0].legend()
    lags = np.arange(LAGS + 1)
    axes[1].plot(
        lags,
        analysis.autocorrelation(r, LAGS),
        marker="o",
        ms=3,
        color=plots.COLORS[0],
        label="аномалія",
    )
    axes[1].plot(lags, 0.81**lags, color=plots.MUTED, ls="--", label="AR(1), φ = 0.81")
    axes[1].set_title("Автокореляція аномалій")
    axes[1].set_xlabel("лаг, днів")
    axes[1].legend()
    h, sizes, rs = analysis.hurst(r.to_numpy())
    axes[2].loglog(sizes, rs, "o", ms=5, color=plots.COLORS[0], label=f"аномалія, H = {h:.2f}")
    axes[2].loglog(
        sizes,
        [analysis.expected_rs(n) for n in sizes],
        color=plots.MUTED,
        label="E[R/S] білого шуму",
    )
    axes[2].set_title("R/S-аналіз")
    axes[2].set_xlabel("довжина вікна n, днів")
    axes[2].legend()
    plots.save(fig, out, "13_weather_properties.png")
    return resid


def stage_correlation(frame, resid, out):
    kyiv = frame[frame["city"] == CITY].set_index("date")[list(LABELS)].asfreq("D")
    _, anomalies = analysis.seasonal_fit(kyiv)
    raw = kyiv.corr()
    anom = anomalies.corr()
    names = [SHORT[v] for v in LABELS]
    print(f"   кореляція показників ({CITY}), сирі ряди → аномалії:")
    for a, b in itertools.combinations(LABELS, 2):
        if abs(raw.loc[a, b]) > 0.3 or abs(anom.loc[a, b]) > 0.3:
            print(
                f"     {SHORT[a]:>9} – {SHORT[b]:<9} {raw.loc[a, b]:+.2f} → {anom.loc[a, b]:+.2f}"
            )

    cities = list(resid.columns)
    pairs = list(itertools.combinations(cities, 2))
    dist = np.array([patterns.distance_km(loaders.CITIES[a], loaders.CITIES[b]) for a, b in pairs])
    corr = np.array([resid[a].corr(resid[b]) for a, b in pairs])
    scale = -np.sum(dist**2) / np.sum(dist * np.log(corr))
    print(
        f"   кореляція аномалій T між містами: {corr.min():.2f} – {corr.max():.2f}, "
        f"r(d) ≈ exp(−d/L), L = {scale:.0f} км"
    )

    asym, dlon = [], []
    for a, b in pairs:
        west, east = sorted((a, b), key=lambda c: loaders.CITIES[c][1])
        lags, r = patterns.cross_correlation(resid[west], resid[east], MAX_LAG)
        asym.append(r[lags == 1][0] - r[lags == -1][0])
        dlon.append(loaders.CITIES[east][1] - loaders.CITIES[west][1])
    asym, dlon = np.array(asym), np.array(dlon)
    fit = stats.linregress(dlon, asym)
    print(
        f"   асиметрія r(+1) − r(−1) (захід → схід) проти Δдовготи: "
        f"нахил {fit.slope:.4f} на градус, "
        f"r = {fit.rvalue:.2f}, p = {fit.pvalue:.1e}"
    )
    lead = {}
    for west, east in [("Львів", "Київ"), (WEST, EAST), ("Ужгород", EAST), ("Одеса", "Чернігів")]:
        lag, r = patterns.peak_lag(resid[west], resid[east], MAX_LAG)
        lead[(west, east)] = (lag, r)
        print(f"     {west} → {east}: пік r = {r:.3f} на лагу {lag:+d} дн.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plots.heatmap(axes[0], raw, names, names)
    axes[0].set_title("Сирі ряди")
    plots.heatmap(axes[1], anom, names, names)
    axes[1].set_title("Аномалії (без сезонності)")
    fig.suptitle(f"Кореляція показників, {CITY}", fontsize=11)
    plots.save(fig, out, "14_weather_variables.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(dist, corr, s=18, color=plots.COLORS[0])
    d = np.linspace(0, dist.max(), 100)
    axes[0].plot(
        d, np.exp(-d / scale), color=plots.MUTED, ls="--", label=f"exp(−d/L), L = {scale:.0f} км"
    )
    axes[0].set_xlabel("відстань, км")
    axes[0].set_ylabel("r аномалій")
    axes[0].set_title("Кореляція і відстань")
    axes[0].legend()
    for (west, east), color in zip(lead, plots.COLORS, strict=False):
        lags, r = patterns.cross_correlation(resid[west], resid[east], MAX_LAG)
        axes[1].plot(lags, r, marker="o", ms=3, color=color, label=f"{west} → {east}")
    axes[1].axvline(0, color=plots.AXIS, lw=0.8)
    axes[1].set_xlabel("лаг τ, днів")
    axes[1].set_title("Крос-кореляція r(τ)")
    axes[1].legend(fontsize=7)
    axes[2].scatter(dlon, asym, s=18, color=plots.COLORS[0])
    x = np.array([0, dlon.max()])
    axes[2].plot(x, fit.intercept + fit.slope * x, color=plots.MUTED, ls="--")
    axes[2].axhline(0, color=plots.AXIS, lw=0.8)
    axes[2].set_xlabel("різниця довгот, °")
    axes[2].set_ylabel("r(+1) − r(−1)")
    axes[2].set_title("Перенесення із заходу на схід")
    plots.save(fig, out, "15_weather_spatial.png")
    return scale, fit, lead


def stage_clustering(features, resid, out):
    print("   кліматичні ознаки міст:")
    print(features.round(2).to_string())
    k, _, scores = patterns.best_kmeans(features)
    print(
        "   силует K-means: "
        + ", ".join(f"k={key}: {v:.3f}" for key, v in scores.items())
        + f" → k = {k}"
    )
    lon = [loaders.CITIES[c][1] for c in features.index]
    tree = patterns.ward(features)
    by_climate = patterns.relabel(patterns.cut(tree, K_CITIES), lon)
    by_dynamics = correlation_clusters(resid)
    ari = patterns.agreement(by_climate, by_dynamics)
    print(f"   Уорд за кліматом, k = {K_CITIES}:")
    for i, name in enumerate(CLUSTERS):
        print(f"     {name:<15} {', '.join(features.index[by_climate == i])}")
    print(f"   кластеризація за кореляцією аномалій, k = {K_CITIES}: ARI з кліматичною = {ari:.3f}")

    fig = plt.figure(figsize=(13, 4.2))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.1, 1.1, 1))
    for i, (link, title) in enumerate(
        [
            (tree, "Уорд за кліматичними ознаками"),
            (patterns.correlation_linkage(resid), "За кореляцією аномалій"),
        ]
    ):
        ax = fig.add_subplot(grid[i])
        dendrogram(
            link,
            labels=list(features.index),
            ax=ax,
            color_threshold=0,
            above_threshold_color=plots.MUTED,
            leaf_rotation=60,
            leaf_font_size=7,
        )
        for label in ax.get_xticklabels():
            label.set_color(plots.COLORS[by_climate[list(features.index).index(label.get_text())]])
        ax.set_title(title)
        ax.grid(False)
    ax = fig.add_subplot(grid[2])
    lat = [loaders.CITIES[c][0] for c in features.index]
    for i, name in enumerate(CLUSTERS):
        mask = by_climate == i
        ax.scatter(
            np.array(lon)[mask],
            np.array(lat)[mask],
            s=60,
            color=plots.COLORS[i],
            label=name,
            zorder=3,
        )
    for x, y, city in zip(lon, lat, features.index, strict=True):
        ax.annotate(
            city,
            (x, y),
            fontsize=6.5,
            color=plots.INK_SECONDARY,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("довгота, °E")
    ax.set_ylabel("широта, °N")
    ax.set_title("Кластери на мапі")
    ax.legend(fontsize=7, loc="lower left")
    plots.save(fig, out, "16_weather_clusters.png")
    return by_dynamics, ari


def stage_synthesis(frame, labels, out, rng):
    temp = loaders.wide(frame, "temp")
    model = synthesis.fit_climate(temp)
    print(
        f"   VAR за BIC: порядок {model['var_z']['order']} для аномалій, "
        f"{model['var_scores']['order']} для нормальних рангів"
    )

    def metric(f):
        return metrics(f, labels)

    reports, samples = {}, {}
    for method, title in METHODS.items():
        reports[method], samples[method] = synthesis.monte_carlo(
            metric, temp, lambda g, m=method: synthesis.synth_climate(model, g, m), RUNS, rng
        )
        print(
            f"   {title}: у межах {int(reports[method]['в межах'].sum())} "
            f"з {len(reports[method])} показників"
        )

    rate = pd.DataFrame(index=CITY_METRICS + GLOBAL_METRICS, columns=list(METHODS), dtype=float)
    for method, report in reports.items():
        for name in CITY_METRICS:
            rate.loc[name, method] = report.loc[
                [f"{c}|{name}" for c in temp.columns], "в межах"
            ].mean()
        for name in GLOBAL_METRICS:
            rate.loc[name, method] = float(report.loc[name, "в межах"])
    print("   частка міст, де показник у 90 % інтервалі моделі:")
    print(rate.rename(columns=METHODS).round(2).to_string())

    main = reports[MAIN_METHOD]
    detail = main.loc[[f"{CITY}|{m}" for m in CITY_METRICS] + GLOBAL_METRICS].copy()
    detail.index = [i.split("|")[-1] for i in detail.index]
    print(f"   {CITY}, {METHODS[MAIN_METHOD]}:")
    print(detail.to_string(float_format=lambda v: f"{v:.3f}"))

    synth = synthesis.synth_climate(model, rng, MAIN_METHOD)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
    window = slice("2023-01-01", "2024-12-31")
    axes[0].plot(
        temp.loc[window].index, temp.loc[window, CITY], color=plots.REAL, lw=0.8, label="реальна"
    )
    axes[0].plot(
        synth.loc[window].index,
        synth.loc[window, CITY],
        color=plots.COLORS[0],
        lw=0.8,
        label="модельна",
    )
    axes[0].set_title(f"{CITY}: реальна і модельна температура (фрагмент 2023–2024)")
    axes[0].set_ylabel("°C")
    axes[0].legend()
    _, real_resid = analysis.seasonal_fit(temp)
    _, synth_resid = analysis.seasonal_fit(synth)
    plots.heatmap(
        axes[1],
        np.vstack([real_resid.corr().loc[WEST].values, synth_resid.corr().loc[WEST].values]),
        ["реальні", "модельні"],
        list(temp.columns),
        vmin=0,
        vmax=1,
        cmap=plots.SEQUENTIAL,
    )
    axes[1].set_title("Кореляція аномалій міст зі Львовом")
    plots.save(fig, out, "17_weather_synthetic.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = plots.heatmap(
        ax,
        rate.values,
        list(rate.index),
        list(METHODS.values()),
        cmap=plots.SEQUENTIAL,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Частка міст, де показник у 90 % інтервалі моделі")
    fig.colorbar(image, ax=ax, pad=0.02)
    plots.save(fig, out, "18_weather_methods.png")

    fig, axes = plt.subplots(len(SHOWN), len(METHODS), figsize=(11, 7))
    for row, (name, city) in enumerate(SHOWN):
        key = f"{city}|{name}" if city else name
        for col, method in enumerate(METHODS):
            plots.mc_panel(
                axes[row, col],
                samples[method][key],
                reports[method].loc[key, "реальний"],
                f"{METHODS[method]}\n{name}" if row == 0 else name,
            )
    fig.suptitle(
        f"Розподіл показників на {RUNS} модельних реалізаціях і реальне значення (лінія)",
        fontsize=11,
    )
    plots.save(fig, out, "19_weather_adequacy.png")
    return rate, detail


def run(data, out, rng):
    frame = loaders.weather(data / "weather.csv")
    hourly = loaders.hourly_temperature(data / "kyiv_hourly.csv")
    print("\n2.1. Вхідні дані")
    stage_input(frame, hourly, out)
    print("\n2.2. Декомпозиція")
    seasonality = stage_decomposition(frame, hourly, out)
    features = climate_features(frame)
    print("\n2.3. Властивості")
    stage_gradients(frame, seasonality, out)
    resid = stage_properties(frame, out)
    print("\n2.4. Кореляційний аналіз")
    stage_correlation(frame, resid, out)
    print("\n2.5. Кластеризація міст")
    labels, _ = stage_clustering(features, resid, out)
    print("\n2.6. Синтез моделі та перевірка адекватності")
    return stage_synthesis(frame, labels, out, rng)
