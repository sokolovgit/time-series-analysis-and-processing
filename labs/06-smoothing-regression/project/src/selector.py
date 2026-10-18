"""Алгоритм вибору між експоненційним і регресійним згладжуванням за властивостями ряду.

Порівнюються три стратегії вибору сім'ї моделей для нового ряду:
- дерево рішень за ознаками ряду, навчене на синтетичних рядах з відомою кращою сім'єю;
- внутрішня крос-валідація: сім'я з меншою CV-похибкою на тих самих стартах у вікні;
- комбінація: середнє прогнозів обох сімей, без вибору;
- гібрид: одна сім'я, лише якщо її CV-похибка менша щонайменше в k разів, інакше комбінація.
Поріг k підбирається на навчальній частині набору.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import exponential
import features
import regression
import synthetic

DEPTH = 3
MIN_LEAF = 15
FAMILIES = ("експоненційне", "регресійне")
TEST_SHARE = 1 / 3
THRESHOLDS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, np.inf)


def family_forecasts(train, h, season):
    """Прогнози обох сімей, кожна з власним CV-вибором, і їхні найменші CV-похибки."""
    horizon = len(train) // 4
    best, alpha, _, alphas = exponential.select_cv(train, season, horizon)
    spec, s = exponential.candidates(season)[best]
    f_exp = exponential.fit(train, spec, s, alpha).forecast(h)
    name, rows = regression.select(train, horizon, season)
    kind, kw = rows[name]["spec"]
    f_reg = regression.run(kind, train, h, kw)[0]
    return {
        "forecast": {FAMILIES[0]: f_exp, FAMILIES[1]: f_reg, "комбінація": (f_exp + f_reg) / 2},
        "CV": {FAMILIES[0]: alphas[alpha], FAMILIES[1]: rows[name]["CV RMSE"]},
        "моделі": {FAMILIES[0]: best, FAMILIES[1]: name},
    }


def evaluate_series(y, window, known_period=None):
    """Ознаки за вікном навчання, CV-похибки і RMSE на відкладеній частині 0,5W."""
    train, test = y[:window], y[window : window + window // 2]
    season, _ = features.period(train, known_period)
    out = family_forecasts(train, len(test), season)
    row = dict(features.selector_features(train, known_period))
    for k, f in out["forecast"].items():
        row[f"RMSE {k}"] = float(np.sqrt(np.mean((f - test) ** 2)))
    for k, v in out["CV"].items():
        row[f"CV {k}"] = v
    return row | {"виявлений період": season}


def build_benchmark(rng, count):
    rows = []
    for _ in range(count):
        y, meta = synthetic.benchmark_series(rng)
        window = 2 * len(y) // 3
        rows.append(evaluate_series(y, window) | meta)
    table = pd.DataFrame(rows)
    table["краще"] = np.where(
        table["RMSE експоненційне"] <= table["RMSE регресійне"], FAMILIES[0], FAMILIES[1]
    )
    return table


def feature_columns(table):
    return [c for c in table.columns if c in features.SELECTOR_FEATURES]


def train(table):
    """Дерево рішень на навчальній частині набору; повертає модель і текст правил."""
    cols = feature_columns(table)
    tree = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=0)
    tree.fit(table[cols].to_numpy(), table["краще"].to_numpy())
    return tree, rules(tree, cols)


def rules(tree, cols):
    """Правила дерева текстом; розгалуження з однаковим класом в обох гілках згортається."""
    t = tree.tree_
    names = tree.classes_

    def label(node):
        if t.children_left[node] == -1:
            return names[np.argmax(t.value[node])]
        left, right = label(t.children_left[node]), label(t.children_right[node])
        return left if left == right and left is not None else None

    def walk(node, depth):
        pad = "    " * depth
        cls = label(node)
        if cls is not None:
            n = int(t.n_node_samples[node])
            return [f"{pad}→ {cls} ({n} рядів)"]
        f, thr = cols[t.feature[node]], t.threshold[node]
        return [
            f"{pad}якщо {f} ≤ {thr:.3f}:",
            *walk(t.children_left[node], depth + 1),
            f"{pad}інакше ({f} > {thr:.3f}):",
            *walk(t.children_right[node], depth + 1),
        ]

    return "\n".join(walk(0, 0))


def choose(tree, feats):
    return tree.predict(np.array([list(feats.values())]))[0]


def regret(table, chosen):
    """Відношення RMSE стратегії до RMSE кращої з двох сімей (оракула) і частка влучань."""
    best = np.minimum(table["RMSE експоненційне"], table["RMSE регресійне"])
    ratio = np.asarray(chosen) / best
    return {
        "RMSE / оракул": float(np.mean(ratio)),
        "медіана": float(np.median(ratio)),
        "гірше за оракул на 20 %+": float(np.mean(ratio > 1.2)),
    }


def picked(table, picks):
    return np.where(picks == FAMILIES[0], table["RMSE експоненційне"], table["RMSE регресійне"])


def strategies(tree, k, test):
    cols = feature_columns(test)
    by_cv = np.where(test["CV експоненційне"] <= test["CV регресійне"], *FAMILIES)
    by_tree = tree.predict(test[cols].to_numpy())
    out = {
        "дерево за ознаками": regret(test, picked(test, by_tree)) | hits(test, by_tree),
        "внутрішня CV": regret(test, picked(test, by_cv)) | hits(test, by_cv),
        "комбінація": regret(test, test["RMSE комбінація"]),
        "гібрид": regret(test, hybrid_rmse(test, k)),
    }
    for f in FAMILIES:
        out[f"завжди {f}"] = regret(test, test[f"RMSE {f}"])
    return out


def hybrid_choice(cv_exp, cv_reg, k):
    """Вибір гібриду за відношенням CV-похибок: q = CV_exp / CV_reg."""
    q = np.asarray(cv_exp) / np.asarray(cv_reg)
    return np.where(q * k <= 1, FAMILIES[0], np.where(q >= k, FAMILIES[1], "комбінація"))


def hybrid_rmse(table, k):
    choice = hybrid_choice(table["CV експоненційне"], table["CV регресійне"], k)
    return np.array([table[f"RMSE {c}"].iloc[i] for i, c in enumerate(choice)])


def tune_threshold(table):
    """Поріг k з найменшим середнім відношенням RMSE до оракула на навчальних рядах."""
    scores = {k: regret(table, hybrid_rmse(table, k))["RMSE / оракул"] for k in THRESHOLDS}
    return min(scores, key=lambda k: scores[k]), scores


def hits(table, picks):
    return {"влучання": float(np.mean(picks == table["краще"].to_numpy()))}


def cross_validate(table, folds=3):
    """Кожна стратегія оцінюється на рядах, яких не бачили ні дерево, ні підбір порогу k."""
    parts = np.array_split(np.arange(len(table)), folds)
    runs = []
    for idx in parts:
        test, train_part = table.iloc[idx], table.drop(table.index[idx])
        tree, _ = train(train_part)
        k, _ = tune_threshold(train_part)
        runs.append(
            {"k": k} | {n: r["RMSE / оракул"] for n, r in strategies(tree, k, test).items()}
        )
    return runs
