"""Закономірності між рядами: кореляційний і крос-кореляційний аналіз, кластеризація."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

EARTH_RADIUS = 6371.0
K_RANGE = range(2, 7)
SEED = 0


def correlation(frame, method="pearson"):
    return frame.corr(method=method)


def cross_correlation(x, y, max_lag):
    """r(τ) = corr(x(t), y(t + τ)). Пік при τ > 0 означає, що x випереджає y на τ відліків."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    r = []
    for lag in lags:
        ahead, behind = max(0, lag), max(0, -lag)
        r.append(np.corrcoef(x[behind : n - ahead], y[ahead : n - behind])[0, 1])
    return lags, np.array(r)


def peak_lag(x, y, max_lag):
    lags, r = cross_correlation(x, y, max_lag)
    i = np.argmax(r)
    return lags[i], r[i]


def distance_km(a, b):
    """Відстань по великому колу (формула гаверсинусів) між точками (φ, λ) у градусах."""
    (lat1, lon1), (lat2, lon2) = np.radians(a), np.radians(b)
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(h))


def best_kmeans(features, k_range=K_RANGE):
    """K-means на стандартизованих ознаках, k обирається за максимумом коефіцієнта силуету."""
    x = StandardScaler().fit_transform(features)
    scores = {}
    labels = {}
    for k in k_range:
        labels[k] = KMeans(k, n_init=20, random_state=SEED).fit_predict(x)
        scores[k] = silhouette_score(x, labels[k])
    best = max(scores, key=lambda key: scores[key])
    return best, labels[best], scores


def ward(features):
    """Ієрархічна кластеризація Уорда на стандартизованих ознаках."""
    return linkage(StandardScaler().fit_transform(features), method="ward")


def correlation_linkage(frame):
    """Ієрархія за кореляційною відстанню d = √(2(1 − r)), метод середнього зв'язку."""
    r = frame.corr().values
    d = np.sqrt(np.clip(2 * (1 - r), 0, None))
    np.fill_diagonal(d, 0.0)
    return linkage(squareform(d, checks=False), method="average")


def cut(tree, k):
    return fcluster(tree, k, criterion="maxclust") - 1


def agreement(a, b):
    """Скоригований індекс Ренда: 1 – розбиття збігаються, ≈0 – випадкова відповідність."""
    return adjusted_rand_score(a, b)


def relabel(labels, order):
    """Перенумерувати кластери в порядку першої появи за ключем order (час, довгота)."""
    seen = []
    for i in np.argsort(order, kind="stable"):
        if labels[i] not in seen:
            seen.append(labels[i])
    mapping = {old: new for new, old in enumerate(seen)}
    return np.array([mapping[v] for v in labels])


def segments(labels):
    """Неперервні ділянки однакової мітки: список (мітка, початок, кінець) з кінцем не включно."""
    labels = np.asarray(labels)
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [(labels[s], s, e) for s, e in zip(starts, ends, strict=True)]
