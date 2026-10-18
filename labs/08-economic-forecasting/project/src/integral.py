"""Багатокритеріальна інтегрована оцінка альтернатив за нелінійною схемою компромісів."""

import numpy as np
import pandas as pd


def normalize(matrix):
    """Нормування на суму по альтернативах: y⁰ᵢₖ = yᵢₖ / Σᵢ yᵢₖ, тому 0 ≤ y⁰ < 1 за двох і більше
    альтернатив. Усі критерії мінімізуються."""
    x = np.asarray(matrix, dtype=float)
    total = x.sum(axis=0)
    return np.divide(x, total, out=np.zeros_like(x), where=total > 0)


def compromise(matrix, weights):
    """Нелінійна схема компромісів: Yᵢ = Σₖ λₖ (1 − y⁰ᵢₖ)⁻¹ → min, Σ λₖ = 1.

    Чим ближче частковий критерій до своєї межі, тим швидше зростає штраф, тому
    альтернатива з одним дуже поганим критерієм програє рівномірно непоганій."""
    w = np.asarray(weights, dtype=float)
    return (w / w.sum() / (1.0 - normalize(matrix))).sum(axis=1)


def entropy_weights(matrix):
    """Об'єктивні ваги: критерій, за яким альтернативи відрізняються сильніше, важить більше.

    pᵢₖ = y⁰ᵢₖ,  Eₖ = −Σ pᵢₖ ln pᵢₖ / ln m,  λₖ = (1 − Eₖ) / Σ (1 − Eⱼ)."""
    p = normalize(matrix)
    m = p.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=0) / np.log(m)
    d = 1.0 - e
    return d / d.sum() if d.sum() > 0 else np.full(len(d), 1.0 / len(d))


def rate(table, weights):
    """Інтегральний показник і ранг кожної альтернативи (рядка таблиці критеріїв)."""
    out = pd.DataFrame(index=table.index)
    out["Y"] = compromise(table.to_numpy(), [weights[c] for c in table.columns])
    out["ранг"] = out["Y"].rank(method="min").astype(int)
    return out


def sensitivity(table, weights):
    """Ранги за трьома наборами ваг: заданими, рівними й ентропійними. Збіг рангів – ознака того,
    що висновок не залежить від суб'єктивного вибору ваг."""
    equal = dict.fromkeys(table.columns, 1.0)
    entropy = dict(zip(table.columns, entropy_weights(table.to_numpy()), strict=True))
    return pd.DataFrame(
        {
            "задані": rate(table, weights)["ранг"],
            "рівні": rate(table, equal)["ранг"],
            "ентропійні": rate(table, entropy)["ранг"],
        }
    ), entropy
