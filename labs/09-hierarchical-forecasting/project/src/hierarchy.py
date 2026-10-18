"""Ієрархія Україна → область → тип місцевості: вузли, рівні, сумувальна матриця S."""

import numpy as np

TOP = "Україна"
REGIONS = (
    "Вінницька",
    "Волинська",
    "Дніпропетровська",
    "Донецька",
    "Житомирська",
    "Закарпатська",
    "Запорізька",
    "Івано-Франківська",
    "Київська",
    "Кіровоградська",
    "Луганська",
    "Львівська",
    "Миколаївська",
    "Одеська",
    "Полтавська",
    "Рівненська",
    "Сумська",
    "Тернопільська",
    "Харківська",
    "Херсонська",
    "Хмельницька",
    "Черкаська",
    "Чернівецька",
    "Чернігівська",
    "Київ",
)
EXCLUDED = ("Автономна Республіка Крим", "Севастополь")
LOCAL = ("міська", "сільська")
LEVELS = ("країна", "область", "місцевість")


def build(births, depth, last=None):
    """Ієрархія глибини 2 (країна → область) або 3 (+ міська/сільська) до року last.

    Повертає dict: names, level, parent, bottom (індекси рядів у S), S розміром n×m,
    years і Y розміром T×n. Верхні рівні Y рахуються як S·b, тож ієрархія когерентна точно.
    """
    frame = births if last is None else births[births["year"] <= last]
    table = frame.pivot_table(index="year", columns=["region", "area"], values="births")
    names, level, parent = [TOP], [0], [-1]
    leaves = []
    for region in REGIONS:
        names.append(region)
        level.append(1)
        parent.append(0)
        if depth == 2:
            leaves.append((len(names) - 1, (region, "усього")))
    if depth == 3:
        for r, region in enumerate(REGIONS, start=1):
            for area in LOCAL:
                if (region, area) in table.columns:
                    names.append(f"{region}, {area}")
                    level.append(2)
                    parent.append(r)
                    leaves.append((len(names) - 1, (region, area)))
    parent = np.array(parent)
    bottom = np.array([i for i, _ in leaves])
    S = np.zeros((len(names), len(bottom)))
    for j, i in enumerate(bottom):
        node = i
        while node >= 0:
            S[node, j] = 1.0
            node = parent[node]
    B = table[[key for _, key in leaves]].to_numpy(dtype=float)
    return {
        "names": names,
        "level": np.array(level),
        "parent": parent,
        "bottom": bottom,
        "S": S,
        "years": table.index.to_numpy(),
        "Y": B @ S.T,
        "depth": depth,
    }
