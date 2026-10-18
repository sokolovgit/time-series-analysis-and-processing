"""Паралельне навчання мереж у пулі процесів з кешем результатів на диску."""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

WORKERS = max(1, (os.cpu_count() or 2) - 2)


def init():
    # кожна мережа маленька, тож потоки torch всередині процесу лише заважають одне одному
    import torch

    torch.set_num_threads(1)


def job(task):
    import strategies

    return task["key"], strategies.run(
        task["strategy"], task["kind"], task["y"], task["steps"], task["spec"]
    )


def run(tasks, path, fresh=False):
    """Виконати задачі {key, strategy, kind, y, steps, spec}; готові ключі беруться з кешу."""
    cache = {}
    if path.exists() and not fresh:
        with path.open("rb") as f:
            cache = pickle.load(f)
    todo = [t for t in tasks if t["key"] not in cache]
    print(f"   задач {len(tasks)}, з кешу {len(tasks) - len(todo)}, навчається {len(todo)}")
    if todo:
        with ProcessPoolExecutor(WORKERS, initializer=init) as pool:
            futures = [pool.submit(job, t) for t in todo]
            for i, future in enumerate(as_completed(futures), 1):
                key, result = future.result()
                cache[key] = result
                if i % 10 == 0 or i == len(todo):
                    print(f"   готово {i}/{len(todo)}", flush=True)
                    with path.open("wb") as f:
                        pickle.dump(cache, f)
    return {t["key"]: cache[t["key"]] for t in tasks}
