"""Нейромережі прогнозу на крок: багатошаровий перцептрон, LSTM і одновимірна згорткова мережа."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
# на MPS дрібні мережі повільніші за CPU, а Conv1D у кількох процесах падає з assertion
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")

import time

import keras
import numpy as np

KINDS = ("MLP", "LSTM", "CNN")
SIZES = {"малий": 16, "базовий": 32}
LOSSES = ("MSE", "Huber")
HUBER_DELTA = 1.0
EPOCHS, BATCH, PATIENCE, VAL_SHARE, LR = 300, 32, 25, 0.2, 1e-3
KERNEL = 3


def build(kind, lookback, units):
    """MLP: L → 2u → u → 1; LSTM: L×1 → LSTM(u) → 1; CNN: L×1 → Conv(u) → Conv(u) → u → 1."""
    layers = keras.layers
    if kind == "MLP":
        inputs = keras.Input((lookback,), name="window")
        x = layers.Dense(2 * units, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(units, activation="relu", name="dense_2")(x)
    elif kind == "LSTM":
        inputs = keras.Input((lookback, 1), name="window")
        x = layers.LSTM(units, name="lstm")(inputs)
    else:
        inputs = keras.Input((lookback, 1), name="window")
        x = layers.Conv1D(units, KERNEL, activation="relu", name="conv_1")(inputs)
        x = layers.Conv1D(units, KERNEL, activation="relu", name="conv_2")(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(units, activation="relu", name="dense")(x)
    outputs = layers.Dense(1, name="output")(x)
    return keras.Model(inputs, outputs, name=kind)


def loss_fn(name):
    """MSE = mean(e²); Huber = mean(e²/2 при |e| ≤ δ, δ(|e| − δ/2) інакше)."""
    return "mse" if name == "MSE" else keras.losses.Huber(delta=HUBER_DELTA)


def windows(x, lookback):
    """Навчальні пари: вхід x[j − L : j], ціль x[j], j = L … n − 1."""
    x = np.asarray(x, dtype="float32")
    idx = np.arange(lookback, len(x))
    return np.stack([x[j - lookback : j] for j in idx]), x[idx]


def shaped(kind, x):
    return x if kind == "MLP" else x[..., None]


def fit(kind, x, lookback, units, loss, seed):
    """Навчання на стандартизованому ряді; рання зупинка за останніми VAL_SHARE навчальних пар."""
    keras.utils.set_random_seed(seed)
    model = build(kind, lookback, units)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss=loss_fn(loss))
    inputs, targets = windows(x, lookback)
    stop = keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    start = time.perf_counter()
    history = model.fit(
        shaped(kind, inputs),
        targets,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=VAL_SHARE,
        callbacks=[stop],
        verbose=0,
    )
    return model, {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "секунд": time.perf_counter() - start,
        "параметрів": model.count_params(),
        "епох": int(np.argmin(history.history["val_loss"])) + 1,
    }


def predict(model, kind, inputs):
    out = model(shaped(kind, np.asarray(inputs, dtype="float32")), training=False)
    return keras.ops.convert_to_numpy(out)[:, 0].astype(float)


def one_step(model, kind, x, lookback):
    """Прогноз на крок на ділянці спостереження: x̂[j] = f(x[j − L : j])."""
    inputs, _ = windows(x, lookback)
    return predict(model, kind, inputs)


def recursive(model, kind, history, lookback, steps):
    """Рекурсивна екстраполяція: x̂[n + h] = f(…, x̂[n + h − 1]), прогноз стає входом."""
    buffer = list(np.asarray(history[-lookback:], dtype="float32"))
    out = np.empty(steps)
    for h in range(steps):
        out[h] = predict(model, kind, [buffer[-lookback:]])[0]
        buffer.append(out[h])
    return out
