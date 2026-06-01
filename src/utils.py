import os
import random
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from . import config as cfg

def set_seed(seed: int = cfg.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU is not available, running on CPU.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"Configured memory growth for {len(gpus)} GPU(s).")

def load_processed_data(data_dir: Path = cfg.DATA_DIR):
    paths = {
        "X_train": data_dir / "train" / "X_train.npy",
        "y_train": data_dir / "train" / "y_train.npy",
        "X_val": data_dir / "val" / "X_val.npy",
        "y_val": data_dir / "val" / "y_val.npy",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing data files:\n" + "\n".join(missing))

    X_train = np.load(paths["X_train"]).astype("float32")
    y_train = np.load(paths["y_train"]).astype("int32")
    X_val = np.load(paths["X_val"]).astype("float32")
    y_val = np.load(paths["y_val"]).astype("int32")

    # Ensure shape has channels dimension (None, 2500, 1)
    X_train = ensure_input_shape(X_train)
    X_val = ensure_input_shape(X_val)

    print("Datasets loaded successfully:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, "y_val:  ", y_val.shape)

    return X_train, y_train, X_val, y_val

def ensure_input_shape(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        X = X[..., np.newaxis]
    if X.shape[1:] != cfg.INPUT_SHAPE:
        raise ValueError(f"Input shape mismatch. Expected (*, {cfg.INPUT_SHAPE}), got {X.shape}")
    return X

def to_one_hot(y: np.ndarray) -> np.ndarray:
    return tf.keras.utils.to_categorical(y, num_classes=cfg.NUM_CLASSES)

def get_class_weights(y_train: np.ndarray) -> dict:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
