# Project: Rebuild Dilated-SE-FireNet bằng TensorFlow

> Mục tiêu: tạo lại pipeline huấn luyện, đánh giá và export TFLite INT8 cho mô hình **Dilated-SE-FireNet** phát hiện rung nhĩ AFIB từ ECG 1 kênh.

---

## 0. Tóm tắt bài toán

Mô hình nhận một cửa sổ ECG 10 giây:

```text
Sampling rate: 250 Hz
Window length: 10 giây
Input length: 2500 mẫu
Input shape: (2500, 1)
Output: 2 lớp
0 = Normal
1 = AFIB
```

Luồng tổng thể:

```text
Raw ECG
→ lọc nhiễu + resample + chuẩn hóa
→ cắt window 10 giây
→ gán nhãn Normal / AFIB
→ train Dilated-SE-FireNet bằng TensorFlow
→ đánh giá float32
→ export TFLite float32 / INT8
→ verify lại TFLite INT8 trên PC
→ deploy xuống edge device
```

---

## 1. Ý tưởng kiến trúc Dilated-SE-FireNet

Kiến trúc được ghép từ 3 khối chính:

| Thành phần | Mục đích |
|---|---|
| **Fire Module** | Giảm số tham số bằng squeeze 1x1 và expand 1x1 + 3x3 |
| **Dilated Conv1D** | Mở rộng vùng nhìn để học R-R interval dài hạn |
| **SE Block** | Tự chọn kênh đặc trưng quan trọng, giảm kênh nhiễu |

Cấu hình chính dùng trong project này:

| Block | Input channels | Cấu hình | Output channels |
|---|---:|---|---:|
| Stem | 1 | Conv1D filters=12, kernel=7, stride=2 | 12 |
| Block 1 | 12 | squeeze=12, expand=24, dilation=1 | 48 |
| Block 2 | 48 | squeeze=16, expand=32, dilation=2 | 64 |
| Block 3 | 64 | squeeze=24, expand=48, dilation=4 | 96 |
| Block 4 | 96 | squeeze=32, expand=64, dilation=8 | 128 |

Classifier:

```text
GlobalAveragePooling1D
→ Dropout
→ Dense(2, softmax)
```

---

## 2. Cấu trúc thư mục project

Tạo project như sau:

```text
dilated_se_firenet_tf/
│
├── data/
│   ├── raw_afdb/                  # chứa file .hea, .dat, .atr nếu preprocessing từ raw AFDB
│   └── processed/                 # chứa X_train.npy, y_train.npy, ...
│
├── outputs/
│   ├── checkpoints/
│   ├── reports/
│   └── tflite/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── model.py
│   ├── preprocess_afdb.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_tflite.py
│   └── verify_tflite.py
│
├── requirements.txt
└── README.md
```

---

## 3. Cài môi trường

### 3.1. Tạo môi trường Python

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3.2. File `requirements.txt`

Tạo file `requirements.txt`:

```txt
tensorflow>=2.10
numpy
scipy
scikit-learn
matplotlib
pandas
wfdb
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Ghi chú:

```text
Nếu dùng Windows GPU native, TensorFlow 2.10.x thường dễ dùng hơn.
Nếu dùng CPU hoặc Linux/WSL, có thể dùng TensorFlow mới hơn.
```

---

## 4. File `src/config.py`

```python
from pathlib import Path

# ============================================================
# PATH CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_AFDB_DIR = PROJECT_ROOT / "data" / "raw_afdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"
TFLITE_DIR = OUTPUT_DIR / "tflite"

for p in [DATA_DIR, RAW_AFDB_DIR, CKPT_DIR, REPORT_DIR, TFLITE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# ECG CONFIG
# ============================================================
TARGET_FS = 250
WINDOW_SECONDS = 10
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS  # 2500
STEP_SIZE = int(WINDOW_SIZE * 0.2)        # 500, overlap 80%
INPUT_SHAPE = (WINDOW_SIZE, 1)

LOW_CUT = 0.5
HIGH_CUT = 40.0
FILTER_ORDER = 4

# ============================================================
# LABEL CONFIG
# ============================================================
LABEL_NORMAL = 0
LABEL_AFIB = 1
NUM_CLASSES = 2

LABEL_MAP = {
    "(N": LABEL_NORMAL,
    "(AFIB": LABEL_AFIB,
}

# ============================================================
# MODEL CONFIG
# ============================================================
L2_RATE = 1e-3
LEAKY_RELU_ALPHA = 0.1
SE_RATIO = 4
DROPOUT_CLASSIFIER = 0.2
SPATIAL_DROPOUT = 0.2

STEM_FILTERS = 12

# name, squeeze_filters, expand_filters, dilation_rate, output_channels
# output_channels = expand_filters * 2
FIRE_BLOCKS = [
    ("block1", 12, 24, 1, 48),
    ("block2", 16, 32, 2, 64),
    ("block3", 24, 48, 4, 96),
    ("block4", 32, 64, 8, 128),
]

# ============================================================
# TRAIN CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 256
EPOCHS = 60
INITIAL_LR = 1e-3
LABEL_SMOOTHING = 0.1
PATIENCE_EARLY_STOP = 10
PATIENCE_REDUCE_LR = 5
```

---

## 5. File `src/utils.py`

```python
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
        print("GPU không khả dụng, đang chạy CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"Đã cấu hình memory growth cho {len(gpus)} GPU.")


def load_processed_data(data_dir: Path = cfg.DATA_DIR):
    paths = {
        "X_train": data_dir / "X_train.npy",
        "y_train": data_dir / "y_train.npy",
        "X_val": data_dir / "X_val.npy",
        "y_val": data_dir / "y_val.npy",
        "X_test": data_dir / "X_test.npy",
        "y_test": data_dir / "y_test.npy",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu file dữ liệu:\n" + "\n".join(missing))

    X_train = np.load(paths["X_train"]).astype("float32")
    y_train = np.load(paths["y_train"]).astype("int32")
    X_val = np.load(paths["X_val"]).astype("float32")
    y_val = np.load(paths["y_val"]).astype("int32")
    X_test = np.load(paths["X_test"]).astype("float32")
    y_test = np.load(paths["y_test"]).astype("int32")

    X_train = ensure_input_shape(X_train)
    X_val = ensure_input_shape(X_val)
    X_test = ensure_input_shape(X_test)

    print("Dataset loaded:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def ensure_input_shape(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        X = X[..., np.newaxis]
    if X.shape[1:] != cfg.INPUT_SHAPE:
        raise ValueError(f"Input shape sai. Cần (*, {cfg.INPUT_SHAPE}), nhận {X.shape}")
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
```

---

## 6. File `src/model.py`

```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from . import config as cfg


@tf.keras.utils.register_keras_serializable(package="DilatedSEFireNet")
class ECGAugmentation(layers.Layer):
    """
    Augmentation chỉ hoạt động khi training=True.
    Khi evaluate/export/inference, layer này tự trả về input gốc.
    """

    def __init__(self, noise_std=0.05, amp_min=0.9, amp_max=1.1, offset=0.05, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.amp_min = amp_min
        self.amp_max = amp_max
        self.offset = offset

    def call(self, x, training=None):
        if not training:
            return x

        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_std, dtype=x.dtype)
        amp = tf.random.uniform(
            shape=(tf.shape(x)[0], 1, 1),
            minval=self.amp_min,
            maxval=self.amp_max,
            dtype=x.dtype,
        )
        dc_offset = tf.random.uniform(
            shape=(tf.shape(x)[0], 1, 1),
            minval=-self.offset,
            maxval=self.offset,
            dtype=x.dtype,
        )
        return x * amp + noise + dc_offset

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "noise_std": self.noise_std,
                "amp_min": self.amp_min,
                "amp_max": self.amp_max,
                "offset": self.offset,
            }
        )
        return config


def conv_bn_lrelu(x, filters, kernel_size, strides=1, dilation_rate=1, name=None):
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=name,
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=None if name is None else f"{name}_lrelu")(x)
    return x


def se_block(x, ratio=cfg.SE_RATIO, name="se"):
    channels = int(x.shape[-1])
    hidden = max(channels // ratio, 1)

    se = layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)
    se = layers.Dense(
        hidden,
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"{name}_fc1",
    )(se)
    se = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"{name}_lrelu")(se)
    se = layers.Dense(
        channels,
        activation="sigmoid",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"{name}_fc2_sigmoid",
    )(se)
    se = layers.Reshape((1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def dilated_fire_module(x, squeeze_filters, expand_filters, dilation_rate, block_name):
    shortcut = x

    # 1. Squeeze: nén kênh bằng Conv1D 1x1
    s = conv_bn_lrelu(
        x,
        filters=squeeze_filters,
        kernel_size=1,
        strides=1,
        name=f"fire_{block_name}_squeeze_1x1",
    )

    # 2. Expand branch 1x1
    e1 = layers.Conv1D(
        filters=expand_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"fire_{block_name}_expand_1x1",
    )(s)
    e1 = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"fire_{block_name}_expand_1x1_lrelu")(e1)

    # 3. Expand branch 3x3 dilation
    e3 = layers.Conv1D(
        filters=expand_filters,
        kernel_size=3,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"fire_{block_name}_expand_3x3_dil{dilation_rate}",
    )(s)
    e3 = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"fire_{block_name}_expand_3x3_lrelu")(e3)

    # 4. Concatenate 1x1 + 3x3
    out = layers.Concatenate(axis=-1, name=f"fire_{block_name}_concat")([e1, e3])
    out = layers.BatchNormalization(name=f"fire_{block_name}_concat_bn")(out)

    # 5. SE attention
    out = se_block(out, ratio=cfg.SE_RATIO, name=f"fire_{block_name}_se")

    # 6. Residual projection nếu số kênh không khớp
    out_channels = int(out.shape[-1])
    shortcut_channels = int(shortcut.shape[-1])

    if shortcut_channels != out_channels:
        shortcut = layers.Conv1D(
            filters=out_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(cfg.L2_RATE),
            name=f"fire_{block_name}_shortcut_proj",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"fire_{block_name}_shortcut_bn")(shortcut)

    out = layers.Add(name=f"fire_{block_name}_add")([out, shortcut])
    out = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"fire_{block_name}_out_lrelu")(out)
    out = layers.SpatialDropout1D(cfg.SPATIAL_DROPOUT, name=f"fire_{block_name}_spatial_dropout")(out)
    return out


def learnable_downsample(x, filters, block_name):
    """
    Thay MaxPooling bằng Conv1D stride=2 để mô hình tự học cách giảm mẫu.
    """
    return conv_bn_lrelu(
        x,
        filters=filters,
        kernel_size=3,
        strides=2,
        name=f"{block_name}_learnable_downsample",
    )


def build_dilated_se_firenet(input_shape=cfg.INPUT_SHAPE, include_augmentation=True):
    inputs = layers.Input(shape=input_shape, name="Input_ECG")
    x = inputs

    if include_augmentation:
        x = ECGAugmentation(name="Augmentation_Block")(x)

    # Stem
    x = conv_bn_lrelu(
        x,
        filters=cfg.STEM_FILTERS,
        kernel_size=7,
        strides=2,
        name="Stem_Conv1D_k7_s2",
    )

    # Fire blocks + learnable downsampling giữa các block
    for i, (name, squeeze, expand, dilation, out_channels) in enumerate(cfg.FIRE_BLOCKS):
        x = dilated_fire_module(
            x,
            squeeze_filters=squeeze,
            expand_filters=expand,
            dilation_rate=dilation,
            block_name=name,
        )

        # Không downsample sau block cuối
        if i < len(cfg.FIRE_BLOCKS) - 1:
            x = learnable_downsample(x, filters=out_channels, block_name=f"down_after_{name}")

    # Classifier
    x = layers.GlobalAveragePooling1D(name="Global_Average_Pooling")(x)
    x = layers.Dropout(cfg.DROPOUT_CLASSIFIER, name="Classifier_Dropout")(x)
    outputs = layers.Dense(
        cfg.NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="Output_Softmax",
    )(x)

    return models.Model(inputs=inputs, outputs=outputs, name="Dilated_SE_FireNet")


if __name__ == "__main__":
    model = build_dilated_se_firenet()
    model.summary()
```

---

## 7. File `src/train.py`

```python
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, losses

from . import config as cfg
from .model import build_dilated_se_firenet
from .utils import (
    set_seed,
    setup_gpu_memory_growth,
    load_processed_data,
    to_one_hot,
    get_class_weights,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.DATA_DIR))
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.INITIAL_LR)
    args = parser.parse_args()

    set_seed()
    setup_gpu_memory_growth()

    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data(Path(args.data_dir))

    y_train_oh = to_one_hot(y_train)
    y_val_oh = to_one_hot(y_val)

    model = build_dilated_se_firenet(include_augmentation=True)
    model.summary()

    steps_per_epoch = max(len(X_train) // args.batch_size, 1)
    decay_steps = steps_per_epoch * args.epochs
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=decay_steps,
        alpha=0.05,
    )

    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=losses.CategoricalCrossentropy(label_smoothing=cfg.LABEL_SMOOTHING),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    class_weight = get_class_weights(y_train)
    print("Class weight:", class_weight)

    ckpt_path = cfg.CKPT_DIR / "best_dilated_se_firenet.keras"

    cb = [
        callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.CSVLogger(str(cfg.REPORT_DIR / "train_log.csv")),
    ]

    model.fit(
        X_train,
        y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        class_weight=class_weight,
        shuffle=True,
    )

    final_path = cfg.CKPT_DIR / "final_dilated_se_firenet.keras"
    model.save(final_path)
    print(f"Saved final model to: {final_path}")
    print(f"Best model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
```

---

## 8. File `src/evaluate.py`

```python
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from . import config as cfg
from .model import ECGAugmentation
from .utils import load_processed_data, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.DATA_DIR))
    parser.add_argument("--model_path", type=str, default=str(cfg.CKPT_DIR / "best_dilated_se_firenet.keras"))
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = parser.parse_args()

    _, _, _, _, X_test, y_test = load_processed_data(Path(args.data_dir))

    model = tf.keras.models.load_model(args.model_path)
    probs = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "AFIB"],
        digits=4,
        output_dict=True,
    )

    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred, target_names=["Normal", "AFIB"], digits=4))

    save_json(
        {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        cfg.REPORT_DIR / "evaluation_float32.json",
    )


if __name__ == "__main__":
    main()
```

---

## 9. File `src/export_tflite.py`

```python
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config as cfg
from .utils import load_processed_data


def representative_dataset_gen(X_rep, max_samples=1000):
    n = min(len(X_rep), max_samples)
    for i in range(n):
        sample = X_rep[i : i + 1].astype("float32")
        yield [sample]


def export_float32(model, output_path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Saved float32 TFLite: {output_path} | Size: {output_path.stat().st_size / 1024:.2f} KB")


def export_int8(model, X_rep, output_path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_rep, max_samples=1000)

    # Full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Saved INT8 TFLite: {output_path} | Size: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(cfg.CKPT_DIR / "best_dilated_se_firenet.keras"))
    parser.add_argument("--data_dir", type=str, default=str(cfg.DATA_DIR))
    args = parser.parse_args()

    X_train, _, _, _, _, _ = load_processed_data(Path(args.data_dir))
    model = tf.keras.models.load_model(args.model_path)

    export_float32(model, cfg.TFLITE_DIR / "dilated_se_firenet_float32.tflite")
    export_int8(model, X_train, cfg.TFLITE_DIR / "dilated_se_firenet_int8.tflite")


if __name__ == "__main__":
    main()
```

---

## 10. File `src/verify_tflite.py`

```python
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from . import config as cfg
from .utils import load_processed_data, save_json


def quantize_input(x_float, input_detail):
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return x_float.astype("float32")

    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Input quantization scale = 0")

    x_q = np.round(x_float / scale + zero_point)

    if dtype == np.int8:
        x_q = np.clip(x_q, -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        x_q = np.clip(x_q, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported input dtype: {dtype}")

    return x_q


def dequantize_output(y_raw, output_detail):
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return y_raw.astype("float32")

    scale, zero_point = output_detail["quantization"]
    return scale * (y_raw.astype("float32") - zero_point)


def run_tflite(tflite_path: Path, X, max_samples=None):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    print("Input detail:", input_detail)
    print("Output detail:", output_detail)

    n = len(X) if max_samples is None else min(len(X), max_samples)
    preds = []

    for i in range(n):
        x = X[i : i + 1].astype("float32")
        x_in = quantize_input(x, input_detail)

        interpreter.set_tensor(input_detail["index"], x_in)
        interpreter.invoke()

        y_raw = interpreter.get_tensor(output_detail["index"])
        y_float = dequantize_output(y_raw, output_detail)
        pred = int(np.argmax(y_float, axis=1)[0])
        preds.append(pred)

    return np.array(preds, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.DATA_DIR))
    parser.add_argument("--tflite_path", type=str, default=str(cfg.TFLITE_DIR / "dilated_se_firenet_int8.tflite"))
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    _, _, _, _, X_test, y_test = load_processed_data(Path(args.data_dir))
    max_samples = None if args.max_samples <= 0 else args.max_samples

    y_pred = run_tflite(Path(args.tflite_path), X_test, max_samples=max_samples)
    y_true = y_test[: len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=["Normal", "AFIB"], digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=["Normal", "AFIB"], digits=4, output_dict=True)

    print("TFLite Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(report_text)

    save_json(
        {
            "tflite_path": args.tflite_path,
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        },
        cfg.REPORT_DIR / "evaluation_tflite.json",
    )


if __name__ == "__main__":
    main()
```

---

## 11. File `src/preprocess_afdb.py`

Phần này dùng khi bạn muốn tạo lại dữ liệu từ raw AFDB. Nếu bạn đã có sẵn `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, bạn có thể bỏ qua file này.

```python
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import wfdb
from wfdb import processing
from scipy.signal import butter, sosfilt, resample_poly

from . import config as cfg


FILTER_SETTLE_SECONDS = 2
DROP_SAMPLES = FILTER_SETTLE_SECONDS * cfg.TARGET_FS


# Bạn chỉnh danh sách record theo dataset hiện tại của bạn.
TRAIN_RECORDS = [
    "04043", "04936", "07162", "07859", "07879", "08455", "06426", "05121",
    "06995", "05261", "06453", "04015", "04908", "04048", "08434", "08378",
]
VAL_RECORDS = ["04746", "07910", "08219"]
# TEST_RECORDS có thể tự lấy phần còn lại hoặc khai báo thủ công.
TEST_RECORDS = []
EXCLUDED_RECORDS = {"00735", "03665"}


def preprocess_signal(signal_data: np.ndarray, original_fs: int) -> np.ndarray:
    sig = signal_data.astype("float32")

    if int(original_fs) != cfg.TARGET_FS:
        gcd_fs = np.gcd(int(original_fs), int(cfg.TARGET_FS))
        up = int(cfg.TARGET_FS // gcd_fs)
        down = int(original_fs // gcd_fs)
        sig = resample_poly(sig, up, down).astype("float32")

    sig = sig - np.mean(sig)

    nyquist = 0.5 * cfg.TARGET_FS
    low = cfg.LOW_CUT / nyquist
    high = cfg.HIGH_CUT / nyquist
    sos = butter(cfg.FILTER_ORDER, [low, high], btype="band", output="sos")
    sig = sosfilt(sos, sig).astype("float32")
    return sig


def normalize_window(x: np.ndarray, clip_value: float = 5.0, eps: float = 0.05) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / max(std, eps)
    x = np.clip(x, -clip_value, clip_value)
    return x.astype("float32")


def create_continuous_label_mask(annotation, signal_length: int, original_fs: int) -> np.ndarray:
    mask = np.full(signal_length, -1, dtype=np.int8)
    ratio = cfg.TARGET_FS / float(original_fs)

    for i, raw_note in enumerate(annotation.aux_note):
        note = raw_note.strip("\x00").strip()

        start = annotation.sample[i]
        end = annotation.sample[i + 1] if i < len(annotation.sample) - 1 else int(signal_length / ratio)

        if int(original_fs) != cfg.TARGET_FS:
            start = int(start * ratio)
            end = int(end * ratio)

        start = max(0, min(start, signal_length))
        end = max(0, min(end, signal_length))

        if note in cfg.LABEL_MAP:
            mask[start:end] = cfg.LABEL_MAP[note]

    return mask


def has_flatline(segment: np.ndarray, fs: int, duration_threshold=0.2, std_threshold=0.005) -> bool:
    chunk_len = int(duration_threshold * fs)
    if chunk_len <= 0 or chunk_len >= len(segment):
        return False

    n = len(segment) // chunk_len
    for i in range(n):
        chunk = segment[i * chunk_len : (i + 1) * chunk_len]
        if np.std(chunk) < std_threshold:
            return True
    return False


def validate_afib_quality(segment: np.ndarray, fs: int) -> bool:
    try:
        qrs_inds = processing.xqrs_detect(sig=segment, fs=fs, verbose=False)
        return len(qrs_inds) >= 5
    except Exception:
        return False


def validate_sinus_rhythm(segment: np.ndarray, fs: int) -> bool:
    try:
        qrs_inds = processing.xqrs_detect(sig=segment, fs=fs, verbose=False)
        if len(qrs_inds) < 4:
            return False

        rr = np.diff(qrs_inds)
        mean_rr = np.mean(rr)
        std_rr = np.std(rr)
        if mean_rr <= 0:
            return False

        cv = std_rr / mean_rr
        if cv > 0.15:
            return False

        lower = 0.8 * mean_rr
        upper = 1.2 * mean_rr
        if np.any(rr < lower) or np.any(rr > upper):
            return False

        return True
    except Exception:
        return False


def decide_window_label(mask_window: np.ndarray):
    valid_ratio = np.mean(mask_window != -1)
    if valid_ratio < 0.8:
        return None

    afib_ratio = np.mean(mask_window == cfg.LABEL_AFIB)
    normal_ratio = np.mean(mask_window == cfg.LABEL_NORMAL)

    if afib_ratio >= 0.5:
        return cfg.LABEL_AFIB
    if normal_ratio >= 0.6:
        return cfg.LABEL_NORMAL
    return None


def process_records(record_list: List[str], raw_dir: Path, split_name: str, save_dir: Path):
    X, y = [], []
    stats = {
        "total_windows": 0,
        "skipped_noise": 0,
        "skipped_label": 0,
        "skipped_irregular_normal": 0,
        "skipped_garbage_afib": 0,
    }

    for rec_name in record_list:
        rec_path = raw_dir / rec_name
        if not (rec_path.with_suffix(".hea").exists() and rec_path.with_suffix(".dat").exists()):
            print(f"Skip {rec_name}: thiếu .hea/.dat")
            continue

        print(f"Processing {split_name}: {rec_name}")
        record = wfdb.rdrecord(str(rec_path), channels=[0])
        annotation = wfdb.rdann(str(rec_path), "atr")

        raw_sig = record.p_signal[:, 0]
        clean_sig = preprocess_signal(raw_sig, int(record.fs))
        mask = create_continuous_label_mask(annotation, len(clean_sig), int(record.fs))

        start = DROP_SAMPLES
        end_limit = len(clean_sig) - cfg.WINDOW_SIZE

        for start_idx in range(start, end_limit, cfg.STEP_SIZE):
            end_idx = start_idx + cfg.WINDOW_SIZE
            seg = clean_sig[start_idx:end_idx]
            m = mask[start_idx:end_idx]
            stats["total_windows"] += 1

            if has_flatline(seg, cfg.TARGET_FS):
                stats["skipped_noise"] += 1
                continue

            label = decide_window_label(m)
            if label is None:
                stats["skipped_label"] += 1
                continue

            if label == cfg.LABEL_NORMAL and not validate_sinus_rhythm(seg, cfg.TARGET_FS):
                stats["skipped_irregular_normal"] += 1
                continue

            if label == cfg.LABEL_AFIB and not validate_afib_quality(seg, cfg.TARGET_FS):
                stats["skipped_garbage_afib"] += 1
                continue

            seg = normalize_window(seg)
            X.append(seg[:, None])
            y.append(label)

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int32")

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"X_{split_name}.npy", X)
    np.save(save_dir / f"y_{split_name}.npy", y)

    print(f"Done {split_name}: X={X.shape}, y={y.shape}")
    print("Stats:", stats)


def infer_test_records(raw_dir: Path, train_records: List[str], val_records: List[str]) -> List[str]:
    all_records = sorted([p.stem for p in raw_dir.glob("*.hea")])
    used = set(train_records + val_records) | EXCLUDED_RECORDS
    return [r for r in all_records if r not in used]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default=str(cfg.RAW_AFDB_DIR))
    parser.add_argument("--save_dir", type=str, default=str(cfg.DATA_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    save_dir = Path(args.save_dir)

    test_records = TEST_RECORDS or infer_test_records(raw_dir, TRAIN_RECORDS, VAL_RECORDS)

    print("Train records:", TRAIN_RECORDS)
    print("Val records:", VAL_RECORDS)
    print("Test records:", test_records)

    process_records(TRAIN_RECORDS, raw_dir, "train", save_dir)
    process_records(VAL_RECORDS, raw_dir, "val", save_dir)
    process_records(test_records, raw_dir, "test", save_dir)


if __name__ == "__main__":
    main()
```

---

## 12. Cách chạy toàn bộ project

### 12.1. Trường hợp đã có sẵn dữ liệu `.npy`

Đặt các file sau vào:

```text
data/processed/
```

Cần có:

```text
X_train.npy
 y_train.npy
X_val.npy
 y_val.npy
X_test.npy
 y_test.npy
```

Shape đúng:

```text
X_train: (N, 2500, 1)
y_train: (N,)
```

Train:

```bash
python -m src.train --data_dir data/processed --epochs 60 --batch_size 256
```

Evaluate float32:

```bash
python -m src.evaluate --data_dir data/processed --model_path outputs/checkpoints/best_dilated_se_firenet.keras
```

Export TFLite:

```bash
python -m src.export_tflite --model_path outputs/checkpoints/best_dilated_se_firenet.keras --data_dir data/processed
```

Verify TFLite INT8:

```bash
python -m src.verify_tflite --data_dir data/processed --tflite_path outputs/tflite/dilated_se_firenet_int8.tflite
```

Verify nhanh 1000 mẫu:

```bash
python -m src.verify_tflite --data_dir data/processed --tflite_path outputs/tflite/dilated_se_firenet_int8.tflite --max_samples 1000
```

---

### 12.2. Trường hợp muốn preprocess lại từ raw AFDB

Đặt raw AFDB vào:

```text
data/raw_afdb/
```

Ví dụ:

```text
04043.hea
04043.dat
04043.atr
...
```

Chạy:

```bash
python -m src.preprocess_afdb --raw_dir data/raw_afdb --save_dir data/processed
```

Sau đó train như mục 12.1.

---

## 13. Checklist kiểm tra model trước khi train lâu

Chạy nhanh file model:

```bash
python -m src.model
```

Cần kiểm tra:

```text
Input shape: (None, 2500, 1)
Output shape: (None, 2)
Output activation: softmax
Số tham số ở mức nhỏ, phù hợp edge
```

Kiểm tra dữ liệu:

```python
import numpy as np

X = np.load("data/processed/X_train.npy")
y = np.load("data/processed/y_train.npy")

print(X.shape)
print(y.shape)
print(np.unique(y, return_counts=True))
print(np.min(X), np.max(X), np.mean(X), np.std(X))
```

Kỳ vọng:

```text
X shape = (N, 2500, 1)
y chỉ có 0 và 1
biên độ X đã chuẩn hóa, thường nằm gần [-5, 5]
```

---

## 14. Checklist sau khi export INT8

Sau khi chạy `verify_tflite.py`, cần so sánh:

```text
Float32 accuracy
INT8 accuracy
Float32 recall AFIB
INT8 recall AFIB
Float32 confusion matrix
INT8 confusion matrix
```

Nếu INT8 tụt mạnh, kiểm tra:

```text
1. Representative dataset có đúng preprocessing không?
2. Input dtype của TFLite là int8 hay float32?
3. Có quantize input đúng scale/zero_point không?
4. Output có dequantize đúng scale/zero_point không?
5. Model có layer nào không hỗ trợ INT8 toàn phần không?
```

---

## 15. Lỗi thường gặp

### Lỗi 1: Input shape sai

Thông báo thường gặp:

```text
expected shape=(None, 2500, 1), found shape=(None, 2500)
```

Cách sửa:

```python
X = X[..., np.newaxis]
```

---

### Lỗi 2: Output softmax nhưng code lại xử lý sigmoid

Model này output:

```text
[p_normal, p_afib]
```

Cách xử lý đúng:

```python
pred = np.argmax(output, axis=1)
```

Không dùng:

```python
if output > 0.5
```

trừ khi model của bạn là `Dense(1, sigmoid)`.

---

### Lỗi 3: TFLite INT8 output nhìn rất lạ

Output raw int8 không phải xác suất thật.

Phải dequantize:

```python
y_float = output_scale * (y_int8 - output_zero_point)
```

---

### Lỗi 4: Model train tốt nhưng test kém

Nguyên nhân thường gặp:

```text
Chia train/val/test theo window, không chia theo bệnh nhân
Dữ liệu Normal còn chứa ngoại tâm thu hoặc nhịp không đều
Dữ liệu AFIB chứa nhiễu quá nặng
Chuẩn hóa train/test không giống nhau
Augmentation quá mạnh
```

---

## 16. Ghi chú cho Edge AI Engineer

Khi chuyển xuống MCU/NPU, cần nhớ 5 thứ:

```text
1. Input shape: [1, 2500, 1]
2. Input dtype: thường là int8 sau quantization
3. Phải dùng đúng input scale và zero_point
4. Output softmax int8 phải dequantize trước khi đọc xác suất
5. Logic kết luận: argmax([p_normal, p_afib])
```

Pseudo-code trên MCU:

```c
// 1. Copy input int8 vào input tensor
for (int i = 0; i < 2500; i++) {
    input_tensor->data.int8[i] = ecg_input_int8[i];
}

// 2. Invoke model
TfLiteStatus invoke_status = interpreter->Invoke();

// 3. Read output
int8_t normal_raw = output_tensor->data.int8[0];
int8_t afib_raw   = output_tensor->data.int8[1];

// 4. Dequantize
float normal_prob = output_scale * (normal_raw - output_zero_point);
float afib_prob   = output_scale * (afib_raw - output_zero_point);

// 5. Decision
if (afib_prob > normal_prob) {
    // AFIB
} else {
    // Normal
}
```

---

## 17. Thứ tự làm việc khuyến nghị

Với người mới Edge AI, làm theo thứ tự này:

```text
Bước 1: Chạy python -m src.model để xem summary
Bước 2: Load thử X_train.npy, kiểm tra shape
Bước 3: Train 3 epoch để chắc pipeline không lỗi
Bước 4: Train full 60 epoch
Bước 5: Evaluate float32
Bước 6: Export TFLite INT8
Bước 7: Verify TFLite INT8 trên PC
Bước 8: So sánh float32 vs int8
Bước 9: Mang .tflite sang tool convert NPU / TFLM
Bước 10: Kiểm tra input/output scale zero_point trên board
```

---

## 18. Kết luận

Project này tái tạo lại mô hình **Dilated-SE-FireNet** bằng TensorFlow theo đúng tinh thần:

```text
Fire Module       → model nhỏ
Dilated Conv      → nhìn được R-R interval dài hạn
SE Block          → chọn lọc feature quan trọng, giảm nhiễu
Conv1D stride=2   → giảm RAM activation
Softmax 2 lớp     → phân loại Normal / AFIB
INT8 TFLite       → sẵn sàng cho edge device
```

Khi triển khai thực tế, phần quan trọng nhất không chỉ là train model, mà là đảm bảo:

```text
Preprocessing PC == Preprocessing board
Input quantization đúng
Output dequantization đúng
Logic post-processing đúng
```
