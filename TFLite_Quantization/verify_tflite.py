import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src import config as cfg
from src.utils import save_json, ensure_input_shape


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


def run_tflite(tflite_path: Path, X, max_samples=None, batch_size=1000):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    n = len(X) if max_samples is None else min(len(X), max_samples)
    preds = []

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        current_batch_size = end_idx - start_idx

        # Resize input tensor dynamically for the current batch size
        interpreter.resize_tensor_input(input_detail["index"], [current_batch_size, 2500, 1])
        interpreter.allocate_tensors()

        # Retrieve the resized tensor details
        resized_input_detail = interpreter.get_input_details()[0]
        resized_output_detail = interpreter.get_output_details()[0]

        # Extract batch and convert/quantize
        x_batch = X[start_idx:end_idx].astype("float32")
        x_in = quantize_input(x_batch, resized_input_detail)

        interpreter.set_tensor(resized_input_detail["index"], x_in)
        interpreter.invoke()

        y_raw = interpreter.get_tensor(resized_output_detail["index"])
        y_float = dequantize_output(y_raw, resized_output_detail)
        
        batch_preds = np.argmax(y_float, axis=1)
        preds.extend(batch_preds.tolist())

    return np.array(preds, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.DATA_DIR))
    parser.add_argument("--tflite_path", type=str, default=str(cfg.TFLITE_DIR / "dilated_se_firenet_int8.tflite"))
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_test_path = data_dir / "test" / "X_test.npy"
    y_test_path = data_dir / "test" / "y_test.npy"

    if not (X_test_path.exists() and y_test_path.exists()):
        raise FileNotFoundError(f"Missing test data files at {data_dir / 'test/'}")

    X_test = np.load(X_test_path).astype("float32")
    y_test = np.load(y_test_path).astype("int32")
    X_test = ensure_input_shape(X_test)

    max_samples = None if args.max_samples <= 0 else args.max_samples
    if max_samples:
        print(f"Running verification on subset of {max_samples} samples (batched)...")
        X_test = X_test[:max_samples]
        y_test = y_test[:max_samples]
    else:
        print(f"Running verification on entire test set of {len(X_test)} samples (batched)...")

    y_pred = run_tflite(Path(args.tflite_path), X_test, batch_size=1000)
    y_true = y_test[: len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=["Normal", "AFIB"], digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=["Normal", "AFIB"], digits=4, output_dict=True)

    print("TFLite Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(report_text)

    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "tflite_path": args.tflite_path,
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        },
        cfg.REPORT_DIR / "evaluation_tflite.json",
    )
    print("Saved evaluation report to outputs/reports/evaluation_tflite.json")


if __name__ == "__main__":
    main()
