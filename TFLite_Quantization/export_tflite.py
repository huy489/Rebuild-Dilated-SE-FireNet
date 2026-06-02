import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src import config as cfg
from src.utils import load_processed_data


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

    # Fixed: load_processed_data returns 4 values, not 6
    X_train, _, _, _ = load_processed_data(Path(args.data_dir))
    model = tf.keras.models.load_model(args.model_path)

    cfg.TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    export_float32(model, cfg.TFLITE_DIR / "dilated_se_firenet_float32.tflite")
    export_int8(model, X_train, cfg.TFLITE_DIR / "dilated_se_firenet_int8.tflite")


if __name__ == "__main__":
    main()
