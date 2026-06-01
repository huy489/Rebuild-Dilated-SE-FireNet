import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers
from src.model import build_dilated_se_firenet
from src.utils import (
    set_seed,
    setup_gpu_memory_growth,
    load_processed_data,
    to_one_hot,
    get_class_weights,
)
from src import config as cfg

def augment_ecg(x, y):
    """
    Apply random data augmentation on the CPU/TensorFlow data graph.
    This replaces having an augmentation layer inside the model to maintain
    100% compatibility with TFLite INT8 serialization.
    """
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.03, dtype=tf.float32)
    amp = tf.random.uniform([], minval=0.9, maxval=1.1, dtype=tf.float32)
    offset = tf.random.uniform([], minval=-0.05, maxval=0.05, dtype=tf.float32)
    x_aug = x * amp + noise + offset
    return x_aug, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.DATA_DIR))
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.INITIAL_LR)
    args = parser.parse_args()

    set_seed()
    setup_gpu_memory_growth()

    # 1. Load preprocessed datasets
    X_train, y_train, X_val, y_val = load_processed_data(Path(args.data_dir))

    # Convert labels to one-hot encoding for CategoricalCrossentropy
    y_train_oh = to_one_hot(y_train)
    y_val_oh = to_one_hot(y_val)

    # 2. Build tf.data pipeline with dynamic augmentation
    print("\n[INFO] Setting up tf.data dataset pipeline:")
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
        .shuffle(buffer_size=10000, seed=cfg.SEED, reshuffle_each_iteration=True)
        .map(augment_ecg, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("  -> tf.data pipeline set up successfully.")

    # 3. Build Model
    model = build_dilated_se_firenet()
    model.summary()

    # 4. Learning Rate Schedule and Optimizer
    steps_per_epoch = max(len(X_train) // args.batch_size, 1)
    decay_steps = steps_per_epoch * args.epochs
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=decay_steps,
        alpha=0.05,
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    # 5. Compile Model
    model.compile(
        optimizer=optimizer,
        loss=losses.CategoricalCrossentropy(label_smoothing=cfg.LABEL_SMOOTHING),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    # Compute class weights to handle imbalance
    class_weight = get_class_weights(y_train)
    print("\nClass weights configured:", class_weight)

    # 6. Callbacks
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
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

    # 7. Fit Model
    print(f"\n[INFO] Starting training for {args.epochs} epochs with batch size {args.batch_size}:")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight,
    )

    # Save final model
    final_path = cfg.CKPT_DIR / "final_dilated_se_firenet.keras"
    model.save(final_path)
    print(f"\n[SUCCESS] Saved final model to: {final_path}")
    print(f"[SUCCESS] Best model checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    main()
