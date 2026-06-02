import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

from . import config as cfg
from .utils import ensure_input_shape, save_json

# Test records list
TEST_RECORDS = ["04126", "05091", "08215", "08405"]

def plot_confusion_matrix(cm, title, save_path):
    """
    Plot and save confusion matrix using matplotlib
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    classes = ["Normal", "AFIB"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Annotate values inside the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix plot: {save_path.name}")

def evaluate_overall(model, data_dir, report_dir):
    """
    Evaluate the model on the overall concatenated test set
    """
    print("\n[INFO] Evaluating overall model performance:")
    X_test_path = data_dir / "test" / "X_test.npy"
    y_test_path = data_dir / "test" / "y_test.npy"
    
    if not (X_test_path.exists() and y_test_path.exists()):
        raise FileNotFoundError(f"Missing overall test files at {data_dir / 'test/'}")
        
    X_test = np.load(X_test_path).astype("float32")
    y_test = np.load(y_test_path).astype("int32")
    X_test = ensure_input_shape(X_test)
    
    print(f"  Loaded overall test dataset: X shape = {X_test.shape}, y shape = {y_test.shape}")
    
    # Predictions
    probs = model.predict(X_test, batch_size=cfg.BATCH_SIZE, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])
    report_dict = classification_report(
        y_test, y_pred, target_names=["Normal", "AFIB"], digits=4, output_dict=True
    )
    
    print(f"\n  Overall Accuracy: {acc:.4f}")
    print(f"  Overall Recall (AFIB): {recall[1]:.4f}")
    print(f"  Overall Precision (AFIB): {precision[1]:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "AFIB"], digits=4))
    
    # Plot and save CM
    plot_confusion_matrix(cm, "Confusion Matrix - Overall Test Set", report_dir / "confusion_matrix_overall.png")
    
    # Save metrics in JSON
    report_path = report_dir / "evaluation_float32.json"
    save_json({
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "metrics_afib": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1_score": float(f1[1])
        }
    }, report_path)
    print(f"  Saved overall evaluation report: {report_path.name}")

def evaluate_by_patient(model, data_dir, report_dir):
    """
    Evaluate the model patient-by-patient for each record in the test set
    """
    print("\n[INFO] Evaluating model performance patient-by-patient:")
    patient_results = {}
    
    for rec in TEST_RECORDS:
        print(f"\n  Processing test record: {rec}")
        X_path = data_dir / "test" / f"X_{rec}.npy"
        y_path = data_dir / "test" / f"y_{rec}.npy"
        
        if not (X_path.exists() and y_path.exists()):
            print(f"  [WARNING] Missing test files for record {rec}, skipping.")
            continue
            
        X_rec = np.load(X_path).astype("float32")
        y_rec = np.load(y_path).astype("int32")
        X_rec = ensure_input_shape(X_rec)
        
        # Predictions
        probs = model.predict(X_rec, batch_size=cfg.BATCH_SIZE, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_rec, y_pred)
        cm = confusion_matrix(y_rec, y_pred)
        
        # In case a patient only has one class present in the window labels, handle it gracefully
        unique_labels = np.unique(y_rec)
        prec_af, rec_af, f1_af = 0.0, 0.0, 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_rec, y_pred, labels=[0, 1], zero_division=0
        )
        
        print(f"    Record {rec} - Accuracy: {acc:.4f} | Recall (AFIB): {recall[1]:.4f} | Precision (AFIB): {precision[1]:.4f}")
        
        # Plot CM
        plot_confusion_matrix(cm, f"Confusion Matrix - Record {rec}", report_dir / f"confusion_matrix_{rec}.png")
        
        # Store in dict
        report_dict = classification_report(
            y_rec, y_pred, target_names=["Normal", "AFIB"], labels=[0, 1], output_dict=True, zero_division=0
        )
        patient_results[rec] = {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "metrics_afib": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1_score": float(f1[1])
            },
            "classification_report": report_dict
        }
        
    report_path = report_dir / "evaluation_float32_patients.json"
    save_json(patient_results, report_path)
    print(f"\n  Saved patient evaluation report: {report_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.DATA_DIR))
    parser.add_argument("--model_path", type=str, default=str(cfg.CKPT_DIR / "best_dilated_se_firenet.keras"))
    parser.add_argument("--report_dir", type=str, default=str(cfg.REPORT_DIR))
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    report_dir = Path(args.report_dir)
    
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print("==================================================")
    print("STARTING DETAILED FLOAT32 MODEL EVALUATION")
    print("==================================================")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}\n"
                                f"Please train the model or place the keras file there first!")
        
    print(f"[INFO] Loading model checkpoint from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 1. Overall evaluation
    evaluate_overall(model, data_dir, report_dir)
    
    # 2. Patient-by-patient evaluation
    evaluate_by_patient(model, data_dir, report_dir)
    
    print("\n==================================================")
    print("[SUCCESS] DETAILED EVALUATION COMPLETED!")
    print("==================================================")

if __name__ == "__main__":
    main()
