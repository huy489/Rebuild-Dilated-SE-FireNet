import os
import numpy as np
from pathlib import Path
import scipy
from scipy.signal import butter, sosfilt
import wfdb
from wfdb import processing
import time

# Configuration
TARGET_FS = 250
WINDOW_SECONDS = 10
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS
STEP_SIZE = int(WINDOW_SIZE * 0.2)
LOW_CUT = 0.5
HIGH_CUT = 40.0
FILTER_ORDER = 4

LABEL_NORMAL = 0
LABEL_AFIB = 1
LABEL_MAP = {
    "(N": LABEL_NORMAL,
    "(AFIB": LABEL_AFIB,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AFDB_DIR = PROJECT_ROOT / "database" / "physionet.org" / "files" / "afdb" / "1.0.0"
DATA_DIR = PROJECT_ROOT / "database" / "processed"

# Patient splits
TRAIN_RECORDS = [
    "04043", "04936", "07162", "07859", "07879", "08455", "06426", "05121",
    "06995", "05261", "06453", "04015", "04908", "04048", "08434", "08378",
]
VAL_RECORDS = ["04746", "07910", "08219"]
EXCLUDED_RECORDS = {"00735", "03665"}

def read_raw_record(rec_name, raw_dir=RAW_AFDB_DIR):
    rec_path = raw_dir / rec_name
    record = wfdb.rdrecord(str(rec_path), channels=[0])
    annotation = wfdb.rdann(str(rec_path), "atr")
    return record.p_signal[:, 0], annotation, int(record.fs)

def create_continuous_label_mask(annotation, signal_length, original_fs):
    mask = np.full(signal_length, -1, dtype=np.int8)
    ratio = TARGET_FS / float(original_fs)

    rhythm_samples = []
    rhythm_notes = []
    for i, raw_note in enumerate(annotation.aux_note):
        note = raw_note.strip("\x00").strip()
        if note.startswith("("):
            rhythm_samples.append(annotation.sample[i])
            rhythm_notes.append(note)
            
    if not rhythm_samples:
        return mask

    for i in range(len(rhythm_samples)):
        start = rhythm_samples[i]
        end = rhythm_samples[i + 1] if i < len(rhythm_samples) - 1 else int(signal_length / ratio)

        if int(original_fs) != TARGET_FS:
            start = int(start * ratio)
            end = int(end * ratio)

        start = max(0, min(start, signal_length))
        end = max(0, min(end, signal_length))

        note = rhythm_notes[i]
        if note in LABEL_MAP:
            mask[start:end] = LABEL_MAP[note]
        else:
            mask[start:end] = -1

    return mask

def preprocess_signal(signal_data):
    sig = signal_data.astype("float32")
    sig = sig - np.mean(sig)

    nyquist = 0.5 * TARGET_FS
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    sos = butter(FILTER_ORDER, [low, high], btype="band", output="sos")
    sig = sosfilt(sos, sig).astype("float32")
    return sig

def has_flatline(segment, duration_threshold=0.2, std_threshold=0.005):
    chunk_len = int(duration_threshold * TARGET_FS)
    if chunk_len <= 0 or chunk_len >= len(segment):
        return False

    n = len(segment) // chunk_len
    for i in range(n):
        chunk = segment[i * chunk_len : (i + 1) * chunk_len]
        if np.std(chunk) < std_threshold:
            return True
    return False

def validate_sinus_rhythm(qrs_in_window):
    if len(qrs_in_window) < 5:
        return False

    rr = np.diff(qrs_in_window)
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)

    if mean_rr <= 0:
        return False

    cv = std_rr / mean_rr
    if cv > 0.15:
        return False

    lower = 0.8 * mean_rr
    upper = 1.2 * mean_rr
    in_range = (rr >= lower) & (rr <= upper)
    if np.mean(in_range) < 0.85:
        return False

    return True

def validate_afib_quality(qrs_in_window):
    return len(qrs_in_window) >= 5

def process_single_record(rec_name, raw_dir=RAW_AFDB_DIR):
    try:
        raw_sig, annotation, fs = read_raw_record(rec_name, raw_dir)
    except Exception as e:
        print(f"Error reading record {rec_name}: {e}")
        return np.array([]), np.array([])
        
    clean_sig = preprocess_signal(raw_sig)
    mask = create_continuous_label_mask(annotation, len(clean_sig), fs)

    try:
        qrs_all = np.array(processing.xqrs_detect(sig=clean_sig, fs=TARGET_FS, verbose=False))
    except Exception as e:
        print(f"Warning: QRS detection failed for record {rec_name}: {e}")
        qrs_all = np.array([])

    X_rec, y_rec = [], []
    start_offset = 2 * TARGET_FS
    end_limit = len(clean_sig) - WINDOW_SIZE

    for start_idx in range(start_offset, end_limit, STEP_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        seg = clean_sig[start_idx:end_idx]
        m = mask[start_idx:end_idx]

        if has_flatline(seg):
            continue

        valid_ratio = np.mean(m != -1)
        if valid_ratio < 0.8:
            continue

        afib_ratio = np.mean(m == LABEL_AFIB)
        normal_ratio = np.mean(m == LABEL_NORMAL)

        if afib_ratio >= 0.5:
            label = LABEL_AFIB
        elif normal_ratio >= 0.6:
            label = LABEL_NORMAL
        else:
            continue

        if len(qrs_all) > 0:
            qrs_in_win = qrs_all[(qrs_all >= start_idx) & (qrs_all < end_idx)]
        else:
            qrs_in_win = []

        if label == LABEL_NORMAL:
            if not validate_sinus_rhythm(qrs_in_win):
                continue
        else:
            if not validate_afib_quality(qrs_in_win):
                continue

        mean = np.mean(seg)
        std = np.std(seg)
        seg_norm = (seg - mean) / max(std, 0.05)
        seg_norm = np.clip(seg_norm, -5.0, 5.0)

        X_rec.append(seg_norm[:, None])
        y_rec.append(label)

    X_rec = np.array(X_rec, dtype="float32")
    y_rec = np.array(y_rec, dtype="int32")
    
    print(f"Processed record {rec_name}: X shape = {X_rec.shape}, y shape = {y_rec.shape}")
    return X_rec, y_rec

def process_records_list(records_list):
    X_all, y_all = [], []
    for rec in records_list:
        start_time = time.time()
        X_rec, y_rec = process_single_record(rec)
        if len(X_rec) > 0:
            X_all.append(X_rec)
            y_all.append(y_rec)
        print(f"  Finished in {time.time() - start_time:.2f} seconds")
    if X_all:
        return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)
    else:
        return np.array([]), np.array([])

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("===============================================")
    print("STARTING PROCESSING FOR TRAIN SPLIT")
    print("===============================================")
    X_train, y_train = process_records_list(TRAIN_RECORDS)
    np.save(DATA_DIR / "X_train.npy", X_train)
    np.save(DATA_DIR / "y_train.npy", y_train)
    print(f"-> Train Split Done. X_train={X_train.shape}, y_train={y_train.shape}\n")

    print("===============================================")
    print("STARTING PROCESSING FOR VAL SPLIT")
    print("===============================================")
    X_val, y_val = process_records_list(VAL_RECORDS)
    np.save(DATA_DIR / "X_val.npy", X_val)
    np.save(DATA_DIR / "y_val.npy", y_val)
    print(f"-> Val Split Done. X_val={X_val.shape}, y_val={y_val.shape}\n")

    print("===============================================")
    print("STARTING PROCESSING FOR TEST SPLIT")
    print("===============================================")
    all_hea_files = sorted(RAW_AFDB_DIR.glob("*.hea"))
    all_records = [f.stem for f in all_hea_files]
    used_records = set(TRAIN_RECORDS + VAL_RECORDS) | EXCLUDED_RECORDS
    TEST_RECORDS = [r for r in all_records if r not in used_records]
    print(f"Inferred Test Records: {TEST_RECORDS}")
    
    X_test, y_test = process_records_list(TEST_RECORDS)
    np.save(DATA_DIR / "X_test.npy", X_test)
    np.save(DATA_DIR / "y_test.npy", y_test)
    print(f"-> Test Split Done. X_test={X_test.shape}, y_test={y_test.shape}\n")

    print("===============================================")
    print("FINAL DATASET SUMMARY")
    print("===============================================")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        vals, counts = np.unique(y, return_counts=True)
        dist_dict = dict(zip(vals, counts))
        print(f"{name} distribution: Normal = {dist_dict.get(0, 0)}, AFIB = {dist_dict.get(1, 0)}")
