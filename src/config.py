from pathlib import Path

# ============================================================
# PATH CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "database" / "processed"
RAW_AFDB_DIR = PROJECT_ROOT / "database" / "physionet.org" / "files" / "afdb" / "1.0.0"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"
TFLITE_DIR = OUTPUT_DIR / "tflite"

# Ensure directories exist
for p in [DATA_DIR, CKPT_DIR, REPORT_DIR, TFLITE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# ECG CONFIG
# ============================================================
TARGET_FS = 250
WINDOW_SECONDS = 10
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS  # 2500
STEP_SIZE = int(WINDOW_SIZE * 0.2)        # 500
INPUT_SHAPE = (WINDOW_SIZE, 1)

# ============================================================
# LABEL CONFIG
# ============================================================
LABEL_NORMAL = 0
LABEL_AFIB = 1
NUM_CLASSES = 2

# ============================================================
# MODEL CONFIG
# ============================================================
L2_RATE = 1e-3
LEAKY_RELU_ALPHA = 0.1
SE_RATIO = 4
DROPOUT_CLASSIFIER = 0.2
SPATIAL_DROPOUT = 0.2
STEM_FILTERS = 12

# Format: (name, squeeze_filters, expand_filters, dilation_rate, output_channels)
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
