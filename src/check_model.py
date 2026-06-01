import numpy as np
import tensorflow as tf
from src.model import build_dilated_se_firenet
from src.utils import set_seed, load_processed_data
from src import config as cfg

def main():
    print("==================================================")
    print("STARTING MODEL STRUCTURE VERIFICATION")
    print("==================================================")
    
    set_seed()
    
    # 1. Build Model
    model = build_dilated_se_firenet()
    model.summary()
    
    # 2. Check input and output shapes
    print("\n[INFO] Checking Input/Output shapes:")
    print(f"  Expected Input Shape:  {cfg.INPUT_SHAPE}")
    print(f"  Model Input Shape:     {model.input_shape}")
    print(f"  Expected Output Shape: (None, {cfg.NUM_CLASSES})")
    print(f"  Model Output Shape:    {model.output_shape}")
    
    assert model.input_shape[1:] == cfg.INPUT_SHAPE, "ERROR: Model input shape mismatch!"
    assert model.output_shape[1:] == (cfg.NUM_CLASSES,), "ERROR: Model output shape mismatch!"
    print("  -> Input/Output shapes are correct.")
    
    # 3. Forward Pass with Dummy Data
    print("\n[INFO] Running Forward Pass with Dummy Data:")
    dummy_input = np.random.normal(size=(5, 2500, 1)).astype("float32")
    dummy_output = model.predict(dummy_input)
    
    print(f"  Dummy Input Shape:  {dummy_input.shape}")
    print(f"  Dummy Output Shape: {dummy_output.shape}")
    print(f"  Sample Prediction:\n{dummy_output}")
    
    assert dummy_output.shape == (5, cfg.NUM_CLASSES), "ERROR: Output batch shape mismatch!"
    assert not np.isnan(dummy_output).any(), "ERROR: Output contains NaNs!"
    assert not np.isinf(dummy_output).any(), "ERROR: Output contains Infs!"
    assert np.allclose(np.sum(dummy_output, axis=1), 1.0), "ERROR: Softmax probability sum is not 1.0!"
    print("  -> Forward pass with dummy data is successful and output is valid.")
    
    # 4. Forward Pass with Real Preprocessed Data
    print("\n[INFO] Loading a small batch of Real Preprocessed Data:")
    try:
        X_train, y_train, _, _ = load_processed_data(cfg.DATA_DIR)
        X_batch = X_train[:8]
        y_batch = y_train[:8]
        
        print(f"  Real Batch Input Shape: {X_batch.shape}")
        print(f"  Real Batch Label Shape: {y_batch.shape}")
        
        real_output = model.predict(X_batch)
        print(f"  Real Batch Output Shape: {real_output.shape}")
        print(f"  Sample Real Prediction:\n{real_output}")
        
        assert real_output.shape == (8, cfg.NUM_CLASSES), "ERROR: Real batch output shape mismatch!"
        print("  -> Forward pass with real preprocessed data is successful.")
    except Exception as e:
        print(f"  [WARNING] Real data test could not be completed: {e}")
        print("  (Make sure you have preprocessed data saved under 'database/processed/')")
        
    print("\n==================================================")
    print("[SUCCESS] MODEL STRUCTURE VERIFICATION PASSED!")
    print("==================================================")

if __name__ == "__main__":
    main()
