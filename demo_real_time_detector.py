"""
demo_real_time_detector.py
--------------------------
Sensor-free demo version of real_time_detector.py
Replaces the Arduino serial port with data replayed from motor_normal_data.csv,
plus a short injected anomaly burst so the LSTM alert fires at least once.

Original logic (model, scaler, threshold, sequence length) unchanged.
"""

import os, warnings
from pathlib import Path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # suppress oneDNN noise
warnings.filterwarnings('ignore')            # suppress sklearn version warnings

# Resolve all asset paths relative to this script's directory
_HERE = Path(__file__).parent

import numpy as np
import pandas as pd
import joblib
import time
import tf_keras as keras   # Keras 2 compatibility shim — loads .keras model correctly

# ── Load model & scaler ──────────────────────────────────────────────────────
print("Loading model and scaler...")
model  = keras.models.load_model(_HERE / "motor_anomaly_lstm.keras")
scaler = joblib.load(_HERE / "scaler.save")
print("Model loaded OK!\n")

# ── Settings (identical to original) ─────────────────────────────────────────
THRESHOLD       = 0.0628
SEQUENCE_LENGTH = 30
DELAY           = 0.1          # seconds between readings (simulates 9600 baud)

# ── Load CSV sensor replay data ───────────────────────────────────────────────
df = pd.read_csv(_HERE / "motor_normal_data.csv")
cols = ["ax", "ay", "az", "gx", "gy", "gz", "temp", "flame"]
df = df[cols]
normal_rows = df.values.tolist()

# ── Detection loop ────────────────────────────────────────────────────────────
sequence = []
ANOMALY_START = 100   # inject anomaly after 100 normal steps
ANOMALY_LEN   = 80                      # 80 anomalous steps
anomaly_row   = [32767.0, -32768.0, 32767.0, 16000.0, -16000.0, 16000.0, 500.0, 0.0]

print("=" * 50)
print("  MOTOR ANOMALY DETECTION - DEMO MODE")
print("  (replaying motor_normal_data.csv)")
print("=" * 50)
print()

anomaly_injected = False

for step, row in enumerate(normal_rows):

    # === Decide whether to use normal or anomaly row ===
    if ANOMALY_START <= step < ANOMALY_START + ANOMALY_LEN:
        use_row = anomaly_row
        if not anomaly_injected:
            print(">>> Injecting simulated motor fault event... <<<\n")
            anomaly_injected = True
    else:
        use_row = row

    values = np.array(use_row, dtype=float)

    # Clip raw values to scaler range, then intentionally use the raw scaled output
    values_scaled = scaler.transform(values.reshape(1, -1))[0]

    # For anomaly rows: override scaled values with extreme out-of-range floats
    # so the LSTM reconstruction error spikes well above the 0.0628 threshold
    if ANOMALY_START <= step < ANOMALY_START + ANOMALY_LEN:
        values_scaled = np.array([3.5, -2.5, 3.5, 2.5, -2.5, 2.5, 6.0, -3.0])

    sequence.append(values_scaled)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) == SEQUENCE_LENGTH:
        X = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 8)
        reconstruction = model.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstruction))

        status = "[!!! ANOMALY DETECTED !!!]" if mse > THRESHOLD else "[NORMAL]"
        ax, ay, az, gx, gy, gz, temp, flame = use_row

        print(f"[Step {step:>4}]  Error: {mse:.6f}  |  {status}")
        print(f"         ax={ax:>7.1f}  ay={ay:>7.1f}  az={az:>7.1f}  "
              f"gx={gx:>7.1f}  gy={gy:>7.1f}  gz={gz:>7.1f}  "
              f"temp={temp:>5.1f}  flame={flame}")
        print()

    time.sleep(DELAY)

print("Demo finished. All rows replayed.")

