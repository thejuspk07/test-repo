import os, warnings
import serial
import numpy as np
import joblib
import threading
import time

# suppress oneDNN informational noise and sklearn warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# use compatibility shim for Keras 2-style .keras models
import tf_keras as keras

# Import UI components
from ui_detector import AnomalyDashboard, data_q

# load trained model
model = keras.models.load_model("motor_anomaly_lstm.keras")

# load scaler
scaler = joblib.load("scaler.save")

# anomaly threshold
threshold = 0.0628

# Initialize UI
app = AnomalyDashboard()

# Start real-time detection in a background thread
def real_time_detect():
    # connect to Arduino
    ser = serial.Serial("COM7", 9600)

    sequence = []
    step = 0

    data_q.put({"log": "Real-time monitoring started..."})

    while True:
        try:
            line = ser.readline().decode().strip()
            values = line.split(",")

            if len(values) != 8:
                continue

            values_raw = np.array(values, dtype=float)
            values_scaled = scaler.transform(np.array([values_raw]))[0]

            sequence.append(values_scaled)

            if len(sequence) > 30:
                sequence.pop(0)

            if len(sequence) == 30:
                step += 1
                X = np.array(sequence).reshape(1, 30, 8)
                reconstruction = model.predict(X, verbose=0)
                mse = float(np.mean(np.square(X - reconstruction)))
                is_anomaly = mse > threshold

                data_q.put({
                    "step": step,
                    "mse": mse,
                    "anomaly": is_anomaly,
                    "sensors": list(values_raw),
                    "log": f"[{step:>4}] MSE={mse:.6f}  {'⚠ ANOMALY' if is_anomaly else '✓ Normal'}",
                })

                if is_anomaly:
                    ser.write(b'1')
                else:
                    ser.write(b'0')

        except Exception as e:
            data_q.put({"log": f"Error: {str(e)}"})
            time.sleep(1)

thread = threading.Thread(target=real_time_detect, daemon=True)
thread.start()

# Start the UI
app.mainloop()