import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib, math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model.keras")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
DATA_PATH = os.path.join(ART_DIR, "data_used.csv")

assert os.path.exists(MODEL_PATH), "Run prepare_pretrained.py first (to create model)"
model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
TARGET = "value" if "value" in df.columns else "sales"

LOOKBACK = 48
HORIZON = 12
TEST_RATIO = 0.2

values = df[[TARGET]].astype(float).values
split = int(len(values)*(1-TEST_RATIO))
test = scaler.transform(values[split:])

def make_windows(arr, lookback, horizon):
    X, y = [], []
    for i in range(len(arr)-lookback-horizon+1):
        X.append(arr[i:i+lookback])
        y.append(arr[i+lookback:i+lookback+horizon,0])
    return np.array(X), np.array(y)

Xte, yte = make_windows(test, LOOKBACK, HORIZON)
pred = model.predict(Xte, verbose=0)

def inv(seq):
    tmp = np.zeros((len(seq),1)); tmp[:,0] = seq
    return scaler.inverse_transform(tmp)[:,0]

# Convert predictions back to original scale
y_true_last = inv(yte[:,-1])
y_pred_last = inv(pred[:,-1])

# --- Metrics ---
mae = mean_absolute_error(y_true_last, y_pred_last)
rmse = math.sqrt(mean_squared_error(y_true_last, y_pred_last))
mape = np.mean(np.abs((y_true_last - y_pred_last) / np.maximum(np.abs(y_true_last), 1e-8))) * 100
r2 = r2_score(y_true_last, y_pred_last)

print("\n=== PRETRAINED DEMO RESULTS ===")
print(f"Mean Absolute Error (MAE):      {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"R² Score:                       {r2:.3f}")

# --- Graph (History + Forecast) ---
i = 0
hist_scaled = Xte[i,:,0]
fut_scaled = pred[i]
hist_inv, fut_inv = inv(hist_scaled), inv(fut_scaled)

# Combine for display
plt.figure(figsize=(8,4))
plt.plot(range(len(hist_inv)), hist_inv, label="History", color='blue')
plt.plot(range(len(hist_inv), len(hist_inv)+len(fut_inv)), fut_inv, label="Forecast", color='orange')
plt.title(f"Forecast (H={HORIZON})")
plt.xlabel("Time Steps")
plt.ylabel(TARGET)
plt.legend()
plt.grid(True)

# Save + Show
os.makedirs(ART_DIR, exist_ok=True)
out_path = os.path.join(ART_DIR, "demo_forecast.png")
plt.tight_layout()
plt.savefig(out_path)
plt.show()  # <-- Opens the graph window
print(f"\nGraph saved to: {out_path}")
