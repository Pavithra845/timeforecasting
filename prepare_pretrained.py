import os, io, math, urllib.request
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED=42
np.random.seed(10); tf.random.set_seed(SEED)

def load_data():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/sunspots.csv"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.rename(columns={'Month':'date','Sunspots':'value'})
        print("Loaded dataset.")
        return df, "value"
    except:
        print("No internet, creating synthetic sales data.")
        n=1000; t=np.arange(n)
        val = 50 + 0.02*t + 10*np.sin(2*np.pi*t/30) + np.random.normal(0,2,size=n)
        df=pd.DataFrame({"date":pd.date_range("2016-01-01",periods=n,freq="D"),
                         "value":val})
        return df, "value"

df, TARGET = load_data(); DATE="date"
LOOKBACK=48; HORIZON=12; TEST_RATIO=0.2; BATCH=128; EPOCHS=8; PATIENCE=3

os.makedirs("artifacts", exist_ok=True)
plt.plot(df[DATE], df[TARGET]); plt.title("Time Series"); plt.xlabel("Date"); plt.ylabel(TARGET)
plt.tight_layout(); plt.savefig("artifacts/eda_timeseries.png"); plt.close()

values=df[[TARGET]].astype(float).values
split=int(len(values)*(1-TEST_RATIO))
scaler=StandardScaler()
train=scaler.fit_transform(values[:split]); test=scaler.transform(values[split:])

def make_windows(arr,lookback,horizon):
    X,y=[],[]
    for i in range(len(arr)-lookback-horizon+1):
        X.append(arr[i:i+lookback]); y.append(arr[i+lookback:i+lookback+horizon,0])
    return np.array(X),np.array(y)

Xtr,ytr=make_windows(train,LOOKBACK,HORIZON); Xte,yte=make_windows(test,LOOKBACK,HORIZON)

def build(shape,horizon):
    m=keras.Sequential([
        layers.Input(shape=shape),
        layers.LSTM(96,return_sequences=True),
        layers.Dropout(0.15),
        layers.LSTM(64),
        layers.Dense(horizon)
    ])
    m.compile(optimizer="adam",loss="mse"); return m

model=build(Xtr.shape[1:],HORIZON)
cb=[keras.callbacks.EarlyStopping(patience=PATIENCE,restore_best_weights=True)]
model.fit(Xtr,ytr,validation_split=0.1,epochs=EPOCHS,batch_size=BATCH,verbose=1,callbacks=cb)

pred=model.predict(Xte,verbose=0)
def inv(seq): tmp=np.zeros((len(seq),1)); tmp[:,0]=seq; return scaler.inverse_transform(tmp)[:,0]
y_true, y_pred = inv(yte[:,-1]), inv(pred[:,-1])
mae=mean_absolute_error(y_true,y_pred); rmse=math.sqrt(mean_squared_error(y_true,y_pred))
print(f"[TRAIN] MAE={mae:.3f} | RMSE={rmse:.3f}")

i=0; hist_scaled=Xte[i,:,0]; fut_scaled=pred[i]
hist_inv,fut_inv=inv(hist_scaled),inv(fut_scaled)
plt.plot(range(len(hist_inv)),hist_inv,label="History")
plt.plot(range(len(hist_inv),len(hist_inv)+len(fut_inv)),fut_inv,label="Forecast")
plt.legend(); plt.tight_layout(); plt.savefig("artifacts/forecast_sample.png"); plt.close()

model.save("artifacts/model.keras")
joblib.dump(scaler,"artifacts/scaler.pkl")
df.to_csv("artifacts/data_used.csv",index=False)
print("Saved pretrained model → artifacts/")
