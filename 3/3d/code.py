# 3D - Fault Detection in NASA Turbofan Engine Sensor Data using LSTM Autoencoder

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. LOAD + SCALE
data = pd.read_csv("NASA TurboFan Engine.csv")
features = data.drop(["unit", "cycle", "RUL", "Fault_Label"], axis=1).values
labels = data["Fault_Label"].values                          # 0 = healthy, 1 = faulty
features = MinMaxScaler().fit_transform(features)            # scale 0-1

# make time windows of 10 rows (LSTM needs sequences)
x, y = [], []
for i in range(len(features) - 10):
    x.append(features[i:i + 10])
    y.append(labels[i + 9])
x, y = np.array(x), np.array(y)

# 2. NORMAL DATA ONLY (healthy engines)
x_normal = x[y == 0]

# 3. MODEL (LSTM autoencoder)
n = x.shape[2]
model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(10, n)),                 # encoder
    keras.layers.RepeatVector(10),
    keras.layers.LSTM(32, return_sequences=True),               # decoder
    keras.layers.TimeDistributed(keras.layers.Dense(n))
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# 4. TRAIN (input = output)
model.fit(x_normal, x_normal, epochs=10, batch_size=64)

# 5. DETECT (high reconstruction error = anomaly)
error = np.mean((x - model.predict(x)) ** 2, axis=(1, 2))
threshold = np.percentile(error[y == 0], 95)
y_pred = (error > threshold).astype(int)
print(classification_report(y, y_pred))
