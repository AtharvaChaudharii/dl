# 3E - Anomaly (Intrusion) Detection using Autoencoder on ECG dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. LOAD + SCALE
data = pd.read_csv("ecg.csv", header=None)   # no header: 140 readings + label column
x = data.iloc[:, :-1].values                 # all columns except last
y = (data.iloc[:, -1].values == 0).astype(int)   # in file 1 = normal, so flip: 1 = anomaly, 0 = normal
x = MinMaxScaler().fit_transform(x)          # scale features 0-1

# 2. NORMAL DATA ONLY
x_normal = x[y == 0]

# 3. MODEL (Dense autoencoder: shrink -> grow back)
n = x.shape[1]
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(n,)),   # encoder
    keras.layers.Dense(16, activation="relu"),                      # bottleneck
    keras.layers.Dense(32, activation="relu"),                      # decoder
    keras.layers.Dense(n)                                           # output = input size
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# 4. TRAIN (input = output)
model.fit(x_normal, x_normal, epochs=20, batch_size=64)

# 5. DETECT (high reconstruction error = anomaly)
error = np.mean((x - model.predict(x)) ** 2, axis=1)
threshold = np.percentile(error[y == 0], 95)
y_pred = (error > threshold).astype(int)
print(classification_report(y, y_pred))
