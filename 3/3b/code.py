# 3B - Anomaly Detection using Convolutional Autoencoder on ECG dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. LOAD + SCALE
data = pd.read_csv("ecg.csv", header=None)   # no header: 140 readings + label column
x = data.iloc[:, :-1].values                 # all columns except last
y = (data.iloc[:, -1].values == 0).astype(int)   # in file 1 = normal, so flip: 1 = anomaly, 0 = normal
x = MinMaxScaler().fit_transform(x)          # scale features 0-1 (normalize)
x = x.reshape(-1, 140, 1)                    # Conv needs shape (length, channels)

# 2. NORMAL DATA ONLY
x_normal = x[y == 0]

# 3. MODEL (Conv autoencoder: Conv+Pool shrink -> Conv+UpSampling grow back)
model = keras.Sequential([
    keras.layers.Conv1D(32, 3, activation="relu", padding="same", input_shape=(140, 1)),
    keras.layers.MaxPooling1D(2),                                            # encoder
    keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
    keras.layers.UpSampling1D(2),                                            # decoder
    keras.layers.Conv1D(1, 3, padding="same")                               # back to signal
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# 4. TRAIN (input = output)
model.fit(x_normal, x_normal, epochs=20, batch_size=64)

# 5. DETECT (high reconstruction error = anomaly)
error = np.mean((x - model.predict(x)) ** 2, axis=(1, 2))
threshold = np.percentile(error[y == 0], 95)
y_pred = (error > threshold).astype(int)
print(classification_report(y, y_pred))
