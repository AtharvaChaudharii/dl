# 3D - Fault Detection in NASA Turbofan Engine Sensor Data using LSTM Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. Preprocess: scale sensor readings
data = pd.read_csv("NASA TurboFan Engine.csv")
features = data.drop(["unit", "cycle", "RUL", "Fault_Label"], axis=1).values
labels = data["Fault_Label"].values                    # 0 = healthy, 1 = faulty

features = MinMaxScaler().fit_transform(features)      # scale 0-1

# make sequences of 10 readings for the LSTM
WINDOW = 10
x, y = [], []
for i in range(len(features) - WINDOW):
    x.append(features[i:i + WINDOW])
    y.append(labels[i + WINDOW - 1])                   # label of last reading
x, y = np.array(x), np.array(y)

x_healthy = x[y == 0]                                  # train only on healthy data

# 2. LSTM autoencoder
n = x.shape[2]   # number of sensors/settings
model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(WINDOW, n)),             # encoder
    keras.layers.RepeatVector(WINDOW),                          # repeat for each time step
    keras.layers.LSTM(32, return_sequences=True),               # decoder
    keras.layers.TimeDistributed(keras.layers.Dense(n))         # rebuild readings
])
model.compile(optimizer="adam", loss="mse")
model.summary()

h = model.fit(x_healthy, x_healthy, epochs=10, batch_size=64, validation_split=0.1)

# 3. Reconstruction error + threshold -> faults
train_error = np.mean((x_healthy - model.predict(x_healthy)) ** 2, axis=(1, 2))
threshold = np.percentile(train_error, 95)
print("Threshold:", threshold)

error = np.mean((x - model.predict(x)) ** 2, axis=(1, 2))
y_pred = (error > threshold).astype(int)               # error > threshold -> fault

print(classification_report(y, y_pred))

# plot error with threshold line
plt.plot(error, label="reconstruction error")
plt.axhline(threshold, color="r", label="threshold")
plt.legend()
plt.show()
