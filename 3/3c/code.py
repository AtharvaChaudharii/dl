# 3C - Anomaly Detection in Energy Consumption using LSTM Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. Preprocess: fill missing values, normalize, create time windows
data = pd.read_csv("energy consumption.csv")
data = data.ffill()                                    # fill missing values

features = data.drop(["Timestamp", "Anomaly_Label"], axis=1).values
labels = data["Anomaly_Label"].values                  # 0 = normal, 1 = anomaly

features = MinMaxScaler().fit_transform(features)      # normalize 0-1

WINDOW = 10
x, y = [], []
for i in range(len(features) - WINDOW):
    x.append(features[i:i + WINDOW])                   # 10 time steps together
    y.append(labels[i + WINDOW - 1])                   # label of last step
x, y = np.array(x), np.array(y)

x_normal = x[y == 0]                                   # train only on normal data

# 2. LSTM autoencoder
n = x.shape[2]   # number of features
model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(WINDOW, n)),             # encoder
    keras.layers.RepeatVector(WINDOW),                          # repeat for each time step
    keras.layers.LSTM(32, return_sequences=True),               # decoder
    keras.layers.TimeDistributed(keras.layers.Dense(n))         # rebuild features
])
model.compile(optimizer="adam", loss="mse")
model.summary()

h = model.fit(x_normal, x_normal, epochs=10, batch_size=32, validation_split=0.1)

# 3. Threshold on reconstruction error
train_error = np.mean((x_normal - model.predict(x_normal)) ** 2, axis=(1, 2))
threshold = np.percentile(train_error, 95)
print("Threshold:", threshold)

error = np.mean((x - model.predict(x)) ** 2, axis=(1, 2))
y_pred = (error > threshold).astype(int)               # error > threshold -> anomaly

print(classification_report(y, y_pred))

# plot error with threshold line
plt.plot(error, label="reconstruction error")
plt.axhline(threshold, color="r", label="threshold")
plt.legend()
plt.show()
