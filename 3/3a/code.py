# 3A - Credit Card Fraud Detection using Autoencoder

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. LOAD + SCALE
data = pd.read_csv("creditcard.csv")
x = data.drop("Class", axis=1).values
y = data["Class"].values                     # 0 = normal, 1 = fraud
x = MinMaxScaler().fit_transform(x)          # scale features 0-1

# 2. NORMAL DATA ONLY (non-fraud)
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
model.fit(x_normal, x_normal, epochs=10, batch_size=256)

# 5. DETECT (high reconstruction error = anomaly)
error = np.mean((x - model.predict(x)) ** 2, axis=1)
threshold = np.percentile(error[y == 0], 95)
y_pred = (error > threshold).astype(int)
print(classification_report(y, y_pred))
