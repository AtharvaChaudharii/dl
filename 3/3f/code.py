# 3F - Sign Language MNIST: detect letter Y (label 24) as anomaly using Autoencoder

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. LOAD + SCALE
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")
x_train = train.drop("label", axis=1).values / 255.0
x = test.drop("label", axis=1).values / 255.0
y = (test["label"].values == 24).astype(int)     # 1 = Y (anomaly), 0 = other letters

# 2. NORMAL DATA ONLY (all letters except Y)
x_normal = x_train[train["label"].values != 24]

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
