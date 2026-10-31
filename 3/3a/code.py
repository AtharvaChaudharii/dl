# 3A - Credit Card Fraud Detection using Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

# 1. Preprocess: load data and scale features
data = pd.read_csv("creditcard.csv")
x = data.drop("Class", axis=1).values     # Class: 0 = normal, 1 = fraud
y = data["Class"].values

x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# train ONLY on normal (non-fraud) transactions
x_train_normal = x_train[y_train == 0]

# 2. Build autoencoder (encoder -> small bottleneck -> decoder)
n = x.shape[1]   # 30 features
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(n,)),   # encoder
    keras.layers.Dense(8, activation="relu"),                       # bottleneck
    keras.layers.Dense(16, activation="relu"),                      # decoder
    keras.layers.Dense(n, activation="linear")                      # output = input size
])

# 4. Compile with Optimizer, Loss and Evaluation Metric
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# input = output, because autoencoder learns to rebuild its input
h = model.fit(x_train_normal, x_train_normal, epochs=10, batch_size=256, validation_split=0.1)

# 3. Reconstruction error + threshold
train_pred = model.predict(x_train_normal)
train_error = np.mean((x_train_normal - train_pred) ** 2, axis=1)
threshold = np.percentile(train_error, 95)    # top 5% error = suspicious
print("Threshold:", threshold)

test_pred = model.predict(x_test)
test_error = np.mean((x_test - test_pred) ** 2, axis=1)
y_pred = (test_error > threshold).astype(int)  # error > threshold -> fraud

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot loss
plt.plot(h.history["loss"], label="train loss")
plt.plot(h.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
