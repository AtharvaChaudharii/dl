# 3F - Sign Language MNIST: detect letter Y (label 24) as anomaly using Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

ANOMALY = 24   # letter Y

# load data and normalize pixels 0-255 -> 0-1
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

x_train = train.drop("label", axis=1).values / 255.0
y_train = train["label"].values
x_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

# train ONLY on the other 23 classes (remove Y)
x_train_normal = x_train[y_train != ANOMALY]

# test labels: 1 = anomaly (Y), 0 = normal
y_true = (y_test == ANOMALY).astype(int)

# Autoencoder: encoder -> bottleneck -> decoder
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(784,)),   # encoder
    keras.layers.Dense(32, activation="relu"),                         # bottleneck
    keras.layers.Dense(128, activation="relu"),                        # decoder
    keras.layers.Dense(784, activation="sigmoid")                      # output image
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# input = output
model.fit(x_train_normal, x_train_normal, epochs=10, batch_size=256, validation_split=0.1)

# reconstruction error + threshold
train_error = np.mean((x_train_normal - model.predict(x_train_normal)) ** 2, axis=1)
threshold = np.percentile(train_error, 95)
print("Threshold:", threshold)

test_pred = model.predict(x_test)
test_error = np.mean((x_test - test_pred) ** 2, axis=1)
y_pred = (test_error > threshold).astype(int)   # error > threshold -> anomaly

print("Avg error normal :", test_error[y_true == 0].mean())
print("Avg error Y      :", test_error[y_true == 1].mean())
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# show a Y image and its reconstruction
i = np.where(y_true == 1)[0][0]
plt.subplot(1, 2, 1); plt.imshow(x_test[i].reshape(28, 28), cmap="gray"); plt.title("Original Y")
plt.subplot(1, 2, 2); plt.imshow(test_pred[i].reshape(28, 28), cmap="gray"); plt.title("Reconstructed")
plt.show()
