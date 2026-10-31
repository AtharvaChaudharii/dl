# 2B - CNN on Handwritten MNIST

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# a. Loading and preprocessing the image data
train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

x_train = train.drop("label", axis=1).values / 255.0   # pixels 0-255 -> 0-1
y_train = train["label"].values
x_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

# CNN needs images, so reshape 784 -> 28x28 with 1 channel (grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# b. Defining the model's architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")      # 10 digits (0-9)
])
model.summary()

# c. Training the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# d. Estimating the model's performance
loss, acc = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

plt.plot(h.history["accuracy"], label="train acc")
plt.plot(h.history["val_accuracy"], label="val acc")
plt.plot(h.history["loss"], label="train loss")
plt.plot(h.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
