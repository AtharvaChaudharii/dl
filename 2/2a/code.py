# 2A - CNN on CIFAR10

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras

# a. Loading and preprocessing the image data
# only one csv file is given, so we split it into train and test (80/20)
data = pd.read_csv("cifar-10.csv")
x = data.drop("label", axis=1).values / 255.0   # pixels 0-255 -> 0-1
y = data["label"].values
# csv stores all red, then green, then blue -> make it 32x32 image with 3 channels
x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# b. Defining the model's architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")      # 10 classes
])
model.summary()

# c. Training the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

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
