# 1A - FNN on Handwritten MNIST (SGD, 11 epochs)

# a. Import the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# b. Load the training and testing data
train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

x_train = train.drop("label", axis=1).values / 255.0   # pixels 0-255 -> 0-1
y_train = train["label"].values
x_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

# c. Define the network architecture using Keras
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(784,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")      # 10 digits (0-9)
])
model.summary()

# d. Train the model using SGD with 11 epochs
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h = model.fit(x_train, y_train, epochs=11, validation_data=(x_test, y_test))

# e. Evaluate the network
loss, acc = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# f. Plot the training loss and accuracy
plt.plot(h.history["loss"], label="train loss")
plt.plot(h.history["val_loss"], label="val loss")
plt.plot(h.history["accuracy"], label="train acc")
plt.plot(h.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.legend()
plt.show()
