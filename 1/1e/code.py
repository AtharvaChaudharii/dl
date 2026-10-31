# 1E - FNN on Sign Language MNIST (Adam)

# a. Import the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# b. Load the training and testing data
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

x_train = train.drop("label", axis=1).values / 255.0   # pixels 0-255 -> 0-1
y_train = train["label"].values
x_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

# c. Define the network architecture using Keras
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(784,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(25, activation="softmax")      # labels 0-24 (letters A-Y)
])
model.summary()

# d. Train the model using Adam optimizer
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

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
