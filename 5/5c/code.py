# 5C - Transfer Learning (VGG16) on Fashion MNIST

# a. Set up the environment
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

DATA = "fashion_mnist"
WEIGHTS = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"   # pre-trained weights (offline)
SIZE = 64          # resize all images to 64x64 (VGG needs at least 32x32)

# b. Loading and preprocessing data (80% train, 20% test, resized)
train_ds = keras.utils.image_dataset_from_directory(
    DATA, validation_split=0.2, subset="training", seed=1, image_size=(SIZE, SIZE), batch_size=32)
test_ds = keras.utils.image_dataset_from_directory(
    DATA, validation_split=0.2, subset="validation", seed=1, image_size=(SIZE, SIZE), batch_size=32)
class_names = train_ds.class_names
print("Classes:", len(class_names))

# c. Choose the pre-trained model (VGG16 without its top classifier)
base = VGG16(weights=WEIGHTS, include_top=False, input_shape=(SIZE, SIZE, 3))

# d. Freeze the base layers and add custom layers
base.trainable = False
model = keras.Sequential([
    keras.Input(shape=(SIZE, SIZE, 3)),
    keras.layers.Lambda(preprocess_input),          # VGG style preprocessing
    base,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# e. Training
model.fit(train_ds, epochs=5, validation_data=test_ds)

# f. Compute loss  &  g. Evaluate the model
loss, acc = model.evaluate(test_ds)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# h. Visualize the prediction
images, labels = next(iter(test_ds))
pred = np.argmax(model.predict(images), axis=1)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title("T:" + class_names[labels[i]] + "\nP:" + class_names[pred[i]], fontsize=8)
    plt.axis("off")
plt.show()

# i. Fine tune: unfreeze the top 4 layers of VGG16 and retrain with small learning rate
base.trainable = True
for layer in base.layers[:-4]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=3, validation_data=test_ds)

loss, acc = model.evaluate(test_ds)
print("After Fine Tuning - Test Loss:", loss)
print("After Fine Tuning - Test Accuracy:", acc)
