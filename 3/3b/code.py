# 3B - Industrial Equipment Anomaly Detection (MVTec AD) using Conv Autoencoder
#
# Put one MVTec category (e.g. "bottle") in this folder like this:
#   bottle/train/good/*.png            <- only normal images
#   bottle/test/good/*.png             <- normal test images
#   bottle/test/broken_large/*.png     <- defective test images (any defect folder name)

import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

SIZE = 128
folder = "bottle"

# 1. Preprocess images: resize + normalize
def load_images(paths):
    imgs = []
    for p in paths:
        img = keras.utils.load_img(p, target_size=(SIZE, SIZE))   # resize
        imgs.append(keras.utils.img_to_array(img) / 255.0)        # normalize 0-1
    return np.array(imgs)

x_train = load_images(glob.glob(folder + "/train/good/*.png"))

test_paths = glob.glob(folder + "/test/*/*.png")
x_test = load_images(test_paths)
y_test = np.array([0 if "/good/" in p else 1 for p in test_paths])   # 0 = normal, 1 = defect

# 2. Build convolutional autoencoder (encoder -> decoder)
model = keras.Sequential([
    # encoder
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(SIZE, SIZE, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2)),
    # decoder
    keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")   # back to image
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# train only on defect-free images (input = output)
model.fit(x_train, x_train, epochs=20, batch_size=16)

# 3. Reconstruction error in pixel values -> detect anomalies
train_error = np.mean((x_train - model.predict(x_train)) ** 2, axis=(1, 2, 3))
threshold = np.percentile(train_error, 95)
print("Threshold:", threshold)

test_pred = model.predict(x_test)
test_error = np.mean((x_test - test_pred) ** 2, axis=(1, 2, 3))
y_pred = (test_error > threshold).astype(int)   # error > threshold -> defect

print("Accuracy:", np.mean(y_pred == y_test))

# show original vs reconstructed for first 3 test images
for i in range(3):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test[i]); plt.title("Defect" if y_pred[i] else "Normal"); plt.axis("off")
    plt.subplot(2, 3, i + 4)
    plt.imshow(test_pred[i]); plt.title("Reconstructed"); plt.axis("off")
plt.show()
