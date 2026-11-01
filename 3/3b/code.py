# 3B - Industrial Equipment Anomaly Detection (MVTec AD) using Conv Autoencoder
#
# Put one MVTec category (e.g. "bottle") in this folder like this:
#   bottle/train/good/*.png            <- only normal images
#   bottle/test/good/*.png             <- normal test images
#   bottle/test/broken_large/*.png     <- defective test images (any defect folder name)

import glob
import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras

# 1. LOAD + SCALE (resize to 128x128, pixels 0-1)
def load_images(paths):
    return np.array([keras.utils.img_to_array(keras.utils.load_img(p, target_size=(128, 128))) / 255.0
                     for p in paths])

test_paths = glob.glob("bottle/test/*/*.png")
x = load_images(test_paths)
y = np.array([0 if "/good/" in p else 1 for p in test_paths])    # 0 = normal, 1 = defect

# 2. NORMAL DATA ONLY (defect-free training images)
x_normal = load_images(glob.glob("bottle/train/good/*.png"))

# 3. MODEL (Conv autoencoder: Conv+Pool shrink -> Conv+UpSampling grow back)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),                                                   # encoder
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    keras.layers.UpSampling2D((2, 2)),                                                   # decoder
    keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")                # back to image
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# 4. TRAIN (input = output)
model.fit(x_normal, x_normal, epochs=20, batch_size=16)

# 5. DETECT (high reconstruction error = anomaly)
error = np.mean((x - model.predict(x)) ** 2, axis=(1, 2, 3))
threshold = np.percentile(error[y == 0], 95)
y_pred = (error > threshold).astype(int)
print(classification_report(y, y_pred))
