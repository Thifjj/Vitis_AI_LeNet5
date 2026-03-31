import os
import numpy as np
import tensorflow as tf

os.makedirs("vai/calib", exist_ok=True)

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)

x_calib = x_train[:200]

np.save("vai/calib/mnist_calib_200.npy", x_calib)
print("Salvo:", "vai/calib/mnist_calib_200.npy")
print("Shape:", x_calib.shape)
