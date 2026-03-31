import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/lenet_mnist_best.h5")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)

pred = model.predict(x_test[:10], verbose=0)
pred_labels = np.argmax(pred, axis=1)

print("Predições :", pred_labels.tolist())
print("Esperado  :", y_test[:10].tolist())
