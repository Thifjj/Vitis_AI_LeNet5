import tensorflow as tf

model = tf.keras.models.load_model("models/lenet_mnist_best.h5")
model.summary()
print("Modelo carregado com sucesso.")
