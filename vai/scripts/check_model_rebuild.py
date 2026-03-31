import tensorflow as tf

def build_lenet():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")

    x = tf.keras.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        name="conv1"
    )(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        name="conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(120, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(84, activation="relu", name="fc2")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lenet_mnist_dualflow")
    return model

model = build_lenet()
model.load_weights("models/lenet_mnist_best.h5")

model.summary()
print("Arquitetura reconstruída e pesos carregados com sucesso.")
