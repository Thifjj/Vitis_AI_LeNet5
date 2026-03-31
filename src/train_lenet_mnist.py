import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

MODEL_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 1. Carregar MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalização
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Adicionar canal
x_train = np.expand_dims(x_train, axis=-1)  # (N, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)

# 4. One-hot nos rótulos
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# 5. Split de validação
x_val = x_train[-5000:]
y_val_cat = y_train_cat[-5000:]
x_train_small = x_train[:-5000]
y_train_small_cat = y_train_cat[:-5000]

# 6. Modelo LeNet-like em Functional API
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "lenet_mnist_best.h5"),
        monitor="val_accuracy",
        save_best_only=True
    )
]

history = model.fit(
    x_train_small,
    y_train_small_cat,
    validation_data=(x_val, y_val_cat),
    epochs=12,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 7. Avaliação final
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# 8. Salvar modelo final
model.save(os.path.join(MODEL_DIR, "lenet_mnist_final.h5"))

# 9. Salvar arquitetura em JSON
with open(os.path.join(MODEL_DIR, "lenet_mnist_architecture.json"), "w") as f:
    f.write(model.to_json())

# 10. Salvar métricas
metrics = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "input_shape": [28, 28, 1],
    "classes": 10
}
with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# 11. Curvas de treino
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "accuracy_curve.png"), dpi=200)

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "loss_curve.png"), dpi=200)

print("Arquivos salvos em models/ e reports/")
