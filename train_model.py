
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

BASE = Path(__file__).parent
PROCESSED = BASE / "processed_data"

CLASSES = [
    "apple",
    "cups",
    "oranges",
    "paper_bags",
    "water_bottle"
]

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 30

# Load preprocessed NumPy arrays
print("Loading preprocessed images...")

X_train = np.load(PROCESSED / "X_train.npy")
y_train = np.load(PROCESSED / "y_train.npy")

X_val = np.load(PROCESSED / "X_val.npy")
y_val = np.load(PROCESSED / "y_val.npy")

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)

# Create TensorFlow datasets
trainData = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

trainData = (
    trainData
    .shuffle(len(X_train), seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

valData = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(160, 160, 3)),

    # Images are already normalized to [0, 1]
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Stop when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    trainData,
    validation_data=valData,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Plot accuracy
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(BASE / "accuracy_graph.png", dpi=300)
plt.close()

# Plot loss
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(BASE / "loss_graph.png", dpi=300)
plt.close()

# Save trained model and class names
model.save(BASE / "object_model.keras")

with open(BASE / "class_names.txt", "w") as file:
    file.write("\n".join(CLASSES))

print("Training completed!")
