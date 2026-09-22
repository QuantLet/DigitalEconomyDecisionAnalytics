"""Train and evaluate a convolutional neural network on MNIST."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import load_model
from keras.utils import to_categorical


RANDOM_SEED = 123
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 12
OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_PATH = OUTPUT_DIR / "training_process_plot.png"
MODEL_PATH = OUTPUT_DIR / "cnn_model.keras"


def plot_training_process(model_history, epochs):
    """Plot training and validation accuracy and loss."""
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), model_history.history["accuracy"], "blue")
    plt.plot(range(1, epochs + 1), model_history.history["val_accuracy"], "red")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy", fontsize=15)
    plt.yticks(fontsize=18)
    plt.xlabel("Epoch", fontsize=15)
    plt.xticks(np.arange(1, epochs + 1), fontsize=15)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), model_history.history["loss"], "blue")
    plt.plot(range(1, epochs + 1), model_history.history["val_loss"], "red")
    plt.title("Model Loss")
    plt.ylabel("Loss", fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    plt.xticks(np.arange(1, epochs + 1), fontsize=15)

    fig.tight_layout()
    return fig


def build_model(input_shape):
    """Build and compile the MNIST CNN."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), padding="valid"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="valid"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.5),
            Dense(NUM_CLASSES),
            BatchNormalization(),
            Activation("softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # Downloads MNIST to the Keras dataset cache on the first run.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_sample_size, row_size, col_size = x_train.shape
    test_sample_size = x_test.shape[0]
    print(
        f"Total Sample Size: {train_sample_size + test_sample_size}, "
        f"Training Sample Size: {train_sample_size}, "
        f"Testing Sample Size: {test_sample_size}"
    )
    print(f"Row pixels: {row_size}, column pixels: {col_size}")

    np.random.seed(RANDOM_SEED)
    input_shape = (row_size, col_size, 1)

    x_train = x_train.reshape(train_sample_size, *input_shape).astype("float32") / 255.0
    x_test = x_test.reshape(test_sample_size, *input_shape).astype("float32") / 255.0
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    model = build_model(input_shape)

    start_time = time.time()
    model_history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    elapsed_minutes = (time.time() - start_time) / 60

    process_plot = plot_training_process(model_history, EPOCHS)
    process_plot.savefig(PLOT_PATH, dpi=300, transparent=True)
    plt.close(process_plot)

    validation_accuracy = model_history.history["val_accuracy"][-1]
    print(f"Training took {elapsed_minutes:.1f} minutes")
    print(f"Final validation accuracy: {validation_accuracy:.2%}")

    model.save(MODEL_PATH)

    loaded_model = load_model(MODEL_PATH)
    _, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Saved training plot to: {PLOT_PATH}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
