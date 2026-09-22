"""Create an MNIST handwritten digit string and predict it with the trained CNN."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model


OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "cnn_model.h5"
STRING_IMAGE_PATH = OUTPUT_DIR / "handwritten_number_string.png"
RESULT_PATH = OUTPUT_DIR / "handwritten_number_prediction.png"
TARGET = "314159"
GAP = 8


def main():
    (_, _), (x_test, y_test) = mnist.load_data()
    model = load_model(MODEL_PATH)

    # Select the first test-set handwriting sample for each requested digit.
    digit_images = []
    for character in TARGET:
        digit = int(character)
        index = int(np.flatnonzero(y_test == digit)[0])
        digit_images.append(x_test[index])

    # Construct one image containing the handwritten number string.
    height = 28
    width = len(digit_images) * 28 + (len(digit_images) - 1) * GAP
    number_string = np.zeros((height, width), dtype=np.uint8)
    slices = []
    x_offset = 0
    for digit_image in digit_images:
        number_string[:, x_offset : x_offset + 28] = digit_image
        slices.append(number_string[:, x_offset : x_offset + 28])
        x_offset += 28 + GAP

    # The known gaps make segmentation deterministic for this teaching example.
    model_input = np.stack(slices).astype("float32") / 255.0
    model_input = model_input[..., np.newaxis]
    probabilities = model.predict(model_input, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    predicted_string = "".join(str(value) for value in predictions)

    # Save the raw handwriting with a transparent background.
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = number_string
    plt.imsave(STRING_IMAGE_PATH, rgba)

    # Save an annotated, transparent result plot.
    fig, axes = plt.subplots(
        2,
        len(digit_images),
        figsize=(12, 4),
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    for axis in axes.flat:
        axis.set_facecolor("none")
        axis.axis("off")

    axes[0, 0].imshow(number_string, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f'Input string: "{TARGET}"', loc="left")
    for axis in axes[0, 1:]:
        axis.set_visible(False)

    for index, (digit_image, prediction, confidence) in enumerate(
        zip(digit_images, predictions, confidences)
    ):
        axes[1, index].imshow(digit_image, cmap="gray", vmin=0, vmax=255)
        axes[1, index].set_title(f"{prediction}\n{confidence:.1%}")

    fig.suptitle(f'CNN prediction: "{predicted_string}"')
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(RESULT_PATH, dpi=240, transparent=True, bbox_inches="tight")
    plt.close(fig)

    print(f"Input: {TARGET}")
    print(f"Prediction: {predicted_string}")
    print("Confidences:", ", ".join(f"{value:.1%}" for value in confidences))
    print(f"String image: {STRING_IMAGE_PATH.resolve()}")
    print(f"Result plot: {RESULT_PATH.resolve()}")


if __name__ == "__main__":
    main()
