#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw a Fourier rabbit and create a transparent tail-wiggle GIF."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageSequence


# First four parameters describe the rabbit outline.
# The fifth parameter describes the eye position.
parameters = [
    56.0 - 36.8j,
    20.3 + 22.5j,
    -5.5 - 0.34j,
    16.7 + 61.3j,
    47.0 + 3.0j,
]


def fourier(t, coefficients):
    """Evaluate a real Fourier series."""
    values = np.zeros_like(t, dtype=float)

    for k, coefficient in enumerate(coefficients):
        values += (
            coefficient.real * np.cos(k * t)
            + coefficient.imag * np.sin(k * t)
        )

    return values


def rabbit(t, rabbit_parameters):
    """Return the x and y coordinates of the rabbit outline."""
    if len(rabbit_parameters) != 5:
        raise ValueError("The rabbit must be described by exactly 5 parameters.")

    number_of_parameters = 6
    coefficients_x = np.zeros(number_of_parameters, dtype=complex)
    coefficients_y = np.zeros(number_of_parameters, dtype=complex)

    # x(t)
    coefficients_x[1] = rabbit_parameters[3].imag + 1j * rabbit_parameters[0].imag
    coefficients_x[2] = 1j * rabbit_parameters[1].imag
    coefficients_x[3] = 1j * rabbit_parameters[2].imag

    # y(t)
    coefficients_y[1] = -1j * rabbit_parameters[0].real
    coefficients_y[2] = -1j * rabbit_parameters[1].real
    coefficients_y[3] = -rabbit_parameters[2].real
    coefficients_y[5] = -rabbit_parameters[3].real

    x = fourier(t, coefficients_x)
    y = fourier(t, coefficients_y)

    return x, y


def wiggle_tail(t, x, y, phase):
    """Move the tail section while leaving the rest nearly unchanged."""
    tail_center = 2.02
    distance = np.angle(np.exp(1j * (t - tail_center)))
    tail_weight = np.exp(-0.5 * (distance / 0.32) ** 2)

    x_moving = x + 2.0 * tail_weight * np.cos(phase)
    y_moving = y + 9.0 * tail_weight * np.sin(phase)

    return x_moving, y_moving


def make_gif_transparent(input_path, output_path, threshold=245):
    """Make near-white pixels transparent in every GIF frame."""
    source = Image.open(input_path)
    transparent_frames = []
    durations = []

    for frame in ImageSequence.Iterator(source):
        rgba = frame.convert("RGBA")
        pixels = list(rgba.getdata())
        transparent_mask = [
            red >= threshold and green >= threshold and blue >= threshold
            for red, green, blue, _ in pixels
        ]

        # GIF supports 256 palette entries. Reserve index 255 for transparency.
        indexed = rgba.convert("RGB").quantize(
            colors=255,
            method=Image.Quantize.MEDIANCUT,
        )
        indices = bytearray(indexed.tobytes())

        for index, is_transparent in enumerate(transparent_mask):
            if is_transparent:
                indices[index] = 255

        transparent_frame = Image.frombytes("P", indexed.size, bytes(indices))
        palette = indexed.getpalette()[:765]
        palette.extend([255, 255, 255])
        transparent_frame.putpalette(palette)
        transparent_frame.info["transparency"] = 255
        transparent_frame.info["disposal"] = 2

        transparent_frames.append(transparent_frame)
        durations.append(
            frame.info.get("duration", source.info.get("duration", 50))
        )

    if not transparent_frames:
        raise ValueError("The generated GIF contains no frames.")

    transparent_frames[0].save(
        output_path,
        save_all=True,
        append_images=transparent_frames[1:],
        duration=durations,
        loop=source.info.get("loop", 0),
        transparency=255,
        disposal=2,
        optimize=False,
    )


def main():
    t = np.linspace(0, 2 * np.pi, 2000)
    x, y = rabbit(t, parameters)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    outline, = ax.plot(x, y, color="blue", linewidth=2.8)

    eye = parameters[4]
    ax.plot(eye.real, eye.imag, "b.", markersize=7)

    ax.set_xlim(-105, 85)
    ax.set_ylim(-100, 85)
    ax.set_aspect("equal")
    ax.axis("off")

    output_directory = Path(__file__).resolve().parent
    png_output = output_directory / "rabbit_fourier_4params.png"
    temporary_gif = output_directory / "rabbit_wiggle_tail_temporary.gif"
    gif_output = output_directory / "rabbit_wiggle_tail_transparent.gif"

    fig.savefig(
        png_output,
        dpi=180,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
    )

    def initialize():
        outline.set_data(x, y)
        return (outline,)

    def update(frame):
        phase = 2 * np.pi * frame / 80
        x_frame, y_frame = wiggle_tail(t, x, y, phase)
        outline.set_data(x_frame, y_frame)
        return (outline,)

    animation = FuncAnimation(
        fig,
        update,
        frames=80,
        init_func=initialize,
        interval=50,
        blit=True,
        repeat=True,
    )

    # Force the temporary animation to be fully opaque. This avoids a bug in
    # Matplotlib 3.10.0's handling of transparent PillowWriter frames.
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1)
    ax.set_facecolor("white")

    animation.save(
        temporary_gif,
        writer=PillowWriter(fps=20),
        # Matplotlib 3.10.0 has a PillowWriter bug when it receives an RGBA
        # frame. Render an opaque temporary GIF and make it transparent in
        # the reliable Pillow post-processing step below.
        savefig_kwargs={"transparent": False, "facecolor": "white"},
    )

    make_gif_transparent(temporary_gif, gif_output, threshold=245)
    temporary_gif.unlink()

    print(f"Saved transparent PNG: {png_output}")
    print(f"Saved transparent GIF: {gif_output}")

    plt.show()


if __name__ == "__main__":
    main()
