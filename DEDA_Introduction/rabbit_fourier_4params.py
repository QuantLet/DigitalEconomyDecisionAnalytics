#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a rabbit outline with four complex Fourier parameters.

The construction follows the same idea and coefficient mapping as the
elephant example in the DEDA slides.  Four complex parameters encode eight
real Fourier coefficients.  All remaining coefficients are zero.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


# Four complex parameters for the rabbit outline.
# (The real and imaginary parts store eight Fourier coefficients.)
parameters = [
    56.0 - 36.8j,
    20.3 + 22.5j,
    -5.5 - 0.34j,
    16.7 + 61.3j,
]


def fourier(t, C):
    """Evaluate a real Fourier series encoded by complex coefficients."""
    f = np.zeros_like(t, dtype=float)
    for k in range(len(C)):
        f += C[k].real * np.cos(k * t) + C[k].imag * np.sin(k * t)
    return f


def rabbit(t, p):
    """Return the x and y coordinates of the rabbit outline."""
    if len(p) != 4:
        raise ValueError("The rabbit must be described by exactly 4 parameters.")

    npar = 6
    Cx = np.zeros(npar, dtype=complex)
    Cy = np.zeros(npar, dtype=complex)

    # This is the same sparse coefficient mapping used for the elephant.
    Cx[1] = 1j * p[0].real
    Cy[1] = p[3].imag + 1j * p[0].imag

    Cx[2] = 1j * p[1].real
    Cy[2] = 1j * p[1].imag

    Cx[3] = p[2].real
    Cy[3] = 1j * p[2].imag

    Cx[5] = p[3].real

    x = fourier(t, Cy)
    y = -fourier(t, Cx)
    return x, y


def wiggle_tail(t, x, y, phase):
    """Move only the tail section while leaving the body unchanged."""
    # The tail bump is centred near t = 2.02 for this Fourier outline.
    tail_center = 2.02

    # Circular angular distance keeps the deformation smooth and local.
    distance = np.angle(np.exp(1j * (t - tail_center)))
    tail_weight = np.exp(-0.5 * (distance / 0.32) ** 2)

    x_moving = x + 2.0 * tail_weight * np.cos(phase)
    y_moving = y + 9.0 * tail_weight * np.sin(phase)
    return x_moving, y_moving


def main():
    t = np.linspace(0, 2 * np.pi, 2000)
    x, y = rabbit(t, parameters)

    fig, ax = plt.subplots(figsize=(8, 6))
    outline, = ax.plot(x, y, color="blue", linewidth=2.8)

    # The eye is only a plotted marker, not another Fourier parameter.
    ax.plot(47, 3, "b.", markersize=7)

    ax.set_xlim(-105, 85)
    ax.set_ylim(-100, 85)
    ax.set_aspect("equal")
    ax.axis("off")

    output = Path(__file__).with_name("rabbit_fourier_4params.png")
    gif_output = Path(__file__).with_name("rabbit_wiggle_tail.gif")
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")

    def init():
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
        init_func=init,
        interval=50,
        blit=True,
        repeat=True,
    )
    animation.save(gif_output, writer=PillowWriter(fps=20))

    print(f"Saved: {output}")
    print(f"Saved: {gif_output}")
    plt.show()


if __name__ == "__main__":
    main()
