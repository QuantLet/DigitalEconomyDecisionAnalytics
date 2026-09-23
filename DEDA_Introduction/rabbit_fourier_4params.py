#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw a rabbit outline with five complex parameters.

The first four complex parameters encode the Fourier coefficients
for the closed rabbit outline.

The fifth complex parameter gives the position of the eye:
    real part      -> x-coordinate
    imaginary part -> y-coordinate
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


# Five complex parameters:
# p[0] ~ p[3]: Fourier parameters for the rabbit outline
# p[4]: eye position
parameters = [
    56.0 - 36.8j,
    20.3 + 22.5j,
    -5.5 - 0.34j,
    16.7 + 61.3j,
    47.0 + 3.0j
]


def fourier(t, C):
    """
    Evaluate a real Fourier series:

        f(t) = sum_k [ Re(C[k]) cos(kt) + Im(C[k]) sin(kt) ]
    """
    f = np.zeros_like(t, dtype=float)

    for k in range(len(C)):
        f += (
            C[k].real * np.cos(k * t)
            + C[k].imag * np.sin(k * t)
        )

    return f


def rabbit(t, p):
    """
    Return the x and y coordinates of the rabbit outline.

    The first four parameters determine the Fourier curve.
    The fifth parameter gives the eye position.
    """

    if len(p) != 5:
        raise ValueError(
            "The rabbit must be described by exactly 5 parameters."
        )

    npar = 6

    Cx = np.zeros(npar, dtype=complex)
    Cy = np.zeros(npar, dtype=complex)

    # x(t)
    Cx[1] = p[3].imag + 1j * p[0].imag
    Cx[2] = 1j * p[1].imag
    Cx[3] = 1j * p[2].imag

    # y(t)
    Cy[1] = -1j * p[0].real
    Cy[2] = -1j * p[1].real
    Cy[3] = -p[2].real
    Cy[5] = -p[3].real


    x = fourier(t, Cx)
    y = fourier(t, Cy)

    return x, y


def wiggle_tail(t, x, y, phase):
    """
    Move only the tail section while leaving the rest
    of the rabbit approximately unchanged.
    """

    # Tail position along the parameter t
    tail_center = 2.02

    # Circular angular distance
    distance = np.angle(
        np.exp(1j * (t - tail_center))
    )

    # Gaussian weight:
    # points close to the tail move more,
    # points far from the tail barely move
    tail_weight = np.exp(
        -0.5 * (distance / 0.32) ** 2
    )

    # Tail movement
    x_moving = (
        x
        + 2.0 * tail_weight * np.cos(phase)
    )

    y_moving = (
        y
        + 9.0 * tail_weight * np.sin(phase)
    )

    return x_moving, y_moving


def main():

    # Parameter t runs over one full period
    t = np.linspace(
        0,
        2 * np.pi,
        2000
    )

    # ------------------------------------------------------------
    # Rabbit outline
    # ------------------------------------------------------------

    x, y = rabbit(
        t,
        parameters
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    outline, = ax.plot(
        x,
        y,
        color="blue",
        linewidth=2.8
    )

    # ------------------------------------------------------------
    # Fifth parameter -> eye position
    # ------------------------------------------------------------

    eye = parameters[4]

    eye_x = eye.real
    eye_y = eye.imag

    ax.plot(
        eye_x,
        eye_y,
        "b.",
        markersize=7
    )

    # ------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------

    ax.set_xlim(
        -105,
        85
    )

    ax.set_ylim(
        -100,
        85
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )

    # ------------------------------------------------------------
    # Save static image
    # ------------------------------------------------------------

    output = Path(__file__).with_name(
        "rabbit_fourier_4params.png"
    )

    gif_output = Path(__file__).with_name(
        "rabbit_wiggle_tail.gif"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        facecolor="white"
    )

    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------

    def init():

        outline.set_data(
            x,
            y
        )

        return (outline,)


    def update(frame):

        phase = (
            2 * np.pi
            * frame
            / 80
        )

        x_frame, y_frame = wiggle_tail(
            t,
            x,
            y,
            phase
        )

        outline.set_data(
            x_frame,
            y_frame
        )

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

    animation.save(
        gif_output,
        writer=PillowWriter(
            fps=20
        )
    )

    print(
        f"Saved: {output}"
    )

    print(
        f"Saved: {gif_output}"
    )

    plt.show()


if __name__ == "__main__":
    main()
