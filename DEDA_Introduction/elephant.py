#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:20:37 2026

@author: haerdle
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:41:18 2026

@author: haerdle
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Elephant parameters
# last parameter controls trunk wiggle amplitude and eye position
p = [50 - 30j, 18 + 8j, 12 - 10j, -14 - 60j, 20 + 20j]


def fourier(t, C):
    f = np.zeros_like(t, dtype=float)
    for k in range(len(C)):
        f += C[k].real * np.cos(k * t) + C[k].imag * np.sin(k * t)
    return f


def elephant(t, p):
    npar = 6
    Cx = np.zeros(npar, dtype=complex)
    Cy = np.zeros(npar, dtype=complex)

    Cx[1] = 1j * p[0].real
    Cy[1] = p[3].imag + 1j * p[0].imag

    Cx[2] = 1j * p[1].real
    Cy[2] = 1j * p[1].imag

    Cx[3] = p[2].real
    Cy[3] = 1j * p[2].imag

    Cx[5] = p[3].real

    x = np.append(fourier(t, Cy), [p[4].imag])
    y = -np.append(fourier(t, Cx), [-p[4].imag])

    return x, y


# Body and trunk parameter ranges
t_body = np.linspace(0.4 + 1.3 * np.pi, 2 * np.pi + 0.9 * np.pi, 1000)
t_trunk = np.linspace(2 * np.pi + 0.9 * np.pi, 0.4 + 3.3 * np.pi, 1000)

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-80, 100)
ax.set_ylim(-80, 100)
ax.set_aspect('equal')
ax.axis('off')

# Draw body once
xb, yb = elephant(t_body, p)
ax.plot(xb, yb, 'b.', ms=2)

# Trunk line to animate
trunk_line, = ax.plot([], [], 'b.', ms=2)


def init():
    xt, yt = elephant(t_trunk, p)
    trunk_line.set_data(xt, yt)
    return trunk_line,


def update(frame):
    xt, yt = elephant(t_trunk, p)

    # Wiggle only the trunk
    phase = frame / 8.0
    wiggle = np.sin(phase) * p[4].real

    for i in range(len(yt) - 1):
        yt[i] -= np.sin((xt[i] - xt[0]) * np.pi / len(yt)) * wiggle

    trunk_line.set_data(xt, yt)
    return trunk_line,


ani = FuncAnimation(
    fig,
    update,
    frames=120,
    init_func=init,
    interval=50,
    blit=True
)

# This creates GIF file

ani.save("elephant_wiggle.gif", writer="pillow", fps=20)