#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:39:23 2026

@author: haerdle
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:51:43 2026

@author: haerdle
"""

from matplotlib import animation
from numpy import append, cos, linspace, pi, sin, zeros
import matplotlib.pyplot as plt

parameters = [50 - 50j, 18 + 80j, 12 - 10j, -14 - 60j, 20 + 20j]

def fourier(t, C):
    f = zeros(t.shape)
    for k in range(len(C)):
        f += C.real[k] * cos(k * t) + C.imag[k] * sin(k * t)
    return f

def peashooter(t, p):
    npar = 6
    Cx = zeros((npar,), dtype='complex')
    Cy = zeros((npar,), dtype='complex')

    Cx[1] = p[0].real * 1j
    Cy[1] = p[3].imag + p[0].imag * 1j

    Cx[2] = p[1].real * 1j
    Cy[2] = p[1].imag * 1j

    Cx[3] = p[2].real
    Cy[3] = p[2].imag * 1j

    Cx[5] = p[3].real

    x = append(fourier(t, Cy), [p[4].real])
    y = -append(fourier(t, Cx), [-p[4].imag])

    return x, y

# Static plot - save it
fig_static, ax_static = plt.subplots(figsize=(10, 8))
fig_static.patch.set_alpha(0.0)
ax_static.patch.set_alpha(0.0)

# Generate complete peashooter with full parameter range
t_full = linspace(0, 2 * pi, 2000)
x_static, y_static = peashooter(t=t_full, p=parameters)

# Plot body (all points except the last one) with thick lines
ax_static.plot(x_static[:-1], y_static[:-1], 'b-', linewidth=5)

# Plot eye (the last point) as a separate marker
ax_static.plot(x_static[-1], y_static[-1], 'bo', markersize=12)

# Set appropriate limits
ax_static.set_xlim([min(x_static) - 20, max(x_static) + 20])
ax_static.set_ylim([min(y_static) - 20, max(y_static) + 20])
ax_static.axis('off')
ax_static.set_aspect('equal')

# Save static plot
fig_static.savefig('peashooter_static.png', transparent=True, facecolor='none', bbox_inches='tight', dpi=150)
plt.close(fig_static)

# Animation setup
def init_plot():
    trunk.set_data([], [])
    eye.set_data([], [])
    return trunk, eye

def move_trunk(i):
    x, y = peashooter(linspace(0.4 + 1.7 * pi, 2 * pi + 0.8 * pi, 1000), parameters)
    phase = 2 * pi * i / 100
    for ii in range(len(y) - 1):
        y[ii] -= sin(((x[ii] - x[0]) * pi / len(y))) * sin(phase) * parameters[4].real
    trunk.set_data(x[:-1], y[:-1])
    eye.set_data([x[-1]], [y[-1]])
    return trunk, eye

# Figure Setup with transparent background
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.set_xlim([min(x_static) - 20, max(x_static) + 20])
ax.set_ylim([min(y_static) - 20, max(y_static) + 20])
ax.axis('off')
ax.set_aspect('equal')
trunk, = ax.plot([], [], 'b-', linewidth=5)
eye, = ax.plot([], [], 'bo', markersize=12)

# Animation
ani = animation.FuncAnimation(fig=fig,
                              func=move_trunk,
                              frames=100,
                              init_func=init_plot,
                              interval=50,
                              blit=True,
                              repeat=True)

# Save GIF with transparent