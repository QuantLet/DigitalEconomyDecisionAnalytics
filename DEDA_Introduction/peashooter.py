#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:51:43 2026

@author: haerdle
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:25:31 2026

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

def init_plot():
    x, y = peashooter(linspace(2 * pi + 0.9 * pi, 0.4 + 3.3 * pi, 1000), parameters)
    for ii in range(len(y) - 1):
        y[ii] -= sin(((x[ii] - x[0]) * pi / len(y))) * sin(float(0)) * parameters[4].real
    trunk.set_data(x, y)
    return trunk,

def move_trunk(i):
    x, y = peashooter(linspace(2 * pi + 0.8 * pi, 0.4 + 3.7 * pi, 1000), parameters)
    for ii in range(len(y) - 1):
        y[ii] -= sin(((x[ii] - x[0]) * pi / len(y))) * sin(float(i)) * parameters[4].real
    trunk.set_data(x, y)
    return trunk,

fig, ax = plt.subplots()
x, y = peashooter(t=linspace(0.4 + 1.7 * pi, 2 * pi + 0.8 * pi, 1000), p=parameters)
plt.plot(x, y, 'b.')
plt.xlim([-175, 190])
plt.ylim([-70, 100])
plt.axis('off')
trunk, = ax.plot([], [], 'b.') 

# Corrected FuncAnimation
ani = animation.FuncAnimation(fig=fig,
                              func=move_trunk,
                              frames=1000,
                              init_func=init_plot,
                              interval=500,
                              blit=False,
                              repeat=True)

plt.show()

# Save GIF file
ani.save("peashooter.gif", writer="pillow", fps=20)