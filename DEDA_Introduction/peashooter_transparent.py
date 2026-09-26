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

from pathlib import Path
from PIL import Image, ImageSequence
from matplotlib.animation import PillowWriter

from matplotlib import animation
from numpy import append, cos, linspace, pi, sin, zeros
import matplotlib.pyplot as plt

parameters = [50 - 50j, 18 + 80j, 12 - 10j, -14 - 60j, 20 + 20j]

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
# Mouth movements per second; increase this value for faster firing.
mouth_frequency = 2.0

def init_plot():
    trunk.set_data([], [])
    eye.set_data([], [])
    return trunk, eye

def move_trunk(i):
    x, y = peashooter(linspace(2 * pi + 0.8 * pi, 0.4 + 3.7 * pi, 1000), parameters)
    phase = 2 * pi * mouth_frequency * i / 20
    for ii in range(len(y) - 1):
        y[ii] -= sin(((x[ii] - x[0]) * pi / len(y))) * sin(phase) * parameters[4].real
    trunk.set_data(x[:-1], y[:-1])
    eye.set_data([x[-1]], [y[-1]])
    return trunk, eye

# Figure Setup with transparent background
fig, ax = plt.subplots(figsize=(10, 8), facecolor="none")
fig.patch.set_alpha(0)
ax.set_facecolor("none")

ax.set_xlim([min(x_static) - 20, max(x_static) + 20])
ax.set_ylim([min(y_static) - 20, max(y_static) + 20])
ax.axis('off')
ax.set_aspect('equal')

# Keep the lower body and legs fixed; animate the complementary head/mouth arc.
t_body = linspace(0.4 + 1.7 * pi, 2 * pi + 0.8 * pi, 1000)
x_body, y_body = peashooter(t_body, parameters)
ax.plot(x_body[:-1], y_body[:-1], 'b-', linewidth=5)

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

# Use the same output paths and GIF-saving procedure as rabbit.
output_directory = Path(__file__).resolve().parent
temporary_gif = output_directory / "peashooter_wiggle_temporary.gif"
gif_output = output_directory / "peashooter_mouth_transparent.gif"

# Force the temporary animation to be fully opaque. This avoids a bug in
# Matplotlib 3.10.0's handling of transparent PillowWriter frames.
fig.patch.set_facecolor("white")
fig.patch.set_alpha(1)
ax.set_facecolor("white")

ani.save(
    temporary_gif,
    writer=PillowWriter(fps=20),
    # Matplotlib 3.10.0 has a PillowWriter bug when it receives an RGBA
    # frame. Render an opaque temporary GIF and make it transparent in
    # the reliable Pillow post-processing step below.
    savefig_kwargs={"transparent": False, "facecolor": "white"},
)

make_gif_transparent(temporary_gif, gif_output, threshold=245)
temporary_gif.unlink()

print(f"Saved transparent GIF: {gif_output}")
