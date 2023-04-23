import numpy as np

from shade_types import shade_types


def shade_triangle(canvas, vertices, colors, shade_t):
    if shade_t == shade_types["flat"]:
        canvas = flats(canvas, vertices, colors)
    elif shade_t == shade_types["gouraud"]:
        canvas = gouraud(canvas, vertices, colors)
    else:
        raise ValueError("Invalid shading type")

    return canvas


def flats(canvas, vertices, colors):
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)

    color = np.mean(colors, axis=0)
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            canvas[x, y] = color

    return canvas


def gouraud(canvas, vertices, colors):
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)

    return canvas
