import numpy as np

from demo_flat import flats
from demo_gouraud import gouraud
from shade_types import shade_types


def shade_triangle(faces, depth, shade_t):
    # Get the index for each element of the sorted depth array
    depth_indexes = np.argsort(depth)
    faces = faces[depth_indexes]

    if shade_t == shade_types["flat"]:
        flats(faces, depth)
    elif shade_t == shade_types["gouraud"]:
        gouraud(faces, depth)
    else:
        raise ValueError("Invalid shading type")
