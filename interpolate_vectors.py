import math

import numpy as np


def interpolate_vectors(p1, p2, v1, v2, xy, dim) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    x, y = 0, 0

    if dim == 1:
        x = xy
        y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    elif dim == 2:
        y = xy
        x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

    distance_to_p1 = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    distance_to_p2 = math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    distance_ratio_to_p1 = distance_to_p2 / (distance_to_p1 + distance_to_p2)
    distance_ratio_to_p2 = 1 - distance_ratio_to_p1
    v = distance_ratio_to_p1 * v1 + distance_ratio_to_p2 * v2

    return v

