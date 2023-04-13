import math

import numpy as np


def interpolate_vectors(p1, p2, v1, v2, xy, dim) -> np.ndarray:
    """
    Calculates the value V of a vector, by linearly interpolating between two given vectors. The calculated vector
    must lie on the line between the given vectors.
    :param p1: The [x, y] coordinates of the first vector
    :param p2: The [x, y] coordinates of the second vector
    :param v1: The value of the first vector
    :param v2: The value of the second vector
    :param xy: The x or y coordinate of the vector to be calculated
    :param dim: The dimension of the vector to be calculated. 1 = x, 2 = y
    :return: The value of the vector
    """
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


if __name__ == "__main__":
    p1 = [0, 2]
    p2 = [10, 12]
    v1 = 0
    v2 = 10
    xy = 6
    dim = 2

    result = interpolate_vectors(p1, p2, v1, v2, xy, dim)
    print(result)
