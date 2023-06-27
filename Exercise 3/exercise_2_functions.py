import numpy as np


def change_coordinate_system(cp, r, c0):
    """
    Returns the coordinates of a set of points in a new coordinate system.
    :param cp: set of points
    :param r: rotation matrix
    :param c0: translation vector
    :return: the coordinates of the points in the new coordinate system
    """
    return np.dot(r, cp - c0).T


def pin_hole(f, cv, cx, cy, cz, p3d):
    """
    Calculates the 2D coordinates of a set of 3D points in the image plane of a pinhole camera.
    :param f: the focal length of the camera
    :param cv: the center of the camera
    :param cx: the x-axis of the camera
    :param cy: the y-axis of the camera
    :param cz: the z-axis of the camera
    :param p3d: the 3D points
    :return: the 2D coordinates of the points in the image plane
    """
    r = np.vstack((cx.T, cy.T, cz.T))

    p3d_camera = change_coordinate_system(p3d, r, cv)

    depth = p3d_camera[:, 2]

    x = (f * p3d_camera[:, 0] / depth)
    y = (f * p3d_camera[:, 1] / depth)
    p2d = np.vstack((x, y))

    return p2d, depth
