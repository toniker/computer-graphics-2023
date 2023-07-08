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


def camera_looking_at(f, cv, ck, cup, p3d):
    """
    calculates the 2D coordinates of a set of 3D points in the image plane of a camera
    :param f: the focal length of the camera
    :param cv: the center of the camera
    :param ck: the target of the camera
    :param cup: the up vector of the camera
    :param p3d: the 3D points
    :return: the 2D coordinates of the points in the image plane
    """
    _ck = np.array(ck - cv)
    cz = _ck / np.linalg.norm(_ck)
    tilt = np.array(cup - np.dot(cz.flatten(), cup.flatten()) * cz)
    cy = tilt / np.linalg.norm(tilt)
    cx = np.cross(cy.flatten(), cz.flatten())
    cx = cx.reshape((-1, 1))

    return pin_hole(f, cv, cx, cy, cz, p3d)


def rasterize(p2d, rows, columns, h, w):
    """
    Rasterizes the 2D points
    :param p2d: the 2d points
    :param rows: the number of vertical pixels
    :param columns: the number of horizontal pixels
    :param h: the height of the camera
    :param w: the width of the camera
    :return: the rasterized 2D points
    """
    vertical_ppi = rows / h
    horizontal_ppi = columns / w

    n2d = p2d
    n2d[0, :] *= horizontal_ppi
    n2d[1, :] *= vertical_ppi

    n2d = n2d.astype(int)

    x_offset = int(columns / 2) - int(n2d[1, :].mean())
    y_offset = int(rows / 2) - int(n2d[0, :].mean())

    n2d[0, :] += x_offset
    n2d[1, :] += y_offset

    return n2d

