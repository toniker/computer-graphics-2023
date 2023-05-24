import time

import cv2
import numpy as np

from shade_triangle import gourauds


def rotmat(theta, unit):
    """
    calculates the rotation matrix R corresponding to clockwise rotation by
    angle theta in rads about an axis with a direction given by the unit vector u.
    The array is implemented from the Rotation matrix from axis and angle section of
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    :param theta: angle in degrees
    :param unit: unit vector
    :return: rotation matrix
    """
    x, y, z = unit / np.linalg.norm(unit)
    theta = np.deg2rad(theta)
    cos = np.cos(theta)
    sin = np.sin(theta)
    r = np.array([[cos + x ** 2 * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin],
                  [y * x * (1 - cos) + z * sin, cos + y ** 2 * (1 - cos), y * z * (1 - cos) - x * sin],
                  [z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z ** 2 * (1 - cos)]])
    return r


def rotate_translate(cp, theta, unit, A, t):
    """
    rotates and translates a set of points
    :param cp: set of points
    :param theta: angle in degrees
    :param unit: the unit vector
    :param A: translation matrix
    :param t: translation vertex
    :return: the translated set of points
    """
    if cp.ndim == 1:
        cp = cp.reshape((-1, 1))

    centered = np.dot(np.linalg.inv(A), cp)

    r = rotmat(theta, unit)
    rotated = np.dot(r, centered)

    translated = np.dot(A, rotated)

    return translated + t.reshape((-1, 1))


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

    return n2d.T


def render(n2d, faces, colors, depth, rows, columns):
    """
    renders the object
    :param n2d: the 2D points
    :param faces: the indexes of the vertices of each face
    :param colors: the colors of the vertices
    :param depth: the depth of each face
    :param rows: the number of vertical pixels
    :param columns: the number of horizontal pixels
    :return: the rendered image
    """
    img = np.ones((rows, columns, 3))

    depths = np.array([np.sum(depth[face]) for face in faces])
    sorted_indices = np.argsort(depths)[::-1]

    for i in sorted_indices:
        face = faces[i]
        vertices = n2d[face]
        color = colors[face]
        img = gourauds(img, vertices, color)

    return img


def render_object(p3d, faces, colors, h, w, rows, columns, f, cv, ck, cup):
    """
    Renders the object
    :param p3d:  3D points of the object, shape (L, 3)
    :param faces: the indices of the vertices of each face, shape (M, 3)
    :param colors: the colors of the vertices, shape (L, 3)
    :param h: height of the camera
    :param w: width of the camera
    :param rows: height of the image
    :param columns: width of the image
    :param f: distance of the camera to the image plane
    :param cv: center of the camera as per the world coordinate system
    :param ck: target of the camera as per the world coordinate system (non-homogeneous)
    :param cup: up vector of the camera as per the world coordinate system (non-homogeneous)
    :return: the image
    """
    p2d, depth = camera_looking_at(f, cv, ck, cup, p3d)

    n2d = rasterize(p2d, rows, columns, h, w)

    img = render(n2d, faces, colors, depth, rows, columns)

    return img * 255


if __name__ == "__main__":
    start_time = time.time()

    width, height = 512, 512
    cam_width, cam_height = 15, 15

    data = np.load("h2.npy", allow_pickle=True).tolist()
    verts3d = data['verts3d']
    vcolors = data['vcolors']
    faces = data['faces']
    c_org = data['c_org']
    c_lookat = data['c_lookat']
    c_up = data['c_up']
    t_1 = data['t_1']
    t_2 = data['t_2']
    u = data['u']
    phi = data['phi']
    focal = data['focal']
    del data

    image = render_object(verts3d, faces, vcolors, cam_height, cam_width, height, width, focal, c_org, c_lookat, c_up)
    cv2.imwrite("original.jpg", image)

    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    verts3d_translated_t1 = rotate_translate(verts3d, 0, u, A, t_1)
    image = render_object(verts3d_translated_t1, faces, vcolors, cam_height, cam_width, height, width, focal, c_org,
                          c_lookat, c_up)
    cv2.imwrite("t1.jpg", image)

    verts3d_rotated = rotate_translate(verts3d_translated_t1, phi, u, A, np.zeros(3))
    image = render_object(verts3d_rotated, faces, vcolors, cam_height, cam_width, height, width, focal, c_org, c_lookat,
                          c_up)
    cv2.imwrite("rotated.jpg", image)

    verts3d_translated_t2 = rotate_translate(verts3d_rotated, 0, u, A, t_2)
    image = render_object(verts3d_translated_t2, faces, vcolors, cam_height, cam_width, height, width, focal, c_org,
                          c_lookat, c_up)

    cv2.imwrite("t2.jpg", image)

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Render finished in {execution_time} seconds")
