import time

import cv2
import numpy as np
from shade_triangle import gourauds


def rotmat(theta, u):
    """
    calculates the rotation matrix R corresponding to clockwise rotation by
    angle theta in rads about an axis with a direction given by the unit vector u.
    The array is implemented from the Rotation matrix from axis and angle section of
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    :param theta: angle in degrees
    :param u: unit vector
    :return: rotation matrix
    """
    u = u / np.linalg.norm(u)
    x, y, z = u
    theta = np.deg2rad(theta)
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos + x ** 2 * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin],
                  [y * x * (1 - cos) + z * sin, cos + y ** 2 * (1 - cos), y * z * (1 - cos) - x * sin],
                  [z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z ** 2 * (1 - cos)]])
    return R


def rotate_translate(cp, theta, u, A, t):
    """
    Transforms a three-dimensional point, by rotating it around a given axis u by a given angle theta,
    and then translating it by a given vector t. All coordinates are given in the world coordinate system.
    :param cp: 3D point
    :param theta: angle in degrees
    :param u: unit vector
    :param A: translation matrix
    :param t: translation vector
    :return: transformed 3D point
    """
    R = rotmat(theta, u)
    return np.dot(R, cp) + A + t


def change_coordinate_system(cp, R, c0):
    """
    Changes the coordinate system of a point from the world coordinate system to the camera coordinate system.
    :param cp: 3D point
    :param R: rotation matrix
    :param c0: camera position
    :return: transformed 3D point
    """
    return np.dot(R, cp - c0)


def pin_hole(f, cv, cx, cy, cz, p3d):
    """
    Projects a 3D point onto the image plane of a pinhole camera.
    :param f: Distance from the camera center to the image plane
    :param cv:
    :param cx:
    :param cy:
    :param cz:
    :param p3d:
    :return:
    """
    return p2d, depth


def camera_looking_at(f, cv, ck, cup, p3d):
    """

    :param f:
    :param cv:
    :param ck:
    :param cup:
    :param p3d:
    :return:
    """
    return p2d, depth


def rasterize(p2d, rows, columns, h, w):
    return n2d


def render_object(p3d, faces, colors, h, w, rows, columns, f, cv, ck, cup):
    p2d, depth = camera_looking_at(f, cv, ck, cup, p3d)
    n2d = rasterize(p2d, rows, columns, h, w)
    image = render(n2d, faces, colors, depth, h, w)
    return image


def render(p2d, faces, colors, depth, h, w):
    return image


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

    image = render_object(verts3d, faces, vcolors, height, width, cam_height, cam_width, focal, c_org, c_lookat, c_up)
    cv2.imwrite("original.png", image)

    verts3d_translated_t1 = rotate_translate(verts3d, 0, u, np.zeros(3), t_1)
    image = render_object(verts3d_translated_t1, faces, vcolors, height, width, cam_height,
                          cam_width, focal, c_org, c_lookat,
                          c_up)
    cv2.imwrite("t1.png", image)

    verts3d_rotated = rotate_translate(verts3d_translated_t1, phi, u, np.zeros(3), np.zeros(3))
    image = render_object(verts3d_rotated, faces, vcolors, height, width, cam_height, cam_width, focal, c_org, c_lookat,
                          c_up)
    cv2.imwrite("rotated.png", image)

    verts3d_translated_t2 = rotate_translate(verts3d_rotated, 0, u, np.zeros(3), t_2)
    image = render_object(verts3d_translated_t2, faces, vcolors, height, width, cam_height, cam_width, focal, c_org,
                          c_lookat, c_up)

    cv2.imwrite("t2.png", image)

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Render finished in {execution_time} seconds")
