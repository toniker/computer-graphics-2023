import time

import numpy as np


def rotmat(theta, u):
    """
    calculates the rotation matrix R corresponding to clockwise rotation by
    angle theta in rads about an axis with a direction given by the unit vector u.
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

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Render finished in {execution_time} seconds")

    # Save the image
    # cv.imwrite("render.jpg", img)
