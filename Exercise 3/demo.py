import time

import cv2
import numpy as np


class PhongMaterial:
    def __init__(self, ka, kd, ks, n):
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.n = n


class PointLight:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity


def calculate_normals(verts, faces):
    """
    Calculates the normal vectors for each face
    :param verts: The vertices of the object
    :param faces: The faces of the object
    :return: The normal vectors for each face
    """
    normals = np.zeros((faces.shape[0], 3))

    for i, face in enumerate(faces):
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]

        # Calculate the normal vector
        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)

        normals[i] = normal

    return normals


def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, n, lights,
                  light_amb):
    img = np.ones((M, N, 3)) * bg_color

    return img


if __name__ == "__main__":
    start_time = time.time()

    data = np.load("h3.npy", allow_pickle=True).tolist()
    verts = data['verts']
    vertex_colors = data['vertex_colors']
    face_indices = data['face_indices']
    cam_eye = data['cam_eye']
    cam_up = data['cam_up']
    cam_lookat = data['cam_lookat']
    ka = data['ka']
    kd = data['kd']
    ks = data['ks']
    n = data['n']
    light_positions = data['light_positions']
    light_intensities = data['light_intensities']
    Ia = data['Ia']
    M = data['M']
    N = data['N']
    W = data['W']
    H = data['H']
    bg_color = data['bg_color']
    focal = data['focal']
    del data

    shaders = ['gouraud', 'phong']
    lighting_models = ['ambient', 'diffusion', 'specular', 'all']

    for shader in shaders:
        for lighting_model in lighting_models:
            # lights = [point lights]
            # mat = phong material
            # faces as for calculate_normals
            # light_amb = [I_r, I_g, I_b]^T
            img = render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                                faces, mat, n, lights, light_amb)

            cv2.imwrite(f"output_{shader}_{lighting_model}.png", img)

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Render finished in {execution_time} seconds")
