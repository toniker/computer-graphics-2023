import time

import cv2
import numpy as np


class PhongMaterial:
    def __init__(self, ka, kd, ks, n):
        """
        Initializes a Phong material
        :param ka: The ambient coefficient
        :param kd: The diffusion coefficient
        :param ks: The specular coefficient
        :param n: The Phong exponent
        """
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.n = n


class PointLight:
    def __init__(self, position, intensity):
        """
        Initializes a point light object
        :param position: A (3,1) array representing the position of the light
        :param intensity: A (3,1) array representing the intensity of the light for each color
        """
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

        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)

        normals[i] = normal

    return normals


def light(point, normal, color, cam_pos, mat, lights):
    """
    Calculates the lighting of a point, which belongs to a Phong material, given the diffused, specular and reflected
    light
    :param point: A (3,1) array representing the point
    :param normal: The (3,1) vector of the surface normal
    :param color: The color of the point
    :param cam_pos: The position of the camera
    :param mat: The Phong material
    :param lights: a list of PointLight objects
    :return: The intensity of the color of the point
    """
    I = np.zeros((1, 3))


    return I


def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, n, lights,
                  light_amb):
    """
    Renders an object given the vertices, faces, material and lights
    :param shader: The shader to use
    :param focal: The focal length
    :param eye: The position of the camera
    :param lookat: The vector pointed where the camera is looking at
    :param up: The camera up vector
    :param bg_color: The background color
    :param M: The height of the image
    :param N: The width of the image
    :param H: The height of the image plane
    :param W: The width of the image plane
    :param verts: The vertices of the object
    :param vert_colors: The colors of the vertices
    :param faces: The faces of the object
    :param mat: The Phong material
    :param n: The Phong exponent
    :param lights: A list of PointLight objects
    :param light_amb: The ambient light
    :return:
    """
    img = np.ones((M, N, 3)) * bg_color

    if shader == 'gouraud':
        img = shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, mat, lights, light_amb, img)
    elif shader == 'phong':
        img = shade_phong(verts_p, verts_n, verts_c, bcoords, cam_pos, mat, lights, light_amb, img)

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

    lights = [PointLight(light_positions[i], light_intensities[i]) for i in range(len(light_positions))]
    mat = PhongMaterial(ka, kd, ks, n)
    light_amb = np.full((1, 3), Ia)

    for shader in shaders:
        for lighting_model in lighting_models:
            # faces as for calculate_normals
            img = render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                                faces, mat, n, lights, light_amb)

            cv2.imwrite(f"output_{shader}_{lighting_model}.png", img)

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Render finished in {execution_time} seconds")
