import time

import cv2
import numpy as np

from exercise_2_functions import camera_looking_at, rasterize
from shade_triangle import gourauds, get_edges_from_vertices, Edge
from interpolate_vectors import interpolate_vectors


class PhongMaterial:
    def __init__(self, ka, kd, ks, n):
        """
        Initializes a Phong material.
        :param ka: The ambient coefficient.
        :param kd: The diffusion coefficient.
        :param ks: The specular coefficient.
        :param n: The Phong exponent.
        """
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.n = n


class PointLight:
    def __init__(self, position, intensity):
        """
        Initializes a point light object.
        :param position: A (3,1) array representing the position of the light.
        :param intensity: A (3,1) array representing the intensity of the light for each color.
        """
        self.position = position
        self.intensity = intensity


def calculate_normals(verts, faces):
    """
    Calculates the normal vectors for each face.
    :param verts: The vertices of the object.
    :param faces: The faces of the object.
    :return: The normal vectors for each face.
    """
    normals = np.zeros_like(faces, dtype=np.float64)

    for i, face in enumerate(faces.T):
        v0 = verts[:, face[0]]
        v1 = verts[:, face[1]]
        v2 = verts[:, face[2]]

        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)

        normals[:, face[0]] += normal
        normals[:, face[1]] += normal
        normals[:, face[2]] += normal

    normals = normals / np.linalg.norm(normals, axis=0)
    return normals


def light(point, normal, color, cam_pos, mat, lights, light_amb):
    """
    Calculates the lighting of a point, which belongs to a Phong material, given the diffused, specular and reflected
    light.
    :param point: A (3,1) array representing the point.
    :param normal: The (3,1) vector of the surface normal.
    :param color: The color of the point.
    :param cam_pos: The position of the camera.
    :param mat: The Phong material.
    :param lights: a list of PointLight objects.
    :param light_amb: The ambient light intensity.
    :return: The intensity of the color of the point.
    """
    I = np.zeros(3)

    I_ambient = mat.ka * light_amb
    if lighting_model == 'ambient' or lighting_model == 'all':
        I += I_ambient

    for _light in lights:
        light_vector = _light.position - point
        light_vector /= np.linalg.norm(light_vector)

        reflection_vector = 2 * normal * np.dot(normal, light_vector) - light_vector
        reflection_vector /= np.linalg.norm(reflection_vector)

        view_vector = cam_pos - point
        view_vector /= np.linalg.norm(view_vector)

        I_d = _light.intensity * mat.kd * np.dot(normal, light_vector)
        I_s = _light.intensity * mat.ks * np.dot(reflection_vector, view_vector) ** mat.n

        if lighting_model == 'diffusion' or lighting_model == 'all':
            I += I_d
        if lighting_model == 'specular' or lighting_model == 'all':
            I += I_s

    return np.clip(I, 0, 255) * color


def shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, mat, lights, light_amb, img):
    """
    Shades the object using the Gouraud shading method.
    :param verts_p: The points of the vertices.
    :param verts_n: The normals of the vertices.
    :param verts_c: The colors of the vertices.
    :param bcoords: The barycentric coordinates of the point.
    :param cam_pos: The position of the camera.
    :param mat: The Phong material.
    :param lights: A list of PointLight objects.
    :param light_amb: The ambient light.
    :param img: The image.
    :return: The image with the object shaded.
    """
    light_at_vertices = np.zeros_like(verts_c)
    for i in range(len(verts_p)):
        light_at_vertices[i] = light(bcoords, verts_n[i], verts_c[i], cam_pos, mat, lights, light_amb)

    img = gourauds(img, verts_p, light_at_vertices)
    return img


def shade_phong(vertices, normals, colors, bcoords, cam_pos, mat, lights, light_amb, img):
    edges = get_edges_from_vertices(vertices)
    y_min = min(edges, key=lambda e: e.min_y).min_y
    y_max = max(edges, key=lambda e: e.max_y).max_y

    active_edges = np.array([], dtype=Edge)

    for y in range(y_min, y_max + 1):
        # Set active edges
        for edge in edges:
            # If the edge is not horizontal and the start of the edge is at the current y
            if edge.min_y == y and edge.max_y != edge.min_y:
                active_edges = np.append(active_edges, edge)
            # If the end of the edge is at the current y
            elif edge.max_y == y:
                active_edges = np.delete(active_edges, np.where(active_edges == edge))

        # If there are less than 2 active edges, then the triangle is not visible
        if len(active_edges) < 2:
            return img

        active_vertices = np.empty((0, 2))

        for active_edge in active_edges:
            # Get the intersecting x value of the edge at the current y
            x = active_edge.get_intersecting_x(y)
            active_vertices = np.vstack((active_vertices, np.array([x, y])))

        active_vertices = np.sort(active_vertices, axis=0)
        color1_a_index = \
            np.equal(vertices, (active_edges[0].x1, active_edges[0].y1)).all(axis=1).nonzero()[0].tolist()[0]
        color2_a_index = \
            np.equal(vertices, (active_edges[0].x2, active_edges[0].y2)).all(axis=1).nonzero()[0].tolist()[0]

        # Interpolate the colors of the vertices in the y direction
        color_a = interpolate_vectors((active_edges[0].x1, active_edges[0].y1),
                                      (active_edges[0].x2, active_edges[0].y2), colors[color1_a_index],
                                      colors[color2_a_index], y, dim=2)
        normal_a = interpolate_vectors((active_edges[0].x1, active_edges[0].y1),
                                       (active_edges[0].x2, active_edges[0].y2), normals[color1_a_index],
                                       normals[color2_a_index], y, dim=2)

        color1_b_index = \
            np.equal(vertices, (active_edges[1].x1, active_edges[1].y1)).all(axis=1).nonzero()[0].tolist()[0]
        color2_b_index = \
            np.equal(vertices, (active_edges[1].x2, active_edges[1].y2)).all(axis=1).nonzero()[0].tolist()[0]
        color_b = interpolate_vectors((active_edges[1].x1, active_edges[1].y1),
                                      (active_edges[1].x2, active_edges[1].y2), colors[color1_b_index],
                                      colors[color2_b_index], y, dim=2)
        normal_b = interpolate_vectors((active_edges[1].x1, active_edges[1].y1),
                                       (active_edges[1].x2, active_edges[1].y2), normals[color1_b_index],
                                       normals[color2_b_index], y, dim=2)

        for x in range(0, len(active_vertices) - 1, 2):
            # Get the x values of the vertices
            x1 = int(active_vertices[x][0])
            x2 = int(active_vertices[x + 1][0])

            # Interpolate the colors of the vertices in the x direction
            color = interpolate_vectors((x1, y), (x2, y), color_a, color_b, x, dim=1)
            normal = interpolate_vectors((x1, y), (x2, y), normal_a, normal_b, x, dim=1)

            # Set the color of the pixels between the two vertices
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            img[img.shape[0] - y, min_x:max_x, :] = light(bcoords, normal, color, cam_pos, mat, lights, light_amb)

    return img


def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, n, lights,
                  light_amb):
    """
    Renders an object given the vertices, faces, material and lights.
    :param shader: The shader to use.
    :param focal: The focal length.
    :param eye: The position of the camera.
    :param lookat: The vector pointed where the camera is looking at.
    :param up: The camera up vector.
    :param bg_color: The background color.
    :param M: The height of the image.
    :param N: The width of the image.
    :param H: The height of the image plane.
    :param W: The width of the image plane.
    :param verts: The vertices of the object.
    :param vert_colors: The colors of the vertices.
    :param faces: The faces of the object.
    :param mat: The Phong material.
    :param n: The Phong exponent.
    :param lights: A list of PointLight objects.
    :param light_amb: The ambient light.
    :return:
    """
    img = np.ones((M, N, 3)) * bg_color

    vert_normals = calculate_normals(verts, faces)

    vert_points, depths = camera_looking_at(focal, eye, lookat, up, verts)

    _verts = verts.T
    barycentric_coords = np.zeros_like(_verts)

    for i, face in enumerate(faces):
        v0 = _verts[face[0]]
        v1 = _verts[face[1]]
        v2 = _verts[face[2]]
        barycentric_coords[i] = (v0 + v1 + v2) / 3

    cam_pos = eye

    n2d = rasterize(vert_points, M, N, H, W)

    depths = np.mean(depths[faces], axis=0)
    sorted_depths = np.argsort(depths)[::-1]
    _faces = faces.T
    _vert_normals = vert_normals.T
    _vert_colors = vert_colors.T

    if shader == 'gouraud':
        for i in sorted_depths:
            face = _faces[i]
            verts_p = n2d[face]
            verts_n = _vert_normals[face]
            verts_c = _vert_colors[face]
            bcoords = barycentric_coords[i]
            img = shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, mat, lights, light_amb, img)
    elif shader == 'phong':
        for i in sorted_depths:
            face = _faces[i]
            verts_p = n2d[face]
            verts_n = _vert_normals[face]
            verts_c = _vert_colors[face]
            bcoords = barycentric_coords[i]
            img = shade_phong(verts_p, verts_n, verts_c, bcoords, cam_pos, mat, lights, light_amb, img)

    return img * 255


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

    lights = [PointLight(position=light_positions[i], intensity=light_intensities[i]) for i in
              range(len(light_positions))]
    del light_positions, light_intensities
    mat = PhongMaterial(ka, kd, ks, n)
    del ka, kd, ks

    for shader in shaders:
        for lighting_model in lighting_models:
            img = render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                                face_indices, mat, n, lights, Ia)
            cv2.imwrite(f"{shader}_{lighting_model}.png", img)

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Render finished in {execution_time} seconds")
