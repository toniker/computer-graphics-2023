import numpy as np

from shade_triangle import shade_triangle


def render(verts2d, faces, vcolors, depth, shade_t) -> np.ndarray:
    """
    Renders a 3D object using the flat shading model.
    :param verts2d: A list of 2D vertices
    :param vcolors: A list of vertex colors
    :param faces: A list of faces
    :param depth: A list of depth values
    :param shade_t: The shading type
    :return: The final image
    """

    # The width and height of the image in pixels
    width = 512
    height = 512

    # Create the image background
    canvas = np.ones((width, height, 3))

    # Set the background to white
    canvas = canvas * 255

    depth_order = np.array(np.mean(depth[faces], axis=1))
    sorted_triangles = list(np.flip(np.argsort(depth_order)))

    # Loop through all faces, starting by the furthest one to the camera
    for triangle in sorted_triangles:
        vertice_indexes = faces[triangle]
        vertices = verts2d[vertice_indexes]
        colors = vcolors[vertice_indexes]
        canvas = shade_triangle(canvas, vertices, colors, shade_t)

    return canvas
