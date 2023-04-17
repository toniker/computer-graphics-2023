import numpy as np
from shade_types import shade_types


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

    # The list of z values
    depth_buffer = np.ones((width, height)) * np.inf

    # Get the index for each element of the sorted depth array
    depth_indexes = np.argsort(depth)

    # Loop through all faces, starting by the closest one to the camera
    for index in depth_indexes:
        face = faces[index]

        # Get the edges of the triangle
        v1 = verts2d[face[0]]
        v2 = verts2d[face[1]]
        v3 = verts2d[face[2]]

        # Get the colors of the vertices
        c1 = vcolors[face[0]]
        c2 = vcolors[face[1]]
        c3 = vcolors[face[2]]

        # Get the depth of the vertices
        d1 = depth[face[0]]
        d2 = depth[face[1]]
        d3 = depth[face[2]]

        # Get the minimum and maximum x and y values of the face
        min_x = int(min(v1[0], v2[0], v3[0]))
        max_x = int(max(v1[0], v2[0], v3[0]))
        min_y = int(min(v1[1], v2[1], v3[1]))
        max_y = int(max(v1[1], v2[1], v3[1]))

        # The scan line is parallel to one of the edges of the triangle.
        if v1[1] == v2[1] or v2[1] == v3[1] or v1[1] == v3[1]:
            color = np.mean([c1, c2, c3], axis=1)

            z = np.mean([d1, d2, d3])

            for x in range(min_x, max_x + 1):
                pass
                # If the pixel is closer to the camera than the previous pixel
                # if z < depth_buffer[x][y]:
                #     # Draw the pixel
                #     img[x][y] = color
                #
                #     # Update the z value
                #     depth_buffer[x][y] = z

    return canvas
