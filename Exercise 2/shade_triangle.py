import numpy as np

from interpolate_vectors import interpolate_vectors


class Edge:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.min_x = min(x1, x2)
        self.max_x = max(x1, x2)
        self.min_y = min(y1, y2)
        self.max_y = max(y1, y2)

    def get_intersecting_x(self, y):
        """
        Calculates the x coordinate of the edge at the given y coordinate
        :param y: The y coordinate
        :return: The x coordinate
        """
        return self.x1 + (y - self.y1) * (self.x2 - self.x1) / (self.y2 - self.y1)


def get_edges_from_vertices(vertices):
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]

    e0 = Edge(x0, y0, x1, y1)
    e1 = Edge(x1, y1, x2, y2)
    e2 = Edge(x2, y2, x0, y0)

    return np.array([e0, e1, e2])


def shade_triangle(canvas, vertices, colors, shade_t):
    if shade_t == "flat":
        canvas = flats(canvas, vertices, colors)
    elif shade_t == "gouraud":
        canvas = gourauds(canvas, vertices, colors)
    else:
        raise ValueError("Invalid shading type")

    return canvas


def flats(canvas, vertices, colors):
    edges = get_edges_from_vertices(vertices)
    y_min = min(edges, key=lambda e: e.min_y).min_y
    y_max = max(edges, key=lambda e: e.max_y).max_y

    # Get the average color of the triangle
    color = np.mean(colors, axis=0)

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
            return canvas

        active_vertices = np.empty((0, 2))
        for active_edge in active_edges:
            # Get the intersecting x value of the edge at the current y
            x = active_edge.get_intersecting_x(y)
            active_vertices = np.vstack((active_vertices, np.array([x, y])))

        # Sort the vertices by x value
        active_vertices = np.sort(active_vertices, axis=0)

        for i in range(0, len(active_vertices) - 1, 2):
            x1 = int(active_vertices[i][0])
            x2 = int(active_vertices[i + 1][0])
            # Set the color of the pixels between the two vertices
            canvas[x1:x2, y, :] = color

    return canvas


def gourauds(canvas, vertices, colors):
    edges = get_edges_from_vertices(vertices)
    y_min = round(min(edges, key=lambda e: e.min_y).min_y)
    y_max = round(max(edges, key=lambda e: e.max_y).max_y)

    active_edges = np.array([], dtype=Edge)
    for y in range(y_min, min(y_max + 1, canvas.shape[1])):
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
            return canvas

        active_vertices = np.empty((0, 2))

        for active_edge in active_edges:
            # Get the intersecting x value of the edge at the current y
            x = active_edge.get_intersecting_x(y)
            active_vertices = np.vstack((active_vertices, np.array([x, y])))

        active_vertices = np.sort(active_vertices, axis=0)

        # For the point A, get the index of the vertices in the vertices array
        # We then use these indexes to get the colors of the vertices from the colors array
        color1_a_index = \
            np.equal(vertices, (active_edges[0].x1, active_edges[0].y1)).all(axis=1).nonzero()[0].tolist()[0]
        color2_a_index = \
            np.equal(vertices, (active_edges[0].x2, active_edges[0].y2)).all(axis=1).nonzero()[0].tolist()[0]

        # Interpolate the colors of the vertices in the y direction
        color_a = interpolate_vectors((active_edges[0].x1, active_edges[0].y1),
                                      (active_edges[0].x2, active_edges[0].y2), colors[color1_a_index],
                                      colors[color2_a_index], y, dim=2)

        color1_b_index = \
            np.equal(vertices, (active_edges[1].x1, active_edges[1].y1)).all(axis=1).nonzero()[0].tolist()[0]
        color2_b_index = \
            np.equal(vertices, (active_edges[1].x2, active_edges[1].y2)).all(axis=1).nonzero()[0].tolist()[0]
        color_b = interpolate_vectors((active_edges[1].x1, active_edges[1].y1),
                                      (active_edges[1].x2, active_edges[1].y2), colors[color1_b_index],
                                      colors[color2_b_index], y, dim=2)

        for x in range(0, len(active_vertices) - 1, 2):
            # Get the x values of the vertices
            x1 = int(active_vertices[x][0])
            x2 = int(active_vertices[x + 1][0])

            # Interpolate the colors of the vertices in the x direction
            color = interpolate_vectors((x1, y), (x2, y), color_a, color_b, x, dim=1)

            # Set the color of the pixels between the two vertices
            canvas[x1:x2, y, :] = color

    return canvas
