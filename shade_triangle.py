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
        self.slope = (y2 - y1) / (x2 - x1)

    def get_intersecting_x(self, y):
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
        canvas = gouraud(canvas, vertices, colors)
    else:
        raise ValueError("Invalid shading type")

    return canvas


def flats(canvas, vertices, colors):
    edges = get_edges_from_vertices(vertices)
    y_min = min(edges, key=lambda e: e.min_y).min_y
    y_max = max(edges, key=lambda e: e.max_y).max_y

    color = np.mean(colors, axis=0)

    active_edges = np.array([], dtype=Edge)
    for y in range(y_min, y_max + 1):
        # Set active edges and vertices
        for edge in edges:
            if edge.min_y == y and edge.max_y != edge.min_y:
                active_edges = np.append(active_edges, edge)
            elif edge.max_y == y:
                active_edges = np.delete(active_edges, np.where(active_edges == edge))

        if len(active_edges) < 2:
            return canvas

        active_vertices = np.empty((0, 2))
        for active_edge in active_edges:
            x = active_edge.get_intersecting_x(y)
            active_vertices = np.vstack((active_vertices, np.array([x, y])))

        active_vertices = np.sort(active_vertices, axis=0)

        for i in range(0, len(active_vertices) - 1, 2):
            x1 = int(active_vertices[i][0])
            x2 = int(active_vertices[i + 1][0])
            canvas[x1:x2, y, :] = color

    return canvas


def gouraud(canvas, vertices, colors):
    edges = get_edges_from_vertices(vertices)
    y_min = min(edges, key=lambda e: e.min_y).min_y
    y_max = max(edges, key=lambda e: e.max_y).max_y

    active_edges = np.array([], dtype=Edge)
    for y in range(y_min, y_max + 1):
        # Set active edges and vertices
        for edge in edges:
            if edge.min_y == y and edge.max_y != edge.min_y:
                active_edges = np.append(active_edges, edge)
            elif edge.max_y == y:
                active_edges = np.delete(active_edges, np.where(active_edges == edge))

        if len(active_edges) < 2:
            return canvas

        active_vertices = np.empty((0, 2))

        for active_edge in active_edges:
            x = active_edge.get_intersecting_x(y)
            active_vertices = np.vstack((active_vertices, np.array([x, y])))

        color1_a_index = \
            np.equal(vertices, (active_edges[0].x1, active_edges[0].y1)).all(axis=1).nonzero()[0].tolist()[0]
        color2_a_index = \
            np.equal(vertices, (active_edges[0].x2, active_edges[0].y2)).all(axis=1).nonzero()[0].tolist()[0]
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

        active_vertices = np.sort(active_vertices, axis=0)

        for x in range(0, len(active_vertices) - 1, 2):
            x1 = int(active_vertices[x][0])
            x2 = int(active_vertices[x + 1][0])
            color = interpolate_vectors((x1, y), (x2, y), color_a, color_b, x, dim=1)
            canvas[x1:x2, y, :] = color

    return canvas
