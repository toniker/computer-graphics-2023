import numpy as np

from shade_types import shade_types


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
        if self.x1 == self.x2:
            x = self.x1
        else:
            offset = self.y1 - self.slope * self.x1
            x = round((y - offset) / self.slope)

        assert self.min_x <= x <= self.max_x
        return x


def get_edges_from_vertices(vertices):
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]

    e0 = Edge(x0, y0, x1, y1)
    e1 = Edge(x1, y1, x2, y2)
    e2 = Edge(x2, y2, x0, y0)

    return np.array([e0, e1, e2])


def shade_triangle(canvas, vertices, colors, shade_t):
    if shade_t == shade_types["flat"]:
        canvas = flats(canvas, vertices, colors)
    elif shade_t == shade_types["gouraud"]:
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
    active_vertices = np.empty((0, 2))

    for edge in edges:
        # Get initial active edges
        if edge.min_y == y_min:
            if edge.slope != 0:
                active_edges = np.append(active_edges, edge)
            else:
                canvas[edge.min_x:edge.max_x, edge.min_y, :] = color

    if len(active_edges) < 2:
        # Can this ever run?
        return canvas

    # Get initial active vertices
    for active_edge in active_edges:
        active_vertices = np.vstack((active_vertices, np.array([active_edge.get_intersecting_x(y_min), y_min])))

    for y in range(y_min, y_max + 1):
        sorted_active_vertices = active_vertices[active_vertices[:, 1].argsort()]

        for i in range(0, len(sorted_active_vertices) - 1, 2):
            x1 = int(sorted_active_vertices[i][0])
            x2 = int(sorted_active_vertices[i + 1][0])
            canvas[x1:x2, y, :] = color

        # Recursively set active edges and vertices
        for edge in edges:
            if edge.min_y == y + 1:
                if edge.slope != 0:
                    active_edges = np.append(active_edges, edge)
                else:
                    canvas[edge.min_x:edge.max_x, edge.min_y, :] = color
            elif edge.max_y == y:
                active_edges = np.delete(active_edges, np.where(active_edges == edge))

        active_vertices = np.empty((0, 2))
        for active_edge in active_edges:
            x = active_edge.get_intersecting_x(y + 1)
            active_vertices = np.vstack((active_vertices, np.array([x, y + 1])))

    return canvas


def gouraud(canvas, vertices, colors):
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)

    return canvas
