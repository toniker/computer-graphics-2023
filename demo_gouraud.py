import numpy as np
import cv2 as cv
from render import render
from shade_types import shade_types
import time


if __name__ == "__main__":
    start_time = time.time()

    # The edges of the triangles. Size of L x 2 and contains the X and Y coordinates of each edge.
    verts2d = np.load("h1.npy", allow_pickle=True).tolist()['verts2d']
    # The colors of the edges. Size of L x 3 and contains the RGB values for each i-th edge.
    vcolors = np.load("h1.npy", allow_pickle=True).tolist()['vcolors']
    # Contains the faces of the triangles as a K x 3 array. Each of the K rows contains the indexes for the verts2d
    # triangle edges.
    faces = np.load("h1.npy", allow_pickle=True).tolist()['faces']
    # Contains the depth of each edge. Size of L x 1
    depth = np.load("h1.npy", allow_pickle=True).tolist()['depth']

    img = render(verts2d, faces, vcolors, depth, shade_types["gouraud"])
    img = img * 255

    execution_time = round(time.time() - start_time, 3)
    print(f"Render finished in {execution_time} seconds")
    cv.imwrite("gouraud.jpg", img)
