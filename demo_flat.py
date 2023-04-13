import numpy as np

if __name__ == "__main__":
    verts2d = np.load("h1.npy", allow_pickle=True).tolist()['verts2d']
    vcolors = np.load("h1.npy", allow_pickle=True).tolist()['vcolors']
    faces = np.load("h1.npy", allow_pickle=True).tolist()['faces']
    depth = np.load("h1.npy", allow_pickle=True).tolist()['depth']
    print("Hello, World!")

