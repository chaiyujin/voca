import os
from meshio import load_mesh
from functools import lru_cache

_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=5)
def get_selection_triangles(part):

    _, tris, _ = load_mesh(os.path.join(_DIR, f"{part}.obj"))
    return tris


@lru_cache(maxsize=5)
def get_selection_vidx(part):
    with open(os.path.join(_DIR, f"{part}.txt")) as fp:
        line = " ".join([x.strip() for x in fp])
        vidx = [int(x) for x in line.split()]
    return vidx


def get_selection_obj(part):
    return os.path.join(_DIR, f"{part}.obj")
