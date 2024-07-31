import numpy as np


def dist_point_line(p1, p2, c):
    d = p2 - p1
    v = c - p1

    # Projection of v onto d
    t = np.dot(v, d) / np.dot(d, d)
    t = max(0, min(1, t))

    p_closest = p1 + t * d
    distance = np.linalg.norm(c - p_closest)

    return distance


def intersects_sphere(p1, p2, c, r):
    return dist_point_line(p1, p2, c) <= r
