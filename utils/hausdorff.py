import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from scipy.spatial.distance import cdist

mpl.use('macosx')

def hausdorff_distance_optimized(a, b, debug=False):
    """Optimized Hausdorff distance calculation using SciPy."""
    assert a.shape[1] == b.shape[1] == 3, "Point sets must have 3D coordinates"

    ca = np.average(a, axis=0)
    cb = np.average(b, axis=0)
    t2 = cb - ca

    a = a + t2

    distance_matrix = cdist(a, b)
    max_dist_a_to_b = np.max(np.min(distance_matrix, axis=1))
    max_dist_b_to_a = np.max(np.min(distance_matrix, axis=0))

    if debug:
        print("a to b", np.argmax(np.min(distance_matrix, axis=1)), max_dist_a_to_b)
        print("b to a", np.argmax(np.min(distance_matrix, axis=0)), max_dist_b_to_a)

        d1, d2 = np.min(distance_matrix, axis=1), np.min(distance_matrix, axis=0)
        idx1, idx2 = np.argmax(d1), np.argmax(d2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        iss = []
        for i in range(a.shape[0]):
            p1 = a[i]
            p2 = b[i]
            d = d1[i]
            if d > 0.05:
                iss.append(i)
                print(i, d)
            #     ax.scatter3D(p1[0], p1[1], p1[2], color='m')
            # else:
            #     ax.scatter3D(p1[0], p1[1], p1[2], color='b', alpha=0.5)

        # ax.set_aspect('equal')
        # ax.set_xlim3d(left=a[idx1][0]-10, right=a[idx1][0]+10)
        # ax.set_ylim3d(bottom=a[idx1][1]-10, top=a[idx1][1]+10)
        # plt.show()

        return iss
    return max(max_dist_a_to_b, max_dist_b_to_a)


def hausdorff_distance(a, b):
    # dist = np.zeros(a.shape[0])
    # t = b - a
    # for i in range(a.shape[0]):
    #     dist[i] = max(compute_distance(a + t[i], b), compute_distance(b, a + t[i]))
    #
    # return np.min(dist)

    ca = np.average(a, axis=0)
    cb = np.average(b, axis=0)
    t2 = cb - ca

    dist2 = max(compute_distance(a + t2, b), compute_distance(b, a + t2))
    return dist2


def compute_distance(a, b):
    dist = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        dist[i] = np.min(np.linalg.norm(b - a[i], axis=1))
    return np.max(dist)


if __name__ == '__main__':
    a = np.loadtxt('../assets/skateboard_1372_50_spanning_2_sb.txt', delimiter=',')[:, :3]*3.4
    # a = np.loadtxt('../assets/chess_408_50_spanning_2_sb.txt', delimiter=',')[:, :3]*3.4
    # a = np.loadtxt('../assets/chess_100_50_spanning_2_sb.txt', delimiter=',')[:, :3]*2.3
    # a = np.loadtxt('../assets/dragon_1147_50_spanning_2_sb.txt', delimiter=',')[:, :3]*3.4
    # a = np.loadtxt('../assets/palm_725_50_spanning_2_sb.txt', delimiter=',')[:, :3]*3.4
    b = (1-(1.15/100))*a
    print(hausdorff_distance_optimized(a, b)*10)
