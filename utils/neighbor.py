import numpy as np
from scipy.spatial import distance


def knn(points, k):
    d = distance.squareform(distance.pdist(points))
    closest = np.argsort(d, axis=1)
    return closest[:, 1:k + 1]
