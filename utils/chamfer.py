import numpy as np
from scipy.spatial import KDTree


def chamfer_distance(a, b):
    """Calculates the Chamfer distance between two individual point clouds.

    Args:
        a: A numpy array of shape (num_points_a, 3) representing the first point cloud.
        b: A numpy array of shape (num_points_b, 3) representing the second point cloud.

    Returns:
        The Chamfer distance between the two point clouds.
    """
    assert a.shape[1] == b.shape[1] == 3, "Point clouds must have 3D coordinates"

    # Distance from each point in a to its nearest neighbor in b
    a_nn_sq_dists = np.min(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1), axis=1)

    # Distance from each point in b to its nearest neighbor in a
    b_nn_sq_dists = np.min(np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=-1), axis=1)

    return np.mean(a_nn_sq_dists) + np.mean(b_nn_sq_dists)


def chamfer_distance_optimized(a, b):
    """Optimized Chamfer distance calculation between two individual point clouds.

    Args:
        a: A numpy array of shape (num_points_a, 3) representing the first point cloud.
        b: A numpy array of shape (num_points_b, 3) representing the second point cloud.

    Returns:
        The Chamfer distance between the two point clouds.
    """
    assert a.shape[1] == b.shape[1] == 3, "Point clouds must have 3D coordinates"

    ca = np.average(a, axis=0)
    cb = np.average(b, axis=0)
    t2 = cb - ca

    a = a + t2

    a_tree = KDTree(a)
    b_tree = KDTree(b)

    a_nn_sq_dists = a_tree.query(b)[0] ** 2
    b_nn_sq_dists = b_tree.query(a)[0] ** 2

    return np.mean(a_nn_sq_dists) + np.mean(b_nn_sq_dists)


if __name__ == "__main__":
    point_cloud_a = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 1]])
    point_cloud_b = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
    # point_cloud_b = np.array([[0.1, 0.2, 0], [1, 1.1, 0.1], [-0.1, 1, 0.8]])

    distance = chamfer_distance_optimized(point_cloud_a, point_cloud_b)
    print(distance)
