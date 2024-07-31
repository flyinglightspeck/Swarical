import trimesh
import numpy as np


def sample_points(mesh_file, target_distance, num_iterations=1000):
    # Load mesh
    mesh = trimesh.load(mesh_file)

    # Initialize sampled points
    sampled_points = []

    # Randomly select starting point
    starting_point = mesh.sample(1)[0]
    sampled_points.append(starting_point)

    # Iterate to sample points
    for _ in range(num_iterations):
        # Randomly select a point from existing sampled points
        idx = np.random.randint(len(sampled_points))
        current_point = sampled_points[idx]

        # Find nearest neighbors
        neighbors = mesh.kdtree.query(current_point, k=10)

        # Calculate distances to neighbors
        distances = np.linalg.norm(neighbors[0] - current_point, axis=1)

        # Move point towards neighbors
        for neighbor, distance in zip(neighbors[1], distances):
            move_distance = target_distance - distance
            move_vector = (neighbor - current_point) * (move_distance / distance)
            current_point += move_vector
            sampled_points.append(current_point)

    # Convert list of points to numpy array
    sampled_points = np.array(sampled_points)

    return sampled_points


# Example usage
mesh_file = "example_mesh.obj"
target_distance = 0.1  # Example target distance
sampled_points = sample_points(mesh_file, target_distance)
print("Number of sampled points:", len(sampled_points))
