import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter


color_map = {
    '+X': 'r',
    '+Z': 'g',
    '-Z': 'b',
}

faces = {
        '+X': np.array([1, 0, 0]),
        '-X': np.array([-1, 0, 0]),
        '+Y': np.array([0, 1, 0]),
        '-Y': np.array([0, -1, 0]),
        '+Z': np.array([0, 0, 1]),
        '-Z': np.array([0, 0, -1])
    }

class DuplicateHeading(Exception):
    def __init__(self):
        Exception.__init__(self)


def find_cube_side(vector):
    max_dot_product = -1  # Start with a low value
    closest_side = None

    for side, normal_vector in faces.items():
        dot_product = np.dot(vector, normal_vector)
        if dot_product > max_dot_product:
            max_dot_product = dot_product
            closest_side = side

    return closest_side


def draw_cube():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define vertices of a cube (adjust for desired size and position)
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]

    edges = [
        (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)
    ]

    # Extract x, y, z coordinates
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]

    # Plot edges of the cube
    for start, end in edges:
        ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], color='gray')

    diameters = [
        (0, 6), (1, 7), (2, 4), (3, 5)  # Long diagonals
    ]

    # Plot diameters
    for start, end in diameters:
        ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], color='orange')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.quiver(0.5, 0.5, 0.5, *faces['+X'], color=color_map['+X'])
    ax.quiver(0.5, 0.5, 0.5, *faces['+Z'], color=color_map['+Z'])
    ax.quiver(0.5, 0.5, 0.5, *faces['-Z'], color=color_map['-Z'])
    plt.savefig('arrangement.png', dpi=300)
    # plt.show()


if __name__ == "__main__":
    # with plt.style.context('seaborn-white'):
    #     draw_cube()
    # exit()
    # n = 4
    visualize = True

    # for n in [6]:
    # for n in [4, 6, 8, 10, 14, 20]:
    # for shape in ["chess_544", "skateboard_1912", "dragon_1020"]:
    # for shape in ["chess_408", "skateboard_1372", "dragon_1147", "palm_725", "racecar_3720"]:
    for shape in ["skateboard_1372"]:
    # for shape in ["chess_100"]:
    #     shape = f"grid_{n*n}"

        if visualize:
            mpl.use('macosx')

        A = np.loadtxt(f'../assets/{shape}_50_spanning_2_sb.txt', delimiter=',')
        with open(f'../assets/{shape}_50_spanning_2_sb_localizer.json') as f:
            localizer = json.load(f)
        with open(f'../assets/{shape}_50_spanning_2_sb_intra_localizer.json') as f:
            intra_localizer = json.load(f)
        # pid to coord
        P = {int(row[4]): row[:3] for row in A}
        # pid to camera heading
        camera_heading = {}
        for loc, ancs in localizer.items():
            for anc in ancs:
                if anc[1] is not None:
                    h = P[anc[0]] - P[int(loc)]
                    if loc in camera_heading:
                        raise DuplicateHeading
                    camera_heading[int(loc)] = h
        for loc, anc in intra_localizer.items():
            h = P[anc] - P[int(loc)]
            if loc in camera_heading:
                raise DuplicateHeading
            camera_heading[int(loc)] = h

        camera_placement = {}
        simplified_camera_placement = {}
        for pid, h in camera_heading.items():
            side = find_cube_side(h)
            camera_placement[pid] = side
            if side == '-X':
                simplified_camera_placement[pid] = ('+X', 180)
            elif side == '+Y':
                simplified_camera_placement[pid] = ('+X', 90)
            elif side == '-Y':
                simplified_camera_placement[pid] = ('+X', 270)
            else:
                simplified_camera_placement[pid] = (side, 0)

        if visualize:
            with plt.style.context('seaborn-white'):
                fig = plt.figure(figsize=(15, 5))
                ax = fig.add_subplot(131, projection='3d')
                ax2 = fig.add_subplot(132, projection='3d')
                ax3 = fig.add_subplot(133)
                x = []
                y = []
                z = []
                u = []
                v = []
                w = []
                colors = []
                for pid, h in camera_heading.items():
                    x.append(P[pid][0])
                    y.append(P[pid][1])
                    z.append(P[pid][2])
                    u.append(h[0])
                    v.append(h[1])
                    w.append(h[2])
                    colors.append(color_map[simplified_camera_placement[pid][0]])
                    # ax.text(P[pid][0], P[pid][1], P[pid][2], str(pid))
                Q = ax.quiver(x, y, z, u, v, w, colors=colors, arrow_length_ratio=0.5)

                ax.set_aspect('equal')
                ax2.scatter3D(A[:, 0], A[:, 1], A[:, 2], color='blue', s=1.5, depthshade=True)
                ax2.set_aspect('equal')
                # ax.view_init(azim=ax.azim+90)
                # ax2.view_init(azim=ax2.azim+90)
                hist = Counter([h[0] for h in simplified_camera_placement.values()])
                hist["+X"] += 1
                percent = {x: 100*(y/A.shape[0]) for x, y in hist.items()}
                plt.bar(range(len(hist)), hist.values(), color=[color_map[h] for h in hist.keys()])
                plt.xticks(range(len(hist)), hist.keys())
                plt.savefig(f"{shape}_arrangement.png", dpi=300)
                # plt.show()
                print(shape, "total number of FLSs: ", A.shape[0],
                      f"number of each variant: {' '.join([f'{k}:{v}' for k, v in hist.items()])}",
                      f"percent of each variant: {' '.join([f'{k}:{v}' for k, v in percent.items()])}")
