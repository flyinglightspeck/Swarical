import json
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import os

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

root_color = '#0871f5'
primary_color = '#f98557'
anchor_color = '#ef80fd'
fls_color = '#3fafff'

line_color = '#dcdcdc'

primary_marker = 's'
root_marker = 'D'
anchor_marker = 'v'
fls_marker = 'o'
sizes = {
    's': 25,
    'v': 30,
    'o': 25,
    'D': 25,
}


def read_fls_tree(path):
    intra_localizer_path = path + '_intra_localizer.json'
    with open(intra_localizer_path) as f:
        intra_localizer = json.load(f)

    fls_tree = nx.Graph()
    for i, j in intra_localizer.items():
        fls_tree.add_edge(int(i), int(j))

    # nx.draw(fls_tree)


def read_swarm_tree(path):
    localizer_path = path + '_localizer.json'
    intra_localizer_path = path + '_intra_localizer.json'
    with open(intra_localizer_path) as f:
        intra_localizer = json.load(f)
    with open(localizer_path) as f:
        localizer = json.load(f)

    points = np.loadtxt(f'{path}.txt', delimiter=',')

    sids = points[:, 3].astype(int).tolist()
    fids = points[:, 4].astype(int).tolist()
    coords = points[:, 0:3]
    fid_to_sid = dict(zip(fids, sids))
    fid_to_coord = dict(zip(fids, coords))
    sid_to_pid = dict()
    sid_to_coord = dict()

    for row in points:
        gid = int(row[3])
        if gid in sid_to_coord:
            sid_to_coord[gid].append(row[0:3])
        else:
            sid_to_coord[gid] = [row[0:3]]

    sid_to_centroid = {sid: np.mean(np.array(coords), axis=0) for sid, coords in sid_to_coord.items()}

    fls_tree = nx.DiGraph()
    for i, j in intra_localizer.items():
        i = int(i)
        if fid_to_sid[i] == 0:  # for viz only
            fls_tree.add_edge(i, j)
        if str(j) not in intra_localizer and str(j) not in localizer:
            sid_to_pid[0] = j

    swarm_tree = nx.DiGraph()
    for i, js in localizer.items():
        for j in js:
            if j[1] is not None:
                i = int(i)
                sid_to_pid[j[1]] = i
                # if fid_to_sid[i] == 0:
                if fid_to_sid[i] == 0 or fid_to_sid[i] == 1 or fid_to_sid[i] == 2 or fid_to_sid[i] == 3:
                    fls_tree.add_edge(i, j[0])
                swarm_tree.add_edge(fid_to_sid[i], fid_to_sid[j[0]])

    # print(sid_to_pid)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Draw nodes
    # nx.draw_networkx_nodes(fls_tree, fid_to_coord, node_color='blue', ax=ax)

    # Draw edges
    # nx.draw_networkx_edges(fls_tree, fid_to_coord, ax=ax)
    print(len(sid_to_coord[7]))
    nx.draw_kamada_kawai(swarm_tree, with_labels=True, arrows=True)
    plt.show()
    # exit()

    with plt.style.context('seaborn-white'):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        x = []
        y = []
        z = []
        u = []
        v = []
        w = []
        colors = {}
        arrow_colors = []
        markers = {}
        for k, (i, j) in enumerate(fls_tree.edges()):
            x.append(fid_to_coord[i][0])
            y.append(fid_to_coord[i][1])
            z.append(fid_to_coord[i][2])
            u.append(fid_to_coord[j][0] - fid_to_coord[i][0])
            v.append(fid_to_coord[j][1] - fid_to_coord[i][1])
            w.append(fid_to_coord[j][2] - fid_to_coord[i][2])
            color = fls_color
            marker = fls_marker

            if fid_to_sid[i] == 0:
                ax.plot([fid_to_coord[i][0], fid_to_coord[j][0]],
                        [fid_to_coord[i][1], fid_to_coord[j][1]],
                        [fid_to_coord[i][2], fid_to_coord[j][2]], color=line_color, zorder=1)

            if fid_to_sid[i] != 0:
                color = primary_color
                marker = primary_marker
                colors[j] = anchor_color
                markers[j] = anchor_marker
                # ax.text(fid_to_coord[i][0], fid_to_coord[i][1], fid_to_coord[i][2], "primary")
            if i not in colors:
                colors[i] = color
                markers[i] = marker
            arrow_colors.append(color)

        # Q = ax.quiver(x, y, z, u, v, w, color='#ccc', arrow_length_ratio=0.2, zorder=2)
        root = fid_to_coord[sid_to_pid[0]]
        markers = list(markers.values())
        colors = list(colors.values())
        for i, (p, q, m) in enumerate(zip(x, y, z)):
            if markers[i] == primary_marker:
                continue
            ax.scatter3D(p, q, m,
                         marker=markers[i],
                         color=colors[i], depthshade=False, zorder=2, s=sizes[markers[i]])
        ax.scatter3D([root[0]], [root[1]], [root[2]], marker=root_marker, color=root_color, depthshade=False, zorder=2, s=25)
        # ax.text(root[0], root[1], root[2], "primary")

        ax.set_aspect('equal')

        ax.view_init(azim=-110, elev=20)
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.axis('off')
        # plt.savefig(f"{shape}.png", dpi=300)

        # print(np.stack(sid_to_coord[0]))
        # exit()
        cmap = mpl.colormaps.get_cmap('plasma')
        # cmap = mpl.colormaps.get_cmap('viridis')
        for sid, coords in sid_to_coord.items():
            s_points = np.stack(coords)
            normalized_value = sid / 9
            swarm_color = cmap(normalized_value)[:3] + (0.25,)
            centroid_color = cmap(normalized_value)[:3] + (1.0,)

            # group points
            # ax2.scatter3D(s_points[:, 0], s_points[:, 1], s_points[:, 2])

            # centroid
            ax2.scatter3D(sid_to_centroid[sid][0], sid_to_centroid[sid][1], sid_to_centroid[sid][2], color=centroid_color, s=40,
                          depthshade=False)
            hull = ConvexHull(s_points)
            boundary_points = s_points[hull.vertices, :]
            # boundary_points = ensure_counterclockwise_order(boundary_points)
            # face_color = (0.25, 1 - sid / 10, sid / 10, 0.25)

            for simplex in hull.simplices:
                collection = Poly3DCollection([s_points[simplex]], facecolors=[swarm_color], zorder=0)
                ax2.add_collection3d(collection)
            # ax2.plot(s_0_points[simplex, 0], s_0_points[simplex, 1], s_0_points[simplex, 2], 'k-')
        # ax2.plot(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2])

        for i, j in swarm_tree.edges:
            ax2.plot([sid_to_centroid[i][0], sid_to_centroid[j][0]],
                     [sid_to_centroid[i][1], sid_to_centroid[j][1]],
                     [sid_to_centroid[i][2], sid_to_centroid[j][2]], color='k', zorder=1)

        # inter-group anchors and primaries
        # for i, js in localizer.items():
        #     for j in js:
        #         if j[1] is not None:
        #             i = int(i)
        #             ax2.plot([fid_to_coord[i][0], fid_to_coord[j[0]][0]],
        #                      [fid_to_coord[i][1], fid_to_coord[j[0]][1]],
        #                      [fid_to_coord[i][2], fid_to_coord[j[0]][2]], color=line_color, zorder=1)
        #             ax2.scatter3D(fid_to_coord[i][0], fid_to_coord[i][1], fid_to_coord[i][2], color=primary_color,
        #                           marker=primary_marker, zorder=2)
        #             ax2.scatter3D(fid_to_coord[j[0]][0], fid_to_coord[j[0]][1], fid_to_coord[j[0]][2], color=anchor_color,
        #                           marker=anchor_marker, zorder=2)
        x = [p[0] for p in sid_to_centroid.values()]
        y = [p[1] for p in sid_to_centroid.values()]
        z = [p[2] for p in sid_to_centroid.values()]
        # ax2.scatter3D(x, y, z, s=30, depthshade=False)
        ax2.set_aspect('equal')
        ax2.view_init(azim=-116, elev=27)
        ax2.grid(False)
        ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.axis('off')
        # plt.savefig('figs/trees.png', dpi=300)
        plt.show()


def find_equidistant_point(point1, point2, distance):
    """
    Finds a point that is equidistant from two given 3D coordinates.

    Args:
        point1 (list or tuple): 3D coordinates of the first point (x1, y1, z1).
        point2 (list or tuple): 3D coordinates of the second point (x2, y2, z2).
        distance (float): The desired distance between the points.

    Returns:
        list or tuple: 3D coordinates of the equidistant point (x, y, z).
        None: If there is no valid equidistant point (e.g., distance too large).
    """

    midpoint = (np.array(point2) + np.array(point1)) / 2

    direction_vector = np.array(point2) - np.array(point1)

    if np.abs(direction_vector[0]) < np.abs(direction_vector[1]) and np.abs(direction_vector[0]) < np.abs(direction_vector[2]):
        perpendicular_vector = np.array([0, -direction_vector[2], direction_vector[1]])
    else:
        perpendicular_vector = np.array([0, direction_vector[2], -direction_vector[1]])

    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    unit_perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    mid_distance = np.linalg.norm(direction_vector / 2)
    perpendicular_distance = np.sqrt(distance**2 - mid_distance**2)

    offset = unit_direction_vector * mid_distance + unit_perpendicular_vector * perpendicular_distance

    equidistant_point_1 = point1 + offset

    # print(point1, point2, equidistant_point_1, distance)
    # print(np.linalg.norm(equidistant_point_1 - point1))
    # print(np.linalg.norm(direction_vector))

    # Check if the distance constraint is valid
    if np.linalg.norm(equidistant_point_1 - point1) <= distance + 1e-6:  # Tolerance for floating-point errors
        return equidistant_point_1.tolist(), np.linalg.norm(equidistant_point_1 - point1), np.linalg.norm(equidistant_point_1 - point2)
    else:
        return None  # No valid point within the distance


def add_standbys(path, min=1.8, th=2.3, s=3.4):
    localizer_path = path + '_localizer.json'
    intra_localizer_path = path + '_intra_localizer.json'
    with open(intra_localizer_path) as f:
        intra_localizer = json.load(f)
    with open(localizer_path) as f:
        localizer = json.load(f)

    points = np.loadtxt(f'{path}.txt', delimiter=',')

    sids = points[:, 3].astype(int).tolist()
    fids = points[:, 4].astype(int).tolist()
    coords = points[:, 0:3]
    fid_to_sid = dict(zip(fids, sids))
    fid_to_coord = dict(zip(fids, coords))
    sid_to_pid = dict()
    sid_to_coord = dict()

    for row in points:
        gid = int(row[3])
        if gid in sid_to_coord:
            sid_to_coord[gid].append(row[0:3])
        else:
            sid_to_coord[gid] = [row[0:3]]

    sid_to_centroid = {sid: np.mean(np.array(coords), axis=0) for sid, coords in sid_to_coord.items()}

    new_fid = max(fids) + 1
    new_intra_localizer = deepcopy(intra_localizer)
    new_points = []
    dists = []

    # fls_tree = nx.DiGraph()
    fls_trees = {}
    for sid in sids:
        fls_trees[sid] = nx.DiGraph()

    for i, j in intra_localizer.items():
        i = int(i)
        coord_i = fid_to_coord[i]
        coord_j = fid_to_coord[j]
        dist_ij = np.linalg.norm(coord_j - coord_i)
        if dist_ij > th:
            new_dist = (min + th + 0.1) / 2
            if dist_ij / 2 > new_dist:
                new_dist = dist_ij / 2
            if new_dist > th + 0.1:
                print("Exceeded fls tree", new_dist)
                # return
            standby_coord, d1, d2 = find_equidistant_point(coord_i, coord_j, new_dist)
            sid_to_coord[fid_to_sid[i]].append(standby_coord)

            # new_intra_localizer.append({str(i): new_fid, str(new_fid): j})
            new_intra_localizer[str(i)] = new_fid
            new_intra_localizer[str(new_fid)] = j
            new_points.append([*standby_coord, fid_to_sid[i], new_fid])
            dists.append(d1)
            dists.append(d2)
            fls_trees[fid_to_sid[i]].add_edge(new_fid, i)
            fls_trees[fid_to_sid[i]].add_edge(j, new_fid)
            new_fid += 1
        else:
            fls_trees[fid_to_sid[i]].add_edge(j, i)
            dists.append(dist_ij)

        if str(j) not in intra_localizer and str(j) not in localizer:
            sid_to_pid[0] = j


    dists_2 = []
    swarm_tree = nx.DiGraph()
    new_localizer = deepcopy(localizer)
    for i, js in localizer.items():
        for anchor_idx, k in enumerate(js):
            if k[1] is not None:
                i = int(i)  # localizer
                j = k[0]  # anchor
                sid_to_pid[k[1]] = i

                coord_i = fid_to_coord[i]
                coord_j = fid_to_coord[j]
                dist_ij = np.linalg.norm(coord_j - coord_i)
                if dist_ij > th:
                    new_dist = (min + th + 0.1) / 2
                    if dist_ij / 2 > new_dist:
                        new_dist = dist_ij / 2
                    if new_dist > th + 0.1:
                        print("Exceeded swarm tree", new_dist)
                        # return
                    standby_coord, d1, d2 = find_equidistant_point(coord_i, coord_j, new_dist)
                    anchor_sid = fid_to_sid[j]
                    # add a point to anchor swarm: i -> (new_fid -> j)
                    new_points.append([*standby_coord, anchor_sid, new_fid])
                    sid_to_coord[anchor_sid].append(standby_coord)
                    new_intra_localizer[str(new_fid)] = j
                    fls_trees[fid_to_sid[j]].add_edge(j, new_fid)
                    # change the anchor for this fls
                    # print(i)
                    new_localizer[str(i)][anchor_idx][0] = new_fid
                    new_localizer[str(new_fid)] = [[i, None]]
                    cur_j = new_localizer[str(j)]
                    # print(i, j)
                    if len(cur_j) == 1:
                        new_localizer.pop(str(j))
                    else:
                        new_j = []
                        for cj in cur_j:
                            if cj[0] != i:
                                new_j.append(cj)
                            # else:
                                # print(cj)
                        new_localizer[str(j)] = new_j
                    dists_2.append(d1)
                    dists_2.append(d2)
                    new_fid += 1
                else:
                    dists_2.append(dist_ij)
                # swarm_tree.add_edge(fid_to_sid[i], fid_to_sid[j])
                swarm_tree.add_edge(fid_to_sid[j], fid_to_sid[i])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax1.hist(np.array(dists) * s)
    ax2.hist(np.array(dists_2) * s)
    plt.show()
    return

    A = np.vstack((points, new_points))
    # print(points)
    # print(len(new_points))
    # print(A)

    bfs_tree = nx.bfs_tree(swarm_tree, source=0)
    bf_across_groups = list(dict(bfs_tree.out_degree()).values())

    bf_in_groups = []
    for sid, t in fls_trees.items():
        if sid in sid_to_pid:
            source = sid_to_pid[sid]
        else:
            source = min(t.nodes)
        bfs_tree = nx.bfs_tree(t, source=source)
        bf_in_groups += dict(bfs_tree.out_degree()).values()
    stats = {
        "added_points": len(new_points),
        "dist_in_groups": dists,
        "dist_across_groups": dists_2,
        "bf_across_groups": bf_across_groups,
        "bf_in_groups": bf_in_groups,
        "group_size": [len(g) for g in sid_to_coord.values()]
    }

    np.savetxt(f"{path}_sb.txt", A, delimiter=',')

    with open(f"{path}_sb_localizer.json", "w") as f:
        json.dump(new_localizer, f)
    with open(f"{path}_sb_intra_localizer.json", "w") as f:
        json.dump(new_intra_localizer, f)
    with open(f"{path}_sb_stats.json", "w") as f:
        json.dump(stats, f)

    with plt.style.context('seaborn-white'):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        x = []
        y = []
        z = []
        u = []
        v = []
        w = []
        colors = {}
        arrow_colors = []
        markers = {}
        for k, (i, j) in enumerate(fls_tree.edges()):
            x.append(fid_to_coord[i][0])
            y.append(fid_to_coord[i][1])
            z.append(fid_to_coord[i][2])
            u.append(fid_to_coord[j][0] - fid_to_coord[i][0])
            v.append(fid_to_coord[j][1] - fid_to_coord[i][1])
            w.append(fid_to_coord[j][2] - fid_to_coord[i][2])
            color = fls_color
            marker = fls_marker

            if fid_to_sid[i] == 0:
                ax.plot([fid_to_coord[i][0], fid_to_coord[j][0]],
                        [fid_to_coord[i][1], fid_to_coord[j][1]],
                        [fid_to_coord[i][2], fid_to_coord[j][2]], color=line_color, zorder=1)

            if fid_to_sid[i] != 0:
                color = primary_color
                marker = primary_marker
                colors[j] = anchor_color
                markers[j] = anchor_marker
                # ax.text(fid_to_coord[i][0], fid_to_coord[i][1], fid_to_coord[i][2], "primary")
            if i not in colors:
                colors[i] = color
                markers[i] = marker
            arrow_colors.append(color)

        # Q = ax.quiver(x, y, z, u, v, w, color='#ccc', arrow_length_ratio=0.2, zorder=2)
        root = fid_to_coord[sid_to_pid[0]]
        markers = list(markers.values())
        colors = list(colors.values())
        for i, (p, q, m) in enumerate(zip(x, y, z)):
            if markers[i] == primary_marker:
                continue
            ax.scatter3D(p, q, m,
                         marker=markers[i],
                         color=colors[i], depthshade=False, zorder=2, s=sizes[markers[i]])
        ax.scatter3D([root[0]], [root[1]], [root[2]], marker=root_marker, color=root_color, depthshade=False, zorder=2, s=25)
        # ax.text(root[0], root[1], root[2], "primary")

        ax.set_aspect('equal')

        ax.view_init(azim=-110, elev=20)
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.axis('off')
        # plt.savefig(f"{shape}.png", dpi=300)

        # print(np.stack(sid_to_coord[0]))
        # exit()
        cmap = mpl.colormaps.get_cmap('plasma')
        # cmap = mpl.colormaps.get_cmap('viridis')
        for sid, coords in sid_to_coord.items():
            s_points = np.stack(coords)
            normalized_value = sid / 9
            swarm_color = cmap(normalized_value)[:3] + (0.25,)
            centroid_color = cmap(normalized_value)[:3] + (1.0,)

            # group points
            # ax2.scatter3D(s_points[:, 0], s_points[:, 1], s_points[:, 2])

            # centroid
            ax2.scatter3D(sid_to_centroid[sid][0], sid_to_centroid[sid][1], sid_to_centroid[sid][2], color=centroid_color, s=40,
                          depthshade=False)
            hull = ConvexHull(s_points)
            boundary_points = s_points[hull.vertices, :]
            # boundary_points = ensure_counterclockwise_order(boundary_points)
            # face_color = (0.25, 1 - sid / 10, sid / 10, 0.25)

            for simplex in hull.simplices:
                collection = Poly3DCollection([s_points[simplex]], facecolors=[swarm_color], zorder=0)
                ax2.add_collection3d(collection)
            # ax2.plot(s_0_points[simplex, 0], s_0_points[simplex, 1], s_0_points[simplex, 2], 'k-')
        # ax2.plot(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2])

        for i, j in swarm_tree.edges:
            ax2.plot([sid_to_centroid[i][0], sid_to_centroid[j][0]],
                     [sid_to_centroid[i][1], sid_to_centroid[j][1]],
                     [sid_to_centroid[i][2], sid_to_centroid[j][2]], color='k', zorder=1)

        # inter-group anchors and primaries
        # for i, js in localizer.items():
        #     for j in js:
        #         if j[1] is not None:
        #             i = int(i)
        #             ax2.plot([fid_to_coord[i][0], fid_to_coord[j[0]][0]],
        #                      [fid_to_coord[i][1], fid_to_coord[j[0]][1]],
        #                      [fid_to_coord[i][2], fid_to_coord[j[0]][2]], color=line_color, zorder=1)
        #             ax2.scatter3D(fid_to_coord[i][0], fid_to_coord[i][1], fid_to_coord[i][2], color=primary_color,
        #                           marker=primary_marker, zorder=2)
        #             ax2.scatter3D(fid_to_coord[j[0]][0], fid_to_coord[j[0]][1], fid_to_coord[j[0]][2], color=anchor_color,
        #                           marker=anchor_marker, zorder=2)
        x = [p[0] for p in sid_to_centroid.values()]
        y = [p[1] for p in sid_to_centroid.values()]
        z = [p[2] for p in sid_to_centroid.values()]
        # ax2.scatter3D(x, y, z, s=30, depthshade=False)
        ax2.set_aspect('equal')
        ax2.view_init(azim=-116, elev=27)
        ax2.grid(False)
        ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.axis('off')
        plt.savefig('figs/trees.png', dpi=300)
        # plt.show()
