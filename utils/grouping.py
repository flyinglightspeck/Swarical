import itertools
import json
import math
from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse import csr_matrix
from matching.games import StableRoommates
import networkx.algorithms.approximation as nx_app
import networkx as nx


def construct_graph(points):
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=distance)

    return G


def tsp(points):
    return nx_app.christofides(construct_graph(points), weight="weight")


def mwm(points):
    return nx.min_weight_matching(construct_graph(points))


def matching_bi(points):
    distance_matrix = cdist(points, points)

    np.fill_diagonal(distance_matrix, np.inf)
    distance_matrix = csr_matrix(distance_matrix)
    row_ind, col_ind = min_weight_full_bipartite_matching(distance_matrix)

    matches = list(zip(row_ind, col_ind))
    return matches


def k_means(points, k=4):
    # 10
    kmeans = KMeans(n_clusters=k, random_state=11)
    kmeans.fit(points)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids


def matching_lsa(points):
    distance_matrix = squareform(pdist(points))

    np.fill_diagonal(distance_matrix, np.inf)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    matches = list(zip(row_ind, col_ind))
    return matches


def matching_kd(point_cloud):
    tree = KDTree(point_cloud)

    pairs = []
    for i, point in enumerate(point_cloud):
        dists, indexes = tree.query(point, k=2)
        nearest_neighbor_index = indexes[1]
        pairs.append((i, nearest_neighbor_index))

    return pairs


def greedy_matching(point_cloud):
    dist_matrix = squareform(pdist(point_cloud))
    pairs = []
    grouped_points = set()

    for i in range(point_cloud.shape[0]):
        if i not in grouped_points:  # Check if point is ungrouped
            for j in range(len(dist_matrix[i])):
                nearest_neighbor_idx = np.argmin(dist_matrix[i])
                if nearest_neighbor_idx != i and nearest_neighbor_idx not in grouped_points:
                    pairs.append((i, nearest_neighbor_idx))
                    grouped_points.add(i)
                    grouped_points.add(nearest_neighbor_idx)
                    break
                else:
                    dist_matrix[i][nearest_neighbor_idx] = np.inf

    return pairs


def sr_matching(points):
    d = squareform(pdist(points))
    closest = np.argsort(d, axis=1)
    preferences = {
        p[0]: p[1:] for p in closest
    }

    game = StableRoommates.create_from_dictionary(preferences)
    solution = game.solve()

    return [(m.name, n.name) for m, n in solution.items()]


def get_kmeans_groups(A, k):
    assignments, centroids = k_means(A, k=k)

    # map of group id to list of point indexes
    gid_to_idx = {}
    for i, a in enumerate(assignments):
        if a in gid_to_idx:
            gid_to_idx[a].append(i)
        else:
            gid_to_idx[a] = [i]

    # maximum distance between points for each point in a group
    max_dist = {}
    for gid, g in gid_to_idx.items():
        max_dist[gid] = np.max(squareform(pdist(A[g])), axis=1)

    return gid_to_idx, max_dist, assignments, centroids


def create_hierarchical_groups(A, G, shape, visualize=True):
    k = int(2 ** np.ceil(np.log2(A.shape[0] / G)))

    h_groups, max_dist_1, _, h_centroids = get_kmeans_groups(A, k)

    gid = k
    localizer = {}
    dists = []

    height = int(np.log2(k))
    while height > 0:
        pairs = mwm(h_centroids)
        offset = 2 * (gid - k)
        new_centroids = [(h_centroids[i] + h_centroids[j]) / 2 for i, j in pairs]
        # print(pairs)
        for i, j in pairs:
            l_gid = i + offset
            r_gid = j + offset
            l_group = h_groups[l_gid]
            r_group = h_groups[r_gid]
            if height > 1:
                h_groups[gid] = l_group + r_group
            xdist = cdist(A[l_group], A[r_group])
            am = np.argmin(xdist)
            dists.append(xdist[am // len(r_group), am % len(r_group)])
            l_idx = l_group[am // len(r_group)]
            r_idx = r_group[am % len(r_group)]
            if l_idx in localizer:
                localizer[l_idx].append((r_idx, l_gid))
            else:
                localizer[l_idx] = [(r_idx, l_gid)]
            if r_idx in localizer:
                localizer[r_idx].append((l_idx, r_gid))
            else:
                localizer[r_idx] = [(l_idx, r_gid)]
            gid += 1
        h_centroids = np.array(new_centroids)
        height -= 1

    # print(h_groups)
    # print(localizer)
    # print(dists)

    point_groups = []
    for i in range(len(A)):
        point_groups.append([])

    for gid, pts in h_groups.items():
        for p in pts:
            point_groups[p].append(gid)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax.hist(max_dist_1)
    ax2.hist(dists)
    plt.savefig(f"../assets/{shape}_hierarchical_hist.png")

    np.savetxt(f"../assets/{shape}_hierarchical.txt", np.hstack((A, point_groups)), delimiter=',')

    with open(f"../assets/{shape}_hierarchical_localizer.json", "w") as f:
        json.dump(localizer, f)


def create_overlapping_groups(A, G, shape, visualize=True):
    k = int(2 ** np.ceil(np.log2(A.shape[0] / G)))
    groups, max_dist_1, assignments, centroids = get_kmeans_groups(A, k)

    C1 = tsp(centroids)

    # edges in the cover cycle
    edge_list = list(nx.utils.pairwise(C1))
    gid = len(centroids)
    groups_2 = {2 * gid - 1: []}
    for e in edge_list:
        l_centroid = centroids[e[0]]
        r_centroid = centroids[e[1]]
        m_centroid = (l_centroid + r_centroid) / 2
        l_half = math.ceil(len(groups[e[0]]) / 2)
        l_points = A[groups[e[0]]]
        l_closest_idx = np.argsort(cdist([m_centroid], l_points), axis=1)[0]
        l_idx = l_closest_idx[:l_half]
        l_remain_idx = l_closest_idx[l_half:]
        m_idx = np.array(groups[e[0]])[l_idx].tolist()
        prev_idx = np.array(groups[e[0]])[l_remain_idx].tolist()
        if gid in groups_2:
            groups_2[gid] += m_idx
        else:
            groups_2[gid] = m_idx
        prev_gid = gid - 1
        if prev_gid < len(edge_list):
            prev_gid = len(edge_list) * 2 - 1
        groups_2[prev_gid] += prev_idx
        gid += 1

    assignments_2 = [0] * len(assignments)
    for gid, idxs in groups_2.items():
        for idx in idxs:
            assignments_2[idx] = gid

    asgn_1 = assignments.reshape(-1, 1)
    asgn_2 = np.array(assignments_2).reshape(-1, 1)
    np.savetxt(f"../assets/{shape}_overlapping.txt", np.hstack((A, asgn_1, asgn_2)), delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    max_dist_2 = []

    for g in groups_2.values():
        max_dist_2.append(np.max(squareform(pdist(A[g]))))

    np.savetxt(f"../assets/{shape}_overlapping_max_dist.txt", np.vstack((max_dist_1, max_dist_2)), delimiter=',')

    ax.hist(max_dist_1)
    ax2.hist(max_dist_2)
    plt.savefig(f"../assets/{shape}_hist.svg")

    if visualize:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        for g in groups.values():
            xs = [A[p][0] for p in g]
            ys = [A[p][1] for p in g]
            zs = [A[p][2] for p in g]
            ax.plot3D(xs, ys, zs, '-o')

        for g in groups_2.values():
            xs = [A[p][0] for p in g]
            ys = [A[p][1] for p in g]
            zs = [A[p][2] for p in g]
            ax2.plot3D(xs, ys, zs, '-o')
        # cycle
        xs = [centroids[p][0] for p in C1]
        ys = [centroids[p][1] for p in C1]
        zs = [centroids[p][2] for p in C1]
        ax3.plot3D(xs, ys, zs, '-o')
        # matching
        # for i, j in P1:
        #     p1 = centroids[i]
        #     p2 = centroids[j]
        #     ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '-o')
        plt.savefig(f"../assets/{shape}_groups.svg")


def create_binary_overlapping_groups(A, shape, visualize=True):
    G = construct_graph(A)
    T = nx.minimum_spanning_tree(G)

    assignments = []
    for i in range(A.shape[0]):
        assignments.append([])

    for i, e in enumerate(T.edges):
        assignments[e[0]].append(i)
        assignments[e[1]].append(i)

    np.savetxt(f"../assets/{shape}.txt", A, delimiter=',')

    with open(f"../assets/{shape}_bin_overlapping.json", "w") as f:
        json.dump(assignments, f)

    if visualize:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        # ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], depthshade=False)
        for e in T.edges:
            print(e)
            ax.plot3D(A[e, 0], A[e, 1], A[e, 2], '-o')
        # ax.plot3D(A[T, 0], A[T, 1], A[T, 2] + 1, '-o')
        plt.show()


def create_spanning_tree_groups(A, G, shape, visualize):
    # k = int(2 ** np.ceil(np.log2(A.shape[0] / G)))
    k = int(1.5 * A.shape[0]) // G

    groups, max_dist_1, assignments, centroids = get_kmeans_groups(A, k)

    T = nx.minimum_spanning_tree(construct_graph(centroids))

    degrees = dict(T.degree())
    max_degree_node = max(degrees, key=degrees.get)
    bfs_tree = nx.bfs_tree(T, source=max_degree_node)
    bfs_tree_out_degree = bfs_tree.out_degree()
    print(f"grid{shape}BF = {{{','.join(map(lambda x: str(x), list(dict(bfs_tree_out_degree).values())))}}}")
    print(f"grid{shape}KS = {{{','.join(list(map(lambda x: str(len(x)), groups.values())))}}}")
    bfs_order = list(bfs_tree)

    bfs_order_gid = {bfs_order[i]: i for i in range(len(bfs_order))}

    localizer = {}
    dists = []
    for i, j in T.edges:
        l_gid = i
        r_gid = j
        l_b_gid = bfs_order_gid[l_gid]
        r_b_gid = bfs_order_gid[r_gid]
        l_group = groups[l_gid]
        r_group = groups[r_gid]

        xdist = cdist(A[l_group], A[r_group])
        am = np.argmin(xdist)
        dists.append(xdist[am // len(r_group), am % len(r_group)])
        l_idx = l_group[am // len(r_group)]
        r_idx = r_group[am % len(r_group)]
        if l_idx in localizer:
            localizer[l_idx].append((r_idx, l_b_gid))
        else:
            localizer[l_idx] = [(r_idx, l_b_gid)]
        if r_idx in localizer:
            localizer[r_idx].append((l_idx, r_b_gid))
        else:
            localizer[r_idx] = [(l_idx, r_b_gid)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax.hist(max_dist_1)
    ax2.hist(dists)
    plt.savefig(f"../assets/{shape}_spanning_hist.png")
    new_gid = [bfs_order_gid[a] for a in assignments]
    np.savetxt(f"../assets/{shape}_spanning.txt", np.hstack((A, np.array(new_gid).reshape(-1, 1))), delimiter=',')

    with open(f"../assets/{shape}_spanning_localizer.json", "w") as f:
        json.dump(localizer, f)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], depthshade=False)
        for g in groups.values():
            xs = [A[p][0] for p in g]
            ys = [A[p][1] for p in g]
            zs = [A[p][2] for p in g]
            ax.plot3D(xs, ys, zs, '-bo')

        for i, l in localizer.items():
            for p in l:
                ax.plot3D(A[[i, p[0]], 0], A[[i, p[0]], 1], A[[i, p[0]], 2], '-ro')

        # ax.plot3D(A[T, 0], A[T, 1], A[T, 2] + 1, '-o')
        plt.show()


def create_spanning_tree_groups_2(A, G, shape, visualize):
    k = math.ceil(1 * A.shape[0] / G)
    groups, radio_range, assignments, centroids = get_kmeans_groups(A, k)

    T = nx.minimum_spanning_tree(construct_graph(centroids))

    degrees = dict(T.degree())
    max_degree_node = max(degrees, key=degrees.get)
    bfs_tree = nx.bfs_tree(T, source=max_degree_node)
    bf_across_groups = list(dict(bfs_tree.out_degree()).values())
    group_size = [len(g) for g in groups.values()]
    # print(f"grid{shape}BF = {{{','.join(map(lambda x:str(x),list(dict(bfs_tree_out_degree).values())))}}}")
    # print(f"grid{shape}KS = {{{','.join(list(map(lambda x: str(len(x)), groups.values())))}}}")
    bfs_order = list(bfs_tree)

    # distances = nx.shortest_path_length(bfs_tree, max_degree_node)
    # swarm_tree_height = max(distances.values())
    # print(f"{shape}\tG={G}\t{swarm_tree_height}")
    # return

    bfs_order_gid = {bfs_order[i]: i for i in range(len(bfs_order))}

    localizer = {}
    gid_to_localizer = {}
    dist_across_groups = []
    radio_range_v3 = {i: 0 for i in range(A.shape[0])}
    for i, j in T.edges:
        l_gid = i
        r_gid = j
        l_b_gid = bfs_order_gid[l_gid]
        r_b_gid = bfs_order_gid[r_gid]
        l_group = groups[l_gid]
        r_group = groups[r_gid]
        xdist = cdist(A[l_group], A[r_group])
        am = np.argmin(xdist)
        min_dist = xdist[am // len(r_group), am % len(r_group)]
        dist_across_groups.append(min_dist)
        l_idx = l_group[am // len(r_group)]
        r_idx = r_group[am % len(r_group)]
        if l_b_gid > r_b_gid:
            gid_to_localizer[l_b_gid] = (l_idx, r_idx)
            radio_range_v3[r_idx] = min_dist
        else:
            gid_to_localizer[r_b_gid] = (r_idx, l_idx)
            radio_range_v3[l_idx] = min_dist

    bfs_order_pid = {}
    intra_localizer = {}
    bf_in_groups = []
    dist_in_groups = []
    for gid, pids in groups.items():
        g_points = A[pids]
        g_T = nx.minimum_spanning_tree(construct_graph(g_points))
        b_gid = bfs_order_gid[gid]
        if b_gid in gid_to_localizer:
            source_node = pids.index(gid_to_localizer[b_gid][0])
        else:
            source_node = 0
        bfs_tree = nx.bfs_tree(g_T, source=source_node)
        bfs_order = list(bfs_tree)
        bf_in_groups += dict(bfs_tree.out_degree()).values()
        for i in range(len(bfs_order)):
            bfs_order_pid[pids[bfs_order[i]]] = pids[i]
        # print(gid, pids, bfs_order)
        radio_range_v3[pids[source_node]] = radio_range[gid][source_node]

        for i, j in g_T.edges:
            dist_ij = np.linalg.norm(g_points[i] - g_points[j])
            dist_in_groups.append(dist_ij)
            l_pid = bfs_order_pid[pids[i]]
            r_pid = bfs_order_pid[pids[j]]
            if l_pid > r_pid:
                intra_localizer[l_pid] = r_pid
                radio_range_v3[pids[j]] = max(dist_ij, radio_range_v3.get(pids[j], 0))
            else:
                intra_localizer[r_pid] = l_pid
                radio_range_v3[pids[i]] = max(dist_ij, radio_range_v3.get(pids[i], 0))

    # print(bfs_order_pid)
    for gid, link in gid_to_localizer.items():
        pid_0 = bfs_order_pid[link[0]]
        pid_1 = bfs_order_pid[link[1]]
        if pid_0 in localizer:
            localizer[pid_0].append((pid_1, gid))
        else:
            localizer[pid_0] = [(pid_1, gid)]
        if pid_1 in localizer:
            localizer[pid_1].append((pid_0, None))
        else:
            localizer[pid_1] = [(pid_0, None)]

    new_gid = [bfs_order_gid[a] for a in assignments]
    new_pid = [bfs_order_pid[i] for i in range(A.shape[0])]
    np.savetxt(f"../assets/{shape}_{G}_spanning_2.txt",
               np.hstack((A, np.array(new_gid).reshape(-1, 1), np.array(new_pid).reshape(-1, 1))), delimiter=',')

    with open(f"../assets/{shape}_{G}_spanning_2_localizer.json", "w") as f:
        json.dump(localizer, f)
    with open(f"../assets/{shape}_{G}_spanning_2_intra_localizer.json", "w") as f:
        json.dump(intra_localizer, f)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], depthshade=False)
        for g in groups.values():
            xs = [A[p][0] for p in g]
            ys = [A[p][1] for p in g]
            zs = [A[p][2] for p in g]
            ax.plot3D(xs, ys, zs, '-bo')

        for i, l in localizer.items():
            for p in l:
                ax.plot3D(A[[i, p[0]], 0], A[[i, p[0]], 1], A[[i, p[0]], 2], '-ro')

        # ax.plot3D(A[T, 0], A[T, 1], A[T, 2] + 1, '-o')
        plt.show()

    return {"group_size": group_size,
            "dist_in_groups": dist_in_groups,
            "dist_across_groups": dist_across_groups,
            "bf_in_groups": bf_in_groups,
            "bf_across_groups": bf_across_groups,
            # "radio_range": radio_range.to_list(),
            # "radio_range_v3": radio_range_v3
            }




def create_spanning_tree_groups_2_with_standbys(A, G, shape, visualize, th=2.4):
    standbys = []
    k = math.ceil(1 * A.shape[0] / G)
    groups, radio_range, assignments, centroids = get_kmeans_groups(A, k)

    T = nx.minimum_spanning_tree(construct_graph(centroids))

    degrees = dict(T.degree())
    max_degree_node = max(degrees, key=degrees.get)
    bfs_tree = nx.bfs_tree(T, source=max_degree_node)
    bf_across_groups = list(dict(bfs_tree.out_degree()).values())
    group_size = [len(g) for g in groups.values()]
    # print(f"grid{shape}BF = {{{','.join(map(lambda x:str(x),list(dict(bfs_tree_out_degree).values())))}}}")
    # print(f"grid{shape}KS = {{{','.join(list(map(lambda x: str(len(x)), groups.values())))}}}")
    bfs_order = list(bfs_tree)

    # distances = nx.shortest_path_length(bfs_tree, max_degree_node)
    # swarm_tree_height = max(distances.values())
    # print(f"{shape}\tG={G}\t{swarm_tree_height}")
    # return

    bfs_order_gid = {bfs_order[i]: i for i in range(len(bfs_order))}

    localizer = {}
    gid_to_localizer = {}
    dist_across_groups = []
    radio_range_v3 = {i: 0 for i in range(A.shape[0])}
    for i, j in T.edges:
        l_gid = i
        r_gid = j
        l_b_gid = bfs_order_gid[l_gid]
        r_b_gid = bfs_order_gid[r_gid]
        l_group = groups[l_gid]  # index of points in the group
        r_group = groups[r_gid]
        xdist = cdist(A[l_group], A[r_group])
        am = np.argmin(xdist)
        min_dist = xdist[am // len(r_group), am % len(r_group)]

        dist_across_groups.append(min_dist)
        l_idx = l_group[am // len(r_group)]
        r_idx = r_group[am % len(r_group)]

        if l_b_gid > r_b_gid:
            gid_to_localizer[l_b_gid] = (l_idx, r_idx)
            radio_range_v3[r_idx] = min_dist
        else:
            gid_to_localizer[r_b_gid] = (r_idx, l_idx)
            radio_range_v3[l_idx] = min_dist

    bfs_order_pid = {}
    intra_localizer = {}
    bf_in_groups = []
    dist_in_groups = []
    for gid, pids in groups.items():
        g_points = A[pids]
        g_T = nx.minimum_spanning_tree(construct_graph(g_points))
        b_gid = bfs_order_gid[gid]
        if b_gid in gid_to_localizer:
            source_node = pids.index(gid_to_localizer[b_gid][0])
        else:
            source_node = 0
        bfs_tree = nx.bfs_tree(g_T, source=source_node)
        bfs_order = list(bfs_tree)
        bf_in_groups += dict(bfs_tree.out_degree()).values()
        for i in range(len(bfs_order)):
            bfs_order_pid[pids[bfs_order[i]]] = pids[i]
        # print(gid, pids, bfs_order)
        radio_range_v3[pids[source_node]] = radio_range[gid][source_node]

        for i, j in g_T.edges:
            dist_ij = np.linalg.norm(g_points[i] - g_points[j])
            dist_in_groups.append(dist_ij)
            l_pid = bfs_order_pid[pids[i]]
            r_pid = bfs_order_pid[pids[j]]
            if l_pid > r_pid:
                intra_localizer[l_pid] = r_pid
                radio_range_v3[pids[j]] = max(dist_ij, radio_range_v3.get(pids[j], 0))
            else:
                intra_localizer[r_pid] = l_pid
                radio_range_v3[pids[i]] = max(dist_ij, radio_range_v3.get(pids[i], 0))

    # print(bfs_order_pid)
    for gid, link in gid_to_localizer.items():
        pid_0 = bfs_order_pid[link[0]]
        pid_1 = bfs_order_pid[link[1]]
        if pid_0 in localizer:
            localizer[pid_0].append((pid_1, gid))
        else:
            localizer[pid_0] = [(pid_1, gid)]
        if pid_1 in localizer:
            localizer[pid_1].append((pid_0, None))
        else:
            localizer[pid_1] = [(pid_0, None)]

    new_gid = [bfs_order_gid[a] for a in assignments]
    new_pid = [bfs_order_pid[i] for i in range(A.shape[0])]
    np.savetxt(f"../assets/{shape}_{G}_spanning_2.txt",
               np.hstack((A, np.array(new_gid).reshape(-1, 1), np.array(new_pid).reshape(-1, 1))), delimiter=',')

    with open(f"../assets/{shape}_{G}_spanning_2_localizer.json", "w") as f:
        json.dump(localizer, f)
    with open(f"../assets/{shape}_{G}_spanning_2_intra_localizer.json", "w") as f:
        json.dump(intra_localizer, f)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], depthshade=False)
        for g in groups.values():
            xs = [A[p][0] for p in g]
            ys = [A[p][1] for p in g]
            zs = [A[p][2] for p in g]
            ax.plot3D(xs, ys, zs, '-bo')

        for i, l in localizer.items():
            for p in l:
                ax.plot3D(A[[i, p[0]], 0], A[[i, p[0]], 1], A[[i, p[0]], 2], '-ro')

        # ax.plot3D(A[T, 0], A[T, 1], A[T, 2] + 1, '-o')
        plt.show()

    return {"group_size": group_size,
            "dist_in_groups": dist_in_groups,
            "dist_across_groups": dist_across_groups,
            "bf_in_groups": bf_in_groups,
            "bf_across_groups": bf_across_groups,
            # "radio_range": radio_range.to_list(),
            # "radio_range_v3": radio_range_v3
            }


def create_histograms(shape, G, group_size, dist_in_groups, dist_across_groups, bf_in_groups, bf_across_groups, **kwargs):
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[0, 2]
    ax4 = axes[1, 0]
    ax5 = axes[1, 1]

    c = Counter(group_size)
    ax1.bar(c.keys(), c.values())
    ax2.hist(dist_in_groups)
    ax3.hist(dist_across_groups)
    c = Counter(bf_in_groups)
    ax4.bar(c.keys(), c.values())
    c = Counter(bf_across_groups)
    ax5.bar(c.keys(), c.values())

    ax1.set_title("group sizes")
    ax2.set_title("distance within groups")
    ax3.set_title("distance across groups")
    ax4.set_title("branching factor within groups")
    ax5.set_title("branching factor across groups")

    plt.suptitle(f"{shape}, G={G}")
    plt.savefig(f"../assets/Sk_2/{shape}_{G}_hist.png")


def create_spanning_tree_groups_2_with_source(A, G, shape, visualize=False, source='max'):
    k = int(1.5 * A.shape[0]) // G
    groups, max_dist_1, assignments, centroids = get_kmeans_groups(A, k)
    # print(groups)

    T = nx.minimum_spanning_tree(construct_graph(centroids))

    degrees = dict(T.degree())
    if source == 'max':
        bfs_source = max(degrees, key=degrees.get)
    else:
        bfs_source = assignments[source]
    # print(groups[bfs_source], source)
    bfs_tree = nx.bfs_tree(T, source=bfs_source)
    bfs_tree_out_degree = bfs_tree.out_degree()
    # print(f"grid{shape}BF = {{{','.join(map(lambda x:str(x),list(dict(bfs_tree_out_degree).values())))}}}")
    # print(f"grid{shape}KS = {{{','.join(list(map(lambda x: str(len(x)), groups.values())))}}}")
    bfs_order = list(bfs_tree)

    bfs_order_gid = {bfs_order[i]: i for i in range(len(bfs_order))}

    localizer = {}
    gid_to_localizer = {}
    dists = []
    for i, j in T.edges:
        l_gid = i
        r_gid = j
        l_b_gid = bfs_order_gid[l_gid]
        r_b_gid = bfs_order_gid[r_gid]
        l_group = groups[l_gid]
        r_group = groups[r_gid]
        xdist = cdist(A[l_group], A[r_group])
        am = np.argmin(xdist)
        dists.append(xdist[am // len(r_group), am % len(r_group)])
        l_idx = l_group[am // len(r_group)]
        r_idx = r_group[am % len(r_group)]
        if l_b_gid > r_b_gid:
            gid_to_localizer[l_b_gid] = (l_idx, r_idx)
        else:
            gid_to_localizer[r_b_gid] = (r_idx, l_idx)

    bfs_order_pid = {}
    intra_localizer = {}
    for gid, pids in groups.items():
        g_points = A[pids]
        g_T = nx.minimum_spanning_tree(construct_graph(g_points))
        b_gid = bfs_order_gid[gid]
        if b_gid in gid_to_localizer:
            source_node = pids.index(gid_to_localizer[b_gid][0])
        else:
            if source == 'max':
                source_node = 0
            else:
                source_node = pids.index(source)
        bfs_tree = nx.bfs_tree(g_T, source=source_node)
        bfs_order = list(bfs_tree)
        for i in range(len(bfs_order)):
            bfs_order_pid[pids[bfs_order[i]]] = pids[i]
        # print(gid, pids, bfs_order)
        for i, j in g_T.edges:
            l_pid = bfs_order_pid[pids[i]]
            r_pid = bfs_order_pid[pids[j]]
            if l_pid > r_pid:
                intra_localizer[l_pid] = r_pid
            else:
                intra_localizer[r_pid] = l_pid
    # print(bfs_order_pid)
    for gid, link in gid_to_localizer.items():
        if bfs_order_pid[link[0]] in localizer:
            localizer[bfs_order_pid[link[0]]].append((bfs_order_pid[link[1]], gid))
        else:
            localizer[bfs_order_pid[link[0]]] = [(bfs_order_pid[link[1]], gid)]
        if bfs_order_pid[link[1]] in localizer:
            localizer[bfs_order_pid[link[1]]].append((bfs_order_pid[link[0]], None))
        else:
            localizer[bfs_order_pid[link[1]]] = [(bfs_order_pid[link[0]], None)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax.hist(max_dist_1)
    ax2.hist(dists)
    # plt.savefig(f"../assets/{shape}_spanning_2_hist.png")
    new_gid = [bfs_order_gid[a] for a in assignments]
    new_pid = [bfs_order_pid[i] for i in range(A.shape[0])]
    point_cloud = np.hstack((A, np.array(new_gid).reshape(-1, 1), np.array(new_pid).reshape(-1, 1)))

    # np.savetxt(f"../assets/{shape}_spanning_2.txt", point_cloud, delimiter=',')
    #
    # with open(f"../assets/{shape}_spanning_2_localizer.json", "w") as f:
    #     json.dump(localizer, f)
    # with open(f"../assets/{shape}_spanning_2_intra_localizer.json", "w") as f:
    #     json.dump(intra_localizer, f)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], depthshade=False)
        for g in groups.values():
            xs = [A[p][0] for p in g]
            ys = [A[p][1] for p in g]
            zs = [A[p][2] for p in g]
            ax.plot3D(xs, ys, zs, '-bo')

        for i, l in localizer.items():
            for p in l:
                ax.plot3D(A[[i, p[0]], 0], A[[i, p[0]], 1], A[[i, p[0]], 2], '-ro')

        # ax.plot3D(A[T, 0], A[T, 1], A[T, 2] + 1, '-o')
        plt.show()

    return point_cloud, localizer, intra_localizer, bfs_order_pid


def create_spanning_tree_groups_2_dfs(A, G, shape, visualize):
    k = int(1.5 * A.shape[0]) // G
    groups, max_dist_1, assignments, centroids = get_kmeans_groups(A, k)

    T = nx.minimum_spanning_tree(construct_graph(centroids))

    degrees = dict(T.degree())
    max_degree_node = max(degrees, key=degrees.get)
    dfs_tree = nx.dfs_tree(T, source=max_degree_node)
    dfs_tree_out_degree = dfs_tree.out_degree()
    # print(f"grid{shape}BF = {{{','.join(map(lambda x:str(x),list(dict(bfs_tree_out_degree).values())))}}}")
    # print(f"grid{shape}KS = {{{','.join(list(map(lambda x: str(len(x)), groups.values())))}}}")
    dfs_order = list(dfs_tree)

    dfs_order_gid = {dfs_order[i]: i for i in range(len(dfs_order))}

    localizer = {}
    gid_to_localizer = {}
    dists = []
    for i, j in T.edges:
        l_gid = i
        r_gid = j
        l_b_gid = dfs_order_gid[l_gid]
        r_b_gid = dfs_order_gid[r_gid]
        l_group = groups[l_gid]
        r_group = groups[r_gid]
        xdist = cdist(A[l_group], A[r_group])
        am = np.argmin(xdist)
        dists.append(xdist[am // len(r_group), am % len(r_group)])
        l_idx = l_group[am // len(r_group)]
        r_idx = r_group[am % len(r_group)]
        if l_b_gid > r_b_gid:
            gid_to_localizer[l_b_gid] = (l_idx, r_idx)
        else:
            gid_to_localizer[r_b_gid] = (r_idx, l_idx)

    bfs_order_pid = {}
    intra_localizer = {}
    for gid, pids in groups.items():
        g_points = A[pids]
        g_T = nx.minimum_spanning_tree(construct_graph(g_points))
        b_gid = dfs_order_gid[gid]
        if b_gid in gid_to_localizer:
            source_node = pids.index(gid_to_localizer[b_gid][0])
        else:
            source_node = 0
        dfs_tree = nx.bfs_tree(g_T, source=source_node)
        dfs_order = list(dfs_tree)
        for i in range(len(dfs_order)):
            bfs_order_pid[pids[dfs_order[i]]] = pids[i]
        # print(gid, pids, bfs_order)
        for i, j in g_T.edges:
            l_pid = bfs_order_pid[pids[i]]
            r_pid = bfs_order_pid[pids[j]]
            if l_pid > r_pid:
                intra_localizer[l_pid] = r_pid
            else:
                intra_localizer[r_pid] = l_pid
    # print(bfs_order_pid)
    for gid, link in gid_to_localizer.items():
        if bfs_order_pid[link[0]] in localizer:
            localizer[bfs_order_pid[link[0]]].append((bfs_order_pid[link[1]], gid))
        else:
            localizer[bfs_order_pid[link[0]]] = [(bfs_order_pid[link[1]], gid)]
        if bfs_order_pid[link[1]] in localizer:
            localizer[bfs_order_pid[link[1]]].append((bfs_order_pid[link[0]], None))
        else:
            localizer[bfs_order_pid[link[1]]] = [(bfs_order_pid[link[0]], None)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax.hist(max_dist_1)
    ax2.hist(dists)
    # plt.savefig(f"../assets/{shape}_spanning_2dfs_hist.png")
    new_gid = [dfs_order_gid[a] for a in assignments]
    new_pid = [bfs_order_pid[i] for i in range(A.shape[0])]
    np.savetxt(f"../assets/{shape}_{G}_spanning_2dfs.txt",
               np.hstack((A, np.array(new_gid).reshape(-1, 1), np.array(new_pid).reshape(-1, 1))), delimiter=',')

    with open(f"../assets/{shape}_{G}_spanning_2dfs_localizer.json", "w") as f:
        json.dump(localizer, f)
    with open(f"../assets/{shape}_{G}_spanning_2dfs_intra_localizer.json", "w") as f:
        json.dump(intra_localizer, f)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], depthshade=False)
        for g in groups.values():
            xs = [A[p][0] for p in g]
            ys = [A[p][1] for p in g]
            zs = [A[p][2] for p in g]
            ax.plot3D(xs, ys, zs, '-bo')

        for i, l in localizer.items():
            for p in l:
                ax.plot3D(A[[i, p[0]], 0], A[[i, p[0]], 1], A[[i, p[0]], 2], '-ro')

        # ax.plot3D(A[T, 0], A[T, 1], A[T, 2] + 1, '-o')
        plt.show()


def create_clustered_spanning_groups(A, G, shape):
    n = A.shape[0]
    threshold = 100
    # if n > threshold:
    num_clusters = int(np.ceil(n / threshold))
    # num_clusters = 1
    groups, _, assignments, centroids = get_kmeans_groups(A, num_clusters)
    # print(groups)

    T = nx.minimum_spanning_tree(construct_graph(centroids))
    degrees = dict(T.degree())
    bfs_source = max(degrees, key=degrees.get)
    bfs_tree = nx.bfs_tree(T, source=bfs_source)
    bfs_order = list(bfs_tree)
    bfs_order_gid = {bfs_order[i]: i for i in range(len(bfs_order))}
    gid_to_localizer = {}
    for i, j in T.edges:
        l_gid = i
        r_gid = j
        l_b_gid = bfs_order_gid[l_gid]
        r_b_gid = bfs_order_gid[r_gid]
        l_group = groups[l_gid]
        r_group = groups[r_gid]
        xdist = cdist(A[l_group], A[r_group])
        am = np.argmin(xdist)
        l_idx = l_group[am // len(r_group)]
        r_idx = r_group[am % len(r_group)]
        if l_b_gid > r_b_gid:
            gid_to_localizer[l_b_gid] = (l_idx, r_idx, r_b_gid)
        else:
            gid_to_localizer[r_b_gid] = (r_idx, l_idx, l_b_gid)

    g_point_cloud = []
    g_localizer = {}
    g_intra_localizer = {}
    g_bfs_order_pid = {}
    for gid, group in groups.items():
        b_gid = bfs_order_gid[gid]
        offset = (b_gid + 1) * 1000
        if b_gid in gid_to_localizer:
            localizer_idx = len(list(filter(lambda x: x == gid, assignments[:gid_to_localizer[b_gid][0]])))
            point_cloud, localizer, intra_localizer, bfs_order_pid = create_spanning_tree_groups_2_with_source(A[group],
                                                                                                               G,
                                                                                                               shape,
                                                                                                               source=localizer_idx, )
            # print(gid, assignments[gid_to_localizer[b_gid][0]])
            # print(A[gid_to_localizer[b_gid][0]], A[group][localizer_idx])
            g_bfs_order_pid[b_gid] = bfs_order_pid
        else:
            point_cloud, localizer, intra_localizer, bfs_order_pid = create_spanning_tree_groups_2_with_source(A[group],
                                                                                                               G, shape)
            g_bfs_order_pid[b_gid] = bfs_order_pid

        point_cloud[:, 3] += offset
        point_cloud[:, 4] += offset
        point_cloud = np.hstack((point_cloud, np.full((len(point_cloud), 1), b_gid)))
        localizer = {k + offset: [(i + offset, j + offset if j is not None else j) for i, j in v] for k, v in
                     localizer.items()}
        intra_localizer = {k + offset: v + offset for k, v in intra_localizer.items()}
        g_point_cloud.append(point_cloud)
        g_localizer |= localizer
        g_intra_localizer |= intra_localizer

    g_point_cloud = np.vstack(g_point_cloud)
    # print(gid_to_localizer)
    # print(groups)
    for gid, link in gid_to_localizer.items():
        link_0 = len(list(filter(lambda x: bfs_order_gid[x] == gid, assignments[:link[0]])))
        link_1 = len(list(filter(lambda x: bfs_order_gid[x] == link[2], assignments[:link[1]])))
        # print(gid, link_0, link_1)

        pid_0 = g_bfs_order_pid[gid][link_0] + 1000 * (gid + 1)
        pid_1 = g_bfs_order_pid[link[2]][link_1] + 1000 * (link[2] + 1)
        if pid_0 in g_localizer:
            g_localizer[pid_0].append((pid_1, gid))
        else:
            g_localizer[pid_0] = [(pid_1, gid)]
        if pid_1 in g_localizer:
            g_localizer[pid_1].append((pid_0, None))
        else:
            g_localizer[pid_1] = [(pid_0, None)]

    np.savetxt(f"../assets/{shape}_spanning_3.txt", g_point_cloud, delimiter=',')

    with open(f"../assets/{shape}_spanning_3_localizer.json", "w") as f:
        json.dump(g_localizer, f)
    with open(f"../assets/{shape}_spanning_3_intra_localizer.json", "w") as f:
        json.dump(g_intra_localizer, f)


def create_mst_groups(A, shape):
    T = nx.minimum_spanning_tree(construct_graph(A))
    degrees = dict(T.degree())
    max_degree_node = max(degrees, key=degrees.get)
    dfs_tree = nx.dfs_tree(T, source=max_degree_node)
    # dfs_tree_out_degree = dfs_tree.out_degree()
    # print(f"grid{shape}BF = {{{','.join(map(lambda x:str(x),list(dict(bfs_tree_out_degree).values())))}}}")
    # print(f"grid{shape}KS = {{{','.join(list(map(lambda x: str(len(x)), groups.values())))}}}")
    dfs_order = list(dfs_tree)
    dfs_order_pid = {dfs_order[i]: i for i in range(len(dfs_order))}
    pid_to_anchor = {}
    pid_to_localizer = {}
    for i, j in T.edges:
        pid_i = dfs_order_pid[i]
        pid_j = dfs_order_pid[j]
        if pid_i > pid_j:
            localizer = pid_i
            anchor = pid_j
        else:
            localizer = pid_j
            anchor = pid_i
        pid_to_anchor[localizer] = anchor
        if anchor in pid_to_localizer:
            pid_to_localizer[anchor].append(localizer)
        else:
            pid_to_localizer[anchor] = [localizer]

    new_pid = [dfs_order_pid[i] for i in range(A.shape[0])]
    np.savetxt(f"../assets/{shape}_mst.txt",
               np.hstack((A, np.array(new_pid).reshape(-1, 1))), delimiter=',')

    with open(f"../assets/{shape}_mst_localizer.json", "w") as f:
        json.dump({"localizer": pid_to_anchor, "anchor": pid_to_localizer}, f)


if __name__ == "__main__":
    # n = 4
    visualize = False
    # shapes = ["chess_100", "chess_408", "skateboard_1372", "dragon_1147", "palm_725", "racecar_3720", "kangaroo_972"][
    #          2:3]
    # scales = [.2, .4, 1, 1, 1, 1, 1][2:3]
    shapes = ["08point"]
    scales = [0.01]
    # for n in [6]:
    # for n in [6]:
    # for shape in ["chess"]:
    # for shape in ["racecar_3826"]:
    for g in [4]:
    # for g in [5, 10, 50, 150, 200]:
        for shape, scale in zip(shapes, scales):
        # for n in [36]:
            # for shape in ["chess_100"]:
            # shape = f"grid_{n*n}"
            # shape = f"line_{n}"

            if visualize:
                mpl.use('macosx')

            # A = np.random.rand(n, 3)
            # for i in range(1):
            #     for j in range(n):
            #         A[i * n + j] = [i, j, 1]

            A = np.loadtxt(f'../assets/{shape}.xyz', delimiter=' ') * 100 * scale
            # A = np.loadtxt(f'../assets/{shape}.txt', delimiter=',')*0.4
            # A[:, [1, 2, 0]] = A[:, [0, 1, 2]]

            # create_overlapping_groups(A, 5, shape, visualize)
            # create_hierarchical_groups(A, 5, shape, visualize)
            # create_binary_overlapping_groups(A, shape, visualize)
            # create_spanning_tree_groups(A, 5, shape, visualize)

            hists = create_spanning_tree_groups_2(A, g, shape, visualize)
            # with open(f"../assets/spanning_stats_{shape}_{g}.json", "w") as f:
            #     json.dump(hists, f)


            # merged_list = list(hists["radio_range_v3"].values())
            # # merged_list = list(itertools.chain(*radio_range))
            # min_rr = np.min(merged_list)
            # mean_rr = np.mean(merged_list)
            # max_rr = np.max(merged_list)
            #
            # print(f"{shape}\tG={g}\t{min_rr}\t{mean_rr}\t{max_rr}")

            create_histograms(shape, g, **hists)

            # create_spanning_tree_groups_2_dfs(A, 5, shape, visualize)
            # create_clustered_spanning_groups(A, 5, shape)
            # create_mst_groups(A, shape)
