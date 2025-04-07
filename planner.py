import json
import math
import os
from copy import deepcopy
from datetime import datetime
from collections import Counter

import pymeshlab
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

from utils.grouping import get_kmeans_groups, construct_graph, is_flat
from utils.tree import find_equidistant_point
from utils.plot import add_dead_reckoning_error
from utils import create_logger

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


class Planner:
    def __init__(self):
        self.mesh_file_path = None
        self.mesh = None
        self.point_cloud = None
        self.shape_name = ""
        self.grouped_points = None
        self.localizer = None
        self.intra_localizer = None
        self.gid = None
        self.pid = None
        self.swarm_size = None
        self.ms = None
        self.fls_radius = None
        self.fls_min_camera_range = None
        self.fls_max_camera_range = None
        self.mesh_surface_area = None
        self.num_samples = None
        self.logger = None
        self.init_logger()

    def init_logger(self):
        self.logger = create_logger("Planner")

    def set_fls_specs(self, fls_radius, fls_min_camera_range, fls_max_camera_range):
        self.fls_radius = fls_radius
        self.fls_min_camera_range = fls_min_camera_range
        self.fls_max_camera_range = fls_max_camera_range

    def load_mesh(self, mesh_file_path, scale=1.0):
        self.ms = pymeshlab.MeshSet()
        self.ms.load_new_mesh(mesh_file_path)
        self.shape_name = os.path.splitext(os.path.basename(mesh_file_path))[0]
        self.ms.compute_matrix_from_scaling_or_normalization(axisx=scale, axisy=scale, axisz=scale)
        out_dict = self.ms.get_geometric_measures()
        self.mesh_surface_area = out_dict['surface_area']
        self.logger.info(f"Area after scaling: {self.mesh_surface_area} (m^2)")

    def compute_number_of_flss(self):
        min_density = 1 / (math.pi * max(self.fls_max_camera_range / 2, self.fls_radius) ** 2)
        max_density = 1 / (math.pi * max(self.fls_min_camera_range / 2, self.fls_radius) ** 2)
        avg_density = (min_density + max_density) / 2
        self.num_samples = int(avg_density * self.mesh_surface_area)
        self.logger.info(f"Approximately {self.num_samples} FLSs are required")

    def sample_point_from_mesh(self):
        self.ms.generate_sampling_poisson_disk(samplenum=self.num_samples, exactnumflag=True)
        num_sampled_points = self.ms.current_mesh().vertex_number()
        vertices = np.array(self.ms.current_mesh().vertex_matrix())
        vertices[:, [1, 2, 0]] = vertices[:, [0, 1, 2]]
        self.point_cloud = vertices * 100
        current_date_time = datetime.now().strftime("%H-%M-%S_%m-%d-%Y")
        output_dir = os.path.join("assets", "dataset", current_date_time)
        output_file = os.path.join(output_dir, f"{self.shape_name}_{num_sampled_points}.xyz")
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(output_file, self.point_cloud, delimiter=" ")
        self.logger.info(f"Saved point cloud with {num_sampled_points} points to {output_file}")

    def load_point_cloud(self, point_cloud_file_path):
        self.point_cloud = np.loadtxt(point_cloud_file_path, delimiter=' ')
        self.shape_name = os.path.splitext(os.path.basename(point_cloud_file_path))[0]

    def generate_grid_point_cloud(self, grid_size):
        n = grid_size ** 2
        self.point_cloud = np.random.rand(n, 3)
        self.shape_name = f"grid{grid_size}x{grid_size}_{n}"
        for i in range(grid_size):
            for j in range(grid_size):
                self.point_cloud[i * grid_size + j] = [i * self.fls_min_camera_range * 100,
                                                       j * self.fls_min_camera_range * 100, 1]

    def compute_trees(self, swarm_size):
        A = self.point_cloud
        shape = self.shape_name
        G = swarm_size
        self.swarm_size = swarm_size
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

        self.gid = [bfs_order_gid[a] for a in assignments]
        self.pid = [bfs_order_pid[i] for i in range(A.shape[0])]
        # np.savetxt(f"../assets/{shape}_{G}_spanning_2.txt",
        #            np.hstack((A, np.array(new_gid).reshape(-1, 1), np.array(new_pid).reshape(-1, 1))), delimiter=',')

        self.localizer = {str(k): v for k, v in localizer.items()}
        self.intra_localizer = {str(k): v for k, v in intra_localizer.items()}
        self.grouped_points = np.hstack(
            (self.point_cloud, np.array(self.gid).reshape(-1, 1), np.array(self.pid).reshape(-1, 1)))
        # with open(f"../assets/{shape}_{G}_spanning_2_localizer.json", "w") as f:
        #     json.dump(localizer, f)
        # with open(f"../assets/{shape}_{G}_spanning_2_intra_localizer.json", "w") as f:
        #     json.dump(intra_localizer, f)

        # if True:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], depthshade=False)
        #     for g in groups.values():
        #         xs = [A[p][0] for p in g]
        #         ys = [A[p][1] for p in g]
        #         zs = [A[p][2] for p in g]
        #         ax.plot3D(xs, ys, zs, '-bo')
        #
        #     for i, l in localizer.items():
        #         for p in l:
        #             ax.plot3D(A[[i, p[0]], 0], A[[i, p[0]], 1], A[[i, p[0]], 2], '-ro')

        # ax.plot3D(A[T, 0], A[T, 1], A[T, 2] + 1, '-o')
        # plt.show()

        return {"group_size": group_size,
                "dist_in_groups": dist_in_groups,
                "dist_across_groups": dist_across_groups,
                "bf_in_groups": bf_in_groups,
                "bf_across_groups": bf_across_groups,
                # "radio_range": radio_range.to_list(),
                # "radio_range_v3": radio_range_v3
                }

    def load_trees(self, shape_name=None, swarm_size=None, px=""):
        if shape_name is None:
            shape_name = self.shape_name
        if swarm_size is None:
            swarm_size = self.swarm_size
        path = os.path.join("assets", f"{shape_name}_{swarm_size}_spanning_2{px}")
        localizer_path = path + '_localizer.json'
        intra_localizer_path = path + '_intra_localizer.json'
        with open(intra_localizer_path) as f:
            self.intra_localizer = json.load(f)
        with open(localizer_path) as f:
            self.localizer = json.load(f)
        return path

    def add_standbys(self, min_dist=6.12, th=7.82, shape_name=None, swarm_size=None):  # min_dist=1.8, th=2.3, s=3.4
        path = self.load_trees(shape_name, swarm_size)
        localizer = self.localizer
        intra_localizer = self.intra_localizer
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
                new_dist = (min_dist + th + 0.1) / 2
                if dist_ij / 2 > new_dist:
                    new_dist = dist_ij / 2
                # if new_dist > th + 0.1:
                #     print("Exceeded fls tree", new_dist)
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

            if str(j) not in intra_localizer and (
                    (str(j) not in localizer) or (str(j) in localizer and all(not l[1] for l in localizer[str(j)]))):
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
                        new_dist = (min_dist + th + 0.1) / 2
                        if dist_ij / 2 > new_dist:
                            new_dist = dist_ij / 2
                        # if new_dist > th + 0.1:
                        #     print("Exceeded swarm tree", new_dist)
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

        # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # ax1 = axes[0, 0]
        # ax2 = axes[0, 1]
        # ax1.hist(np.array(dists))
        # ax2.hist(np.array(dists_2))
        # plt.show()
        # return

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

        self.logger.info(f"Added {len(new_points)} standby FLS")
        self.logger.info(f"Saved stats and tree files as {path}_sb*")

    def save_trees(self):
        np.savetxt(
            os.path.join("assets", f"{self.shape_name}_{self.swarm_size}_spanning_2.txt"),
            self.grouped_points,
            delimiter=','
        )

        with open(os.path.join("assets", f"{self.shape_name}_{self.swarm_size}_spanning_2_localizer.json"), "w") as f:
            json.dump(self.localizer, f)
        with open(os.path.join("assets", f"{self.shape_name}_{self.swarm_size}_spanning_2_intra_localizer.json"),
                  "w") as f:
            json.dump(self.intra_localizer, f)

        self.logger.info(f"Data structures saved as {self.shape_name}_{self.swarm_size}_spanning_2 in the assets directory")

    def visualize_point_cloud(self, point_cloud=None, color="#0871f5"):
        if point_cloud is None:
            point_cloud = self.point_cloud

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            color=color,
            depthshade=False,
            s=1
        )
        ax.set_aspect('equal')
        plt.show()

    def visualize_deadreckoning(self, alpha=5):
        A_e = []
        for v in self.point_cloud:
            A_e.append(add_dead_reckoning_error(v, alpha))

        A_e = np.vstack(A_e)
        self.visualize_point_cloud(point_cloud=A_e, color='red')

    def visualize_trees(self, shape_name=None, swarm_size=None, px="_sb"):
        path = self.load_trees(shape_name, swarm_size, px=px)
        localizer = self.localizer
        intra_localizer = self.intra_localizer
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
            if str(j) not in intra_localizer and (
                    (str(j) not in localizer) or (str(j) in localizer and all(not l[1] for l in localizer[str(j)]))):
                sid_to_pid[0] = j

        swarm_tree = nx.DiGraph()
        for i, js in localizer.items():
            for j in js:
                if j[1] is not None:
                    i = int(i)
                    sid_to_pid[j[1]] = i
                    if fid_to_sid[j[0]] == 0:
                        fls_tree.add_edge(i, j[0])
                    swarm_tree.add_edge(fid_to_sid[i], fid_to_sid[j[0]])

        # print(sid_to_pid)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # Draw nodes
        # nx.draw_networkx_nodes(fls_tree, fid_to_coord, node_color='blue', ax=ax)

        # Draw edges
        # nx.draw_networkx_edges(fls_tree, fid_to_coord, ax=ax)
        # print(len(sid_to_coord[7]))
        # nx.draw_kamada_kawai(swarm_tree, with_labels=True, arrows=True)
        # plt.show()
        # exit()

        # with plt.style.context('seaborn-white'):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax.set_title("FLS-tree of the root swarm")
        ax2.set_title("Swarm-tree")
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

            if fid_to_sid[i] == 0 or fid_to_sid[j] == 0:
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
        ax.scatter3D([root[0]], [root[1]], [root[2]], marker=root_marker, color=root_color, depthshade=False,
                     zorder=2, s=25)
        # ax.text(root[0], root[1], root[2], "primary")

        ax.set_aspect('equal')

        ax.view_init(azim=-110, elev=20)
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.axis('off')

        legend_elements = [
            Line2D([0], [0], marker=fls_marker, color='w', label='FLS',
                   markerfacecolor=fls_color, markersize=10),
            Line2D([0], [0], marker=root_marker, color='w', label='Root FLS',
                   markerfacecolor=root_color, markersize=10),
            Line2D([0], [0], marker=anchor_marker, color='w', label='Anchor FLS',
                   markerfacecolor=anchor_color, markersize=10)
        ]

        ax.legend(handles=legend_elements, loc='upper left')

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
            ax2.scatter3D(sid_to_centroid[sid][0], sid_to_centroid[sid][1], sid_to_centroid[sid][2],
                          color=centroid_color, s=40,
                          depthshade=False)
            if is_flat(s_points):
                hull = ConvexHull(s_points[:, 0:2])
                for simplex in hull.simplices:
                    ax2.plot(s_points[simplex, 0], s_points[simplex, 1], s_points[simplex, 2], color=swarm_color)

            else:
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

    def find_cube_side(self, vector):
        max_dot_product = -1  # Start with a low value
        closest_side = None

        for side, normal_vector in faces.items():
            dot_product = np.dot(vector, normal_vector)
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                closest_side = side

        return closest_side

    def compute_fls_mix(self, shape_name=None, swarm_size=None):
        path = self.load_trees(shape_name, swarm_size, px="_sb")
        localizer = self.localizer
        intra_localizer = self.intra_localizer
        A = np.loadtxt(f'{path}.txt', delimiter=',')
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
            side = self.find_cube_side(h)
            camera_placement[pid] = side
            if side == '-X':
                simplified_camera_placement[pid] = ('+X', 180)
            elif side == '+Y':
                simplified_camera_placement[pid] = ('+X', 90)
            elif side == '-Y':
                simplified_camera_placement[pid] = ('+X', 270)
            else:
                simplified_camera_placement[pid] = (side, 0)

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
        percent = {x: 100 * (y / A.shape[0]) for x, y in hist.items()}
        plt.bar(range(len(hist)), hist.values(), color=[color_map[h] for h in hist.keys()])
        plt.xticks(range(len(hist)), hist.keys())
        # plt.show()
        self.logger.info(f"Total number of FLSs: {A.shape[0]}")
        self.logger.info(
            f"Number (percentage) of each variant:\n{', '.join([f'{k}:{v} ({percent[k]:.2f}%)' for k, v in hist.items()])}")


if __name__ == '__main__':
    planner = Planner()
    shapes = {
        "chess_small": {"point_cloud": "chess_100.xyz", "mesh": "m1609.off", "scale": 0.68},
        "chess": {"point_cloud": "chess_408.xyz", "mesh": "m1609.off", "scale": 1.36},
        "dragon": {"point_cloud": "dragon_1147.xyz", "mesh": "m1625.off", "scale": 3.4},
        "kangaroo": {"point_cloud": "kangaroo_972.xyz", "mesh": "12271_Kangaroo_v1_L3.obj", "scale": 3.4},
        "palm": {"point_cloud": "palm_725.xyz", "mesh": "m1096.off", "scale": 3.4},
        "racecar": {"point_cloud": "racecar_3720.xyz", "mesh": "m1510.off", "scale": 3.4},
        "skateboard": {"point_cloud": "skateboard_1372.xyz", "mesh": "m1619.off", "scale": 3.4},
    }

    # planner.generate_grid_point_cloud(16)
    # planner.load_point_cloud(os.path.join("assets", "dataset", "point_cloud", shapes["skateboard"]["point_cloud"]))
    # planner.visualize_point_cloud()
    # planner.compute_trees(swarm_size=50)
    # planner.save_trees()
    planner.visualize_trees(shape_name="chess_408", swarm_size=50)
    # planner.visualize_trees(shape_name="grid16x16_256", swarm_size=50)
    # planner.visualize_trees()
    # for shape in shapes.values():
    # planner.load_mesh(os.path.join("assets", "dataset", "mesh", shapes["skateboard"]["mesh"]), scale=3.4)
    # planner.sample_point_from_mesh()
    # planner.load_point_cloud(os.path.join("assets", "dataset", "point_cloud", shape["point_cloud"]))
    # planner.load_point_cloud(os.path.join("assets", "dataset", "09-59-02_04-03-2025", "m1619_1369.xyz"))
    # planner.visualize_point_cloud()
    # planner.generate_trees(swarm_size=50)
    # planner.save_trees()
    # print(shape)
    # planner.add_standbys()
