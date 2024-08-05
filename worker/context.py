import time
from functools import cache

import networkx as nx
import numpy as np
from multiprocessing import shared_memory

import velocity
from config import Config
from utils.obstruction import intersects_sphere


class WorkerContext:
    def __init__(self, count, fid, sid, gtl, el, shm_name, metrics, stand_by_coord, localizer, g_order, intra_localizer, anchor, swarm_gtl, radius):
        self.count = count
        self.fid = fid
        self.gtl = gtl
        self.el = el
        self.yaw = 0
        self.swarm_id = sid
        self.neighbors = dict()
        self.neighbors_gtl = dict()
        self.sweet_neighbors = dict()
        self.radio_range = Config.INITIAL_RANGE
        self.size = 1
        self.query_id = None
        self.challenge_id = None
        self.shm_name = shm_name
        self.message_id = 0
        self.alpha = Config.DEAD_RECKONING_ANGLE / 180 * np.pi
        self.lease = dict()
        self.metrics = metrics
        self.last_expanded = 0
        self.stand_by_coord = stand_by_coord
        self.localizer = localizer
        self.intra_localizer = intra_localizer
        self.hierarchy = None
        self.hid = 0
        self.min_gid = np.max(sid)
        self.set_swarm_id(sid)
        self.g_order = g_order
        self.anchor_for = anchor
        self.tree = nx.DiGraph()
        self.paths = {}
        self.absolute_poses = {}
        self.swarm_gtl = swarm_gtl
        self.rd = 7
        self.radius = radius
        if Config.CAMERA == 'w':
            # wide camera
            x = np.array([200, 150, 100, 75, 50, 45, 42.5, 40]) / 10  # cm
            y = np.array([30.46, 11.68, 5.82, 0.893333, 2.14, 5.4, 4.16471, 6.05]) / 100  # percent
        else:
            # Reg camera
            x = np.array([300, 200, 150, 100, 75, 70]) / 10  # cm
            y = np.array([29.92, 7.74, 3.25333, 2.58, 1.94667, 1.35714]) / 100  # percent

        self.error_coefficients = np.polyfit(x, y, 2) - np.array([0, 0, .015])

    def set_swarm_id(self, swarm_id):
        self.swarm_id = swarm_id
        if self.shm_name:
            shared_mem = shared_memory.SharedMemory(name=self.shm_name)
            shared_array = np.ndarray((1,), dtype=np.float64, buffer=shared_mem.buf)
            shared_array[0] = self.swarm_id
            shared_mem.close()

        self.metrics.log_swarm_change(swarm_id)

        if isinstance(swarm_id, list):
            self.hierarchy = swarm_id[self.hid]

    def go_to_next_hierarchy(self):
        self.hid = (self.hid + 1) % len(self.swarm_id)
        self.hierarchy = self.swarm_id[self.hid]

    def set_el(self, el):
        self.el = el
        self.metrics.log_coordinate_change(el)

    def set_query_id(self, query_id):
        self.query_id = query_id

    def set_challenge_id(self, challenge_id):
        self.challenge_id = challenge_id

    def set_anchor(self, anchor):
        self.anchor = anchor

    def set_radio_range(self, radio_range):
        self.radio_range = radio_range

    def deploy(self):
        self.metrics.log_illuminate(self.gtl)
        self.move(self.gtl - self.el)

    def fail(self):
        self.reset_swarm()
        if Config.STANDBY:
            self.set_el(self.stand_by_coord)
        else:
            self.set_el(np.array([.0, .0, .0]))
        self.radio_range = Config.INITIAL_RANGE
        self.anchor = None
        self.query_id = None
        self.challenge_id = None
        self.lease = dict()
        # self.history.log(MetricTypes.FAILURES, 1)
        self.metrics.log_sum("A5_num_failures", 1)
        self.metrics.log_failure()

    def move(self, vector):
        erred_v = self.add_dead_reckoning_error(vector)
        dest = self.el + erred_v
        # self.history.log(MetricTypes.LOCATION, self.el)
        self.metrics.log_sum("A0_total_distance", np.linalg.norm(vector))
        vm = velocity.VelocityModel(self.el, dest)
        vm.solve()
        dur = vm.total_time
        self.log_wait_time(dur)

        if Config.BUSY_WAITING:
            fin_time = time.time() + dur
            while True:
                if time.time() >= fin_time:
                    break
        else:
            time.sleep(dur)

        self.set_el(dest)

    def add_dead_reckoning_error(self, vector):
        if vector[0] or vector[1]:
            i = np.array([vector[1], -vector[0], 0])
        elif vector[2]:
            i = np.array([vector[2], 0, -vector[0]])
        else:
            return vector

        if self.alpha == 0:
            return vector

        j = np.cross(vector, i)
        norm_i = np.linalg.norm(i)
        norm_j = np.linalg.norm(j)
        norm_v = np.linalg.norm(vector)
        i = i / norm_i
        j = j / norm_j
        phi = np.random.uniform(0, 2 * np.pi)
        error = np.sin(phi) * i + np.cos(phi) * j
        r = np.linalg.norm(vector) * np.tan(self.alpha)

        erred_v = vector + np.random.uniform(0, r) * error
        return norm_v * erred_v / np.linalg.norm(erred_v)

    def update_neighbor(self, ctx):
        if ctx.fid != -1:
            if np.linalg.norm(self.el - ctx.el) <= Config.SWEET_RANGE[1]:
                self.neighbors[ctx.fid] = ctx
            if Config.SWEET_RANGE[0] <= np.linalg.norm(self.el - ctx.el) <= Config.SWEET_RANGE[1]:
                self.sweet_neighbors[ctx.fid] = ctx
            self.update_relative_pose(ctx)

    @cache
    def is_fls_obstructed(self, self_el, neighbors, el):
        for point in neighbors:
            if intersects_sphere(self_el, el, point, self.radius):
                return True

    def quadratic_function(self, x, coeffs):
        a, b, c = coeffs
        return a * x ** 2 + b * x + c

    def add_camera_error(self, v):
        d = np.linalg.norm(v)
        if Config.SS_ERROR_MODEL == 1:
            if d < 1e-9:
                return v, d
            x = self.quadratic_function(d, self.error_coefficients)
            new_d = d + x * d
            return v / d * new_d, new_d
        return v, d

    def update_relative_pose(self, ctx):
        if ctx.swarm_id == self.swarm_id:
            num_edges = len(self.tree.edges)

            if ctx.fid == self.intra_localizer:  # sender is the parent
                p = self.neighbors[ctx.fid].el - self.el
                p, _ = self.add_camera_error(p)
                self.tree.add_edge(ctx.fid, self.fid, p=p)
                self.tree.add_edge(self.fid, ctx.fid, p=-p)
            if ctx.pid == self.fid:  # sender is a child
                p = self.el - self.neighbors[ctx.fid].el
                p, _ = self.add_camera_error(p)
                self.tree.add_edge(self.fid, ctx.fid, p=p)
                self.tree.add_edge(ctx.fid, self.fid, p=-p)
            if ctx.pid in self.neighbors:
                p = self.neighbors[ctx.pid].el - self.neighbors[ctx.fid].el
                p, _ = self.add_camera_error(p)
                self.tree.add_edge(ctx.pid, ctx.fid, p=p)
                self.tree.add_edge(ctx.fid, ctx.pid, p=-p)

            if len(self.tree.edges) > num_edges:
                if self.fid in self.tree.nodes:
                    self.paths = nx.shortest_path(self.tree, self.fid)

            for fid, path in self.paths.items():
                self.absolute_poses[fid] = np.array([.0, .0, .0])
                for i in range(len(path) - 1):
                    if path[i] != path[i + 1]:
                        self.absolute_poses[fid] += self.tree[path[i]][path[i+1]]['p']

    def update_relative_pose_2(self, ctx):
        if ctx.swarm_id == self.swarm_id:
            if ctx.fid == self.intra_localizer:  # sender is the parent
                p = self.neighbors[ctx.fid].el - self.el
                self.rd = np.linalg.norm(p)

    def increment_range(self):
        if time.time() - self.last_expanded > 0.05:
            if self.radio_range < Config.MAX_RANGE:
                self.set_radio_range(self.radio_range + 5)

    def reset_range(self):
        self.set_radio_range(Config.INITIAL_RANGE)

    def reset_swarm(self):
        self.set_swarm_id(self.fid)

    def log_received_message(self, msg_type, length):
        meta = {"length": length}
        # self.history.log(MetricTypes.RECEIVED_MASSAGES, msg_type, meta)
        self.metrics.log_received_msg(msg_type, length)

    def log_dropped_messages(self):
        # self.history.log_sum(MetricTypes.DROPPED_MESSAGES)
        self.metrics.log_sum("A4_num_dropped_messages", 1)

    def log_sent_message(self, msg_type, length):
        meta = {"length": length}
        # self.history.log(MetricTypes.SENT_MESSAGES, msg_type, meta)
        self.metrics.log_sent_msg(msg_type, length)
        self.message_id += 1

    def log_wait_time(self, dur):
        # self.history.log(MetricTypes.WAITS, dur)
        self.metrics.log_sum("A1_num_moved", 1)
        self.metrics.log_sum("A1_total_wait(s)", dur)
        self.metrics.log_max("A1_max_wait(s)", dur)
        self.metrics.log_min("A1_min_wait(s)", dur)
