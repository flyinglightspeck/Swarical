import json
import os
import random
import time
from enum import Enum

import numpy as np


from message import Message, MessageTypes
from config import Config
from worker.network import PrioritizedItem
from .types import StateTypes


class StateMachine:
    def __init__(self, context, sock, metrics, event_queue):
        self.last_challenge_init = 0
        self.last_challenge_accept = 0
        self.state = StateTypes.DEPLOYING
        self.context = context
        self.metrics = metrics
        self.sock = sock
        self.should_fail = False
        self.event_queue = event_queue
        self.start_time = 0
        self.waiting_mode = False
        self.num_localizations = 0
        self.notified = False
        self.last_entered_wm = None
        self.error_coefficients = None
        self.num_intra_loc = 0

    def start(self):
        if Config.CAMERA == 'w':
            # wide camera
            x = np.array([200, 150, 100, 75, 50, 45, 42.5, 40]) / 10  # cm
            y = np.array([30.46, 11.68, 5.82, 0.893333, 2.14, 5.4, 4.16471, 6.05]) / 100  # percent
        else:
            # Reg camera
            x = np.array([300, 200, 150, 100, 75, 70]) / 10  # cm
            y = np.array([29.92, 7.74, 3.25333, 2.58, 1.94667, 1.35714]) / 100  # percent

        self.error_coefficients = np.polyfit(x, y, 2) - np.array([0, 0, .015])

        self.handle_follow = self.handle_follow_all
        if Config.GROUP_TYPE == 'spanning_2':
            self.localize = self.localize_spanning_2
        elif Config.GROUP_TYPE == 'spanning_2_v2':
            self.localize = self.localize_spanning_2_variant_2
        elif Config.GROUP_TYPE == 'spanning_2_v3':
            self.localize = self.localize_spanning_2_variant_3
            self.handle_follow = self.handle_follow_v3
        else:
            raise ValueError('Unknown group type')
        self.context.deploy()
        self.start_time = time.time()
        self.enter(StateTypes.AVAILABLE)

    def quadratic_function(self, x, coeffs):
        a, b, c = coeffs
        return a * x ** 2 + b * x + c

    def add_ss_error_1(self, v, d, coefficients, rd):
        if d < 1e-9:
            return v, d
        # print(rd)
        x = self.quadratic_function(rd, coefficients)
        new_d = d + x * d
        return v / d * new_d, new_d

    def sample_distance(self, _v, _d, c):
        vs = []
        ds = []

        for i in range(Config.SS_NUM_SAMPLES):
            if Config.SS_ERROR_MODEL == 1:
                v, d = self.add_ss_error_1(_v, _d, c, self.context.rd)
            else:
                v, d = _v, _d

            vs.append(v)
            ds.append(d)

        if Config.SS_SAMPLE_DELAY:
            time.sleep(Config.SS_SAMPLE_DELAY * Config.SS_NUM_SAMPLES)

        # average
        avg_d = np.average(ds)
        return _v * avg_d / _d, avg_d

    def set_waiting_mode(self, v):
        self.waiting_mode = v
        if v:
            self.last_entered_wm = time.time()
            self.num_intra_loc = 0

    def handle_follow_all(self, msg):
        self.context.move(msg.args[0])
        self.context.neighbors = {}
        self.context.absolute_poses = {}
        if msg.args[1]:
            self.set_waiting_mode(False)

            for fid, gid in self.context.localizer:
                if gid is None:
                    self.broadcast(Message(MessageTypes.NOTIFY).to_fls_id(fid, "*"))

    def handle_follow_v3(self, msg):
        self.context.move(msg.args[0])
        self.context.neighbors = {}
        self.context.absolute_poses = {}

    def handle_merge(self, msg):
        if msg.dest_swarm_id == "*":
            self.broadcast(Message(MessageTypes.MERGE).to_swarm_id(self.context.hierarchy))

        self.context.go_to_next_hierarchy()

    def handle_stop(self, msg):
        self.broadcast(msg)
        _final_report = self.metrics.get_final_report_()
        file_name = self.context.fid

        with open(os.path.join(self.metrics.results_directory, 'json', f"{file_name:05}.json"), "w") as f:
            json.dump(_final_report, f)

        with open(os.path.join(self.metrics.results_directory, "timeline", f"timeline_{self.context.fid:05}.json"), "w") as f:
            json.dump(self.metrics.timeline, f)

    def compute_v(self, anchor, el=None, gtl=None):
        if el is not None:
            d_gtl = self.context.gtl - gtl
            d_el = el
        else:
            d_gtl = self.context.gtl - anchor.gtl
            d_el = self.context.el - anchor.el

        d_el, _ = self.sample_distance(d_el, np.linalg.norm(d_el), self.error_coefficients)

        v = d_gtl - d_el
        d = np.linalg.norm(v)

        return v, d

    # HC
    def localize_spanning_2(self):
        if not self.waiting_mode:
            n1 = list(filter(lambda x: self.context.min_gid == x.swarm_id, self.context.neighbors.values()))

            adjustments = np.array([[.0, .0, .0]])
            if len(n1):
                adjustments = np.vstack((adjustments, [self.compute_v(n)[0] for n in n1]))
                v = np.mean(adjustments, axis=0)
                if Config.SS_ERROR_MODEL == 1:
                    if self.num_intra_loc < 5:
                        self.context.move(v)
                        self.num_intra_loc += 1
                    else:
                        self.set_waiting_mode(True)
                else:
                    if np.linalg.norm(v) > 1e-6:
                        self.context.move(v)
                    else:
                        self.set_waiting_mode(True)
            self.broadcast(Message(MessageTypes.GOSSIP).to_swarm_id(self.context.min_gid))
        else:
            for fid, gid in self.context.localizer:
                if gid is not None and fid in self.context.neighbors:
                    v, _ = self.compute_v(self.context.neighbors[fid])
                    self.context.move(v)
                    self.num_localizations += 1
                    stop = self.num_localizations == 3
                    self.broadcast(Message(MessageTypes.FOLLOW, args=(v, stop)).to_swarm_id(gid))
                    if stop:
                        self.num_localizations = 0
                        self.set_waiting_mode(False)
                # send your location
                self.broadcast(Message(MessageTypes.GOSSIP).to_fls_id(fid, "*"))

    #v2 ISR in-order inter-group localization
    def localize_spanning_2_variant_2(self):
        if not self.waiting_mode:
            n1 = list(filter(lambda x: self.context.min_gid == x.swarm_id, self.context.neighbors.values()))

            adjustments = np.array([[.0, .0, .0]])
            if len(n1):
                adjustments = np.vstack((adjustments, [self.compute_v(n)[0] for n in n1]))
                v = np.mean(adjustments, axis=0)
                if Config.SS_ERROR_MODEL == 1:
                    if self.num_intra_loc < 5:
                        self.context.move(v)
                        self.num_intra_loc += 1
                    else:
                        self.set_waiting_mode(True)
                else:
                    if np.linalg.norm(v) > 1e-6:
                        self.context.move(v)
                    else:
                        self.set_waiting_mode(True)
            self.broadcast(Message(MessageTypes.GOSSIP).to_swarm_id(self.context.min_gid))
        elif self.notified or self.context.min_gid == 0:
            for fid, gid in self.context.localizer:
                if gid is not None and fid in self.context.neighbors:
                    # primary localizing
                    v, _ = self.compute_v(self.context.neighbors[fid])
                    self.context.move(v)
                    self.num_localizations += 1
                    stop = self.num_localizations == 1
                    self.broadcast(Message(MessageTypes.FOLLOW, args=(v, stop)).to_swarm_id(gid))
                    if stop:
                        self.num_localizations = 0
                        self.set_waiting_mode(False)
                        self.notified = False
                        self.context.neighbors = {}
                else:
                    # anchor
                    self.broadcast(Message(MessageTypes.NOTIFY).to_fls_id(fid, "*"))

    # RSF: in-order intergroup and in-order intra-group localization
    def localize_spanning_2_variant_3(self):
        # self.release_waiting_mode()
        if self.context.intra_localizer is None:
            # the primary
            for fid in self.context.anchor_for:
                self.broadcast(Message(MessageTypes.NOTIFY_INTERNAL).to_fls_id(fid, "*"))

            for fid, gid in self.context.localizer:
                if gid is None:
                    self.broadcast(Message(MessageTypes.NOTIFY).to_fls_id(fid, "*"))

        elif self.context.intra_localizer in self.context.neighbors:
            # non-primary localizer
            v, d = self.compute_v(self.context.neighbors[self.context.intra_localizer])
            self.context.move(v)
            self.context.neighbors = {}
            self.context.absolute_poses = {}

            for fid in self.context.anchor_for:
                self.broadcast(Message(MessageTypes.NOTIFY_INTERNAL).to_fls_id(fid, "*"))

            for fid, gid in self.context.localizer:
                if gid is None:
                    self.broadcast(Message(MessageTypes.NOTIFY).to_fls_id(fid, "*"))

        if self.notified:
            for fid, gid in self.context.localizer:
                if gid is not None and fid in self.context.neighbors:
                    # primary localizing
                    v, _ = self.compute_v(self.context.neighbors[fid])
                    self.context.move(v)
                    self.broadcast(Message(MessageTypes.FOLLOW, args=(v,)).to_swarm_id(gid))
                    self.notified = False
                    self.context.neighbors = {}
                    self.context.absolute_poses = {}

    def fail(self):
        self.should_fail = False
        self.enter(StateTypes.DEPLOYING)
        self.context.fail()
        self.start()

    def put_state_in_q(self, event):
        msg = Message(event).to_fls(self.context)
        item = PrioritizedItem(1, time.time(), msg, False)
        self.event_queue.put(item)

    def enter(self, state, arg={}):
        self.state = state

        # if self.state == StateTypes.AVAILABLE:
        self.localize()

    def reenter_available_state(self):
        self.enter(StateTypes.AVAILABLE)

    def drive(self, msg):
        if self.should_fail:
            self.fail()

        event = msg.type
        self.context.update_neighbor(msg)

        if event == MessageTypes.STOP:
            self.handle_stop(msg)
        else:
            if event == MessageTypes.FOLLOW:
                self.handle_follow(msg)
                self.enter(StateTypes.AVAILABLE)

            elif event == MessageTypes.MERGE:
                self.handle_merge(msg)
                self.enter(StateTypes.AVAILABLE)
            elif event == MessageTypes.NOTIFY:
                self.notified = True
                self.enter(StateTypes.AVAILABLE)
            elif event == MessageTypes.NOTIFY_INTERNAL:
                self.enter(StateTypes.AVAILABLE)

    def broadcast(self, msg):
        msg.from_fls(self.context)
        length = self.sock.broadcast(msg)
        self.context.log_sent_message(msg.type, length)

    def send_to_server(self, msg):
        msg.from_fls(self.context).to_server()
        self.sock.send_to_server(msg)

