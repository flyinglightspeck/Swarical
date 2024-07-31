import math
import time

import numpy as np
from config import Config


class MetricTypes:
    RECEIVED_MASSAGES = 0
    SENT_MESSAGES = 1
    LOCATION = 2
    SWARM_ID = 3
    LEASES = 4
    WAITS = 5
    ANCHOR = 6
    LOCALIZE = 7
    DROPPED_MESSAGES = 8
    FAILURES = 9
    GRANTED_LEASE = 10
    EXPIRED_LEASE = 11
    CANCELED_LEASE = 12
    RELEASED_LEASE = 13


class TimelineEvents:
    COORDINATE = 1
    SWARM = 2
    COLOR = 3
    ILLUMINATE = 4
    FAIL = 5
    LEASE_EXP = 6


def update_dict_sum(obj, key):
    if key in obj:
        obj[key] += 1
    else:
        obj[key] = 1


def log_msg_hist(hist, msg_type, label, cat):
    key_number = f'{cat}0_num_{label}_{msg_type.name}'
    key_num_cat = f'{cat}1_cat_num_{label}_{msg_type.get_cat()}'

    update_dict_sum(hist, key_number)
    update_dict_sum(hist, key_num_cat)


def get_messages_histogram(msgs, label, cat):
    hist = dict()

    for msg_hist in msgs:
        msg_type = msg_hist.value
        log_msg_hist(hist, msg_type, label, cat)

    return hist


class Metrics:
    def __init__(self, history, results_directory):
        self.results_directory = results_directory
        self.history = history
        self.general_metrics = {
            "A0_total_distance": 0,
            "A1_num_moved": 0,
            "A1_min_wait(s)": math.inf,
            "A1_max_wait(s)": 0,
            "A1_total_wait(s)": 0,
            "A2_num_granted_leases": 0,
            "A2_num_expired_leases": 0,
            "A2_num_canceled_leases": 0,
            "A2_num_released_leases": 0,
            "A3_num_anchor": 0,
            "A3_num_localize": 0,
            "A4_bytes_sent": 0,
            "A4_bytes_received": 0,
            "A4_num_messages_sent": 0,
            "A4_num_messages_received": 0,
            "A4_num_dropped_messages": 0,
            "A5_num_failures": 0
        }
        self.sent_msg_hist = {}
        self.received_msg_hist = {}
        self.timeline = []

    def log_sum(self, key, value):
        self.general_metrics[key] += value

    def log_max(self, key, value):
        self.general_metrics[key] = max(self.general_metrics[key], value)

    def log_min(self, key, value):
        self.general_metrics[key] = min(self.general_metrics[key], value)

    def log_coordinate_change(self, coord):
        self.timeline.append([time.time(), TimelineEvents.COORDINATE, coord.tolist()])

    def log_swarm_change(self, swarm):
        self.timeline.append([time.time(), TimelineEvents.SWARM, swarm])

    def log_failure(self):
        self.timeline.append([time.time(), TimelineEvents.FAIL])

    def log_illuminate(self, coord):
        self.timeline.append([time.time(), TimelineEvents.ILLUMINATE, coord.tolist()])

    def log_lease_expiration(self):
        self.timeline.append([time.time(), TimelineEvents.LEASE_EXP])

    def log_received_msg(self, msg_type, length):
        log_msg_hist(self.received_msg_hist, msg_type, 'received', 'C')
        self.log_sum("A4_num_messages_received", 1)
        self.log_sum("A4_bytes_received", length)

    def log_sent_msg(self, msg_type, length):
        log_msg_hist(self.sent_msg_hist, msg_type, 'sent', 'B')
        self.log_sum("A4_num_messages_sent", 1)
        self.log_sum("A4_bytes_sent", length)

    def get_received_messages(self):
        return self.history[MetricTypes.RECEIVED_MASSAGES]

    def get_sent_messages(self):
        return self.history[MetricTypes.SENT_MESSAGES]

    def get_final_report_(self):
        report = {}
        report.update(self.general_metrics)
        report.update(self.sent_msg_hist)
        report.update(self.received_msg_hist)
        return report
