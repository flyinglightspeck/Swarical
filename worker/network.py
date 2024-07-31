import random
import time
from dataclasses import dataclass, field
from socket import socket
from typing import Any
import threading
import numpy as np
from config import Config
import message
from message import MessageTypes
from state import StateTypes


general_messages = {
    MessageTypes.STOP,
    MessageTypes.GOSSIP,
    MessageTypes.FOLLOW,
    MessageTypes.MERGE,
}

valid_state_messages = {
    StateTypes.AVAILABLE: general_messages,
    StateTypes.DEPLOYING: dict(),
}


class NetworkThread(threading.Thread):
    def __init__(self, event_queue, state_machine, context, sock):
        super(NetworkThread, self).__init__()
        self.event_queue = event_queue
        self.state_machine = state_machine
        self.context = context
        self.sock = sock
        self.latest_message_id = dict()
        self.last_lease_renew = 0
        self.last_challenge = 0
        self.start_time = 0
        self.last_fail_check = 0

    def run(self):
        self.start_time = time.time()
        self.last_fail_check = self.start_time + 1
        while True:
            t = time.time()

            if t - self.start_time > Config.DURATION * 1.1:
                break
            # if Config.FAILURE_TIMEOUT and t - self.last_fail_check > Config.FAILURE_TIMEOUT:
            #     self.state_machine.set_fail()
            #     self.last_fail_check = t
            # if random.random() < 0.001:
            #     time.sleep(0.005)
            # if self.sock.is_ready():
            try:
                msg, length = self.sock.receive()
            except BlockingIOError as e:
                continue
            except Exception as e:
                print(e)
                continue
            # self.context.log_received_message(msg.type, length)
            if self.is_message_valid(msg):
                self.context.log_received_message(msg.type, length)
                self.latest_message_id[msg.fid] = msg.id
                self.handle_immediately(msg)
                if msg is not None and msg.type == message.MessageTypes.STOP:
                    break

    def handle_immediately(self, msg):
        self.event_queue.put(NetworkThread.prioritize_message(msg))

    def is_message_valid(self, msg):
        if msg is None:
            self.context.log_dropped_messages()
            return False
        if msg.type == message.MessageTypes.STOP:
            return True
        if Config.DROP_PROB_RECEIVER:
            if np.random.random() <= Config.DROP_PROB_RECEIVER:
                self.context.log_dropped_messages()
                return False
        if msg.fid == self.context.fid:
            return False
        if msg.mod and self.context.fid % msg.mod[0] != msg.mod[1]:
            return False
        if msg.div and (self.context.fid - 1) // msg.div[0] != msg.div[1]:
            return False
        if msg.dest_fid != self.context.fid and msg.dest_fid != '*':
            return False
        if isinstance(self.context.swarm_id, list):
            if msg.dest_swarm_id not in self.context.swarm_id and msg.dest_swarm_id != '*':
                return False
        elif msg.dest_swarm_id != self.context.swarm_id and msg.dest_swarm_id != '*':
            return False
        if msg.fid in self.latest_message_id and msg.id < self.latest_message_id[msg.fid]:
            return False
        # if msg.type == message.MessageTypes.CHALLENGE_INIT or msg.type == message.MessageTypes.GOSSIP:
        #     dist = np.linalg.norm(msg.el - self.context.el)
        #     if dist > msg.range:
        #         return False
        # if self.state_machine.state in valid_state_messages:
        #     if msg.type not in valid_state_messages[self.state_machine.state] or msg.type not in general_messages:
        #         return False
        return True

    @staticmethod
    def prioritize_message(msg):
        t = time.time()
        if msg.type == message.MessageTypes.STOP:
            return PrioritizedItem(0, t, msg, False)
        if msg.type == message.MessageTypes.FOLLOW or msg.type == message.MessageTypes.MERGE:
            return PrioritizedItem(1, t, msg, False)
        return PrioritizedItem(2, t, msg, False)


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    time: float
    event: Any = field(compare=False)
    stale: bool = field(compare=False)
