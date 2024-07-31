import threading
import time
from queue import Empty

from config import Config
from message import MessageTypes, Message
from state import StateTypes
from worker.network import PrioritizedItem


class HandlerThread(threading.Thread):
    def __init__(self, event_queue, state_machine, context):
        super(HandlerThread, self).__init__()
        self.event_queue = event_queue
        self.state_machine = state_machine
        self.context = context
        self.last_challenge = 0
        self.last_lease_renew = 0
        self.start_time = 0

    def run(self):
        self.start_time = time.time()
        self.state_machine.start()
        while True:
            t = time.time()

            if t - self.last_challenge > Config.STATE_TIMEOUT:
                self.state_machine.reenter_available_state()
                self.last_challenge = t
            if t - self.start_time > Config.DURATION * 1.1:
                # print(f"{self.context.fid}_timeout")
                self.state_machine.handle_stop(Message(MessageTypes.STOP).from_server().to_all())
                print(f"{self.context.fid}_stopped")
                break
            try:
                item = self.event_queue.get(timeout=0.05)
            except Empty:
                continue
            if item.stale:
                continue

            event = item.event
            self.state_machine.drive(event)

            if event.type == MessageTypes.STOP:
                break

            # self.flush_queue()

    def flush_queue(self):
        with self.event_queue.mutex:
            for item in self.event_queue.queue:
                t = item.event.type
                if t == MessageTypes.SIZE_REPLY or t == MessageTypes.THAW_SWARM or t == MessageTypes.STOP\
                        or t == MessageTypes.LEASE_RENEW or t == MessageTypes.LEASE_CANCEL:
                    item.stale = False
                elif t == MessageTypes.RENEW_LEASE_INTERNAL \
                        or t == MessageTypes.SET_AVAILABLE_INTERNAL \
                        or t == MessageTypes.FAIL_INTERNAL \
                        or t == MessageTypes.THAW_SWARM_INTERNAL:
                    item.stale = False
                elif t == MessageTypes.CHALLENGE_FIN or t == MessageTypes.CHALLENGE_INIT\
                        or t == MessageTypes.CHALLENGE_ACK or t == MessageTypes.CHALLENGE_ACCEPT:
                    if item.event.swarm_id != self.context.swarm_id:
                        item.stale = False
                    else:
                        item.stale = True
                else:
                    item.stale = True

    def flush_all(self):
        with self.event_queue.mutex:
            for item in self.event_queue.queue:
                item.stale = True
