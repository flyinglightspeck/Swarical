import multiprocessing
import queue

import state
from .network import NetworkThread
from .handler import HandlerThread
from .worker_socket import WorkerSocket
from .context import WorkerContext
from .history import History
from .metrics import Metrics


class WorkerProcess(multiprocessing.Process):
    def __init__(self, count, process_id, sid, gtl, el, shared_el,
                 results_directory, stand_by_coord, localizer, g_order=None,
                 intra_localizer=None, anchor=None, swarm_gtl=None, radius=0.5):
        super(WorkerProcess, self).__init__()
        self.history = History(1)
        self.metrics = Metrics(self.history, results_directory)
        self.context = WorkerContext(count, process_id, sid, gtl, el, shared_el, self.metrics, stand_by_coord, localizer, g_order, intra_localizer, anchor, swarm_gtl, radius)
        self.sock = WorkerSocket()

    def run(self):
        event_queue = queue.PriorityQueue()
        state_machine = state.StateMachine(self.context, self.sock, self.metrics, event_queue)

        network_thread = NetworkThread(event_queue, state_machine, self.context, self.sock)
        handler_thread = HandlerThread(event_queue, state_machine, self.context)
        network_thread.start()
        handler_thread.start()

        handler_thread.join()
        network_thread.join()

        # print(f"process_stopped_{self.context.fid}")

        # while not event_queue.empty():
        #     try:
        #         event_queue.get(False)
        #     except queue.Empty:
        #         continue
        #     event_queue.task_done()
        # exit()
