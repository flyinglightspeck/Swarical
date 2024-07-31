class Constants:
    BROADCAST_ADDRESS = ("<broadcast>", 5000)
    WORKER_ADDRESS = ("", 5000)


import socket
import pickle


class WorkerSocket:
    def __init__(self):
        self.sock = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(Constants.WORKER_ADDRESS)
        self.sock = sock

    def close(self):
        self.sock.close()

    def receive(self):
        data, _ = self.sock.recvfrom(1024)
        try:
            msg = pickle.loads(data)
            return msg, len(data)
        except pickle.UnpicklingError:
            return None, 0

    def broadcast(self, msg, retry=2):
        data = pickle.dumps(msg)
        try:
            self.sock.sendto(data, Constants.BROADCAST_ADDRESS)
        except OSError:
            if retry:
                self.broadcast(msg, retry - 1)
        return len(data)

    def send_test_msgs(self):
        id = 1
        n = 5
        for i in range(n):
            self.broadcast((id, i))

        for i in range(n):
            print(ws.receive())

        self.close()


if __name__ == '__main__':
    ws = WorkerSocket()
    ws.send_test_msgs()
