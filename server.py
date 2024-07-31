import json
import socket
import pickle
import struct

import numpy as np
from multiprocessing import shared_memory
import scipy.io
import time
import os
import threading
from config import Config
from constants import Constants
from message import Message, MessageTypes
import worker
import utils
import sys
from stop import stop_all
import psutil
from datetime import datetime
import pandas as pd

hd_timer = None
hd_round = []
hd_time = []
should_stop = False


def join_config_properties(conf, props):
    return "_".join(
        f"{k[1] if isinstance(k, tuple) else k}{getattr(conf, k[0] if isinstance(k, tuple) else k)}" for k in
        props)


def query_swarm_client(connection):
    query_msg = Message(MessageTypes.QUERY_SWARM)
    connection.send(pickle.dumps(query_msg))


def pull_swarm_client(connection):
    data = recv_msg(connection)
    message = pickle.loads(data)
    return message.args[0], message.args[1]


def send_msg(sock, msg):
    # Prefix each message with a 4-byte big-endian unsigned integer (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def stop_client(connection):
    stop_msg = Message(MessageTypes.STOP)
    connection.send(pickle.dumps(stop_msg))


def wait_for_client(sock):
    sock.recv(1)
    sock.close()


def set_stop():
    global should_stop
    should_stop = True
    print('will stop next round')


def compute_hd(sh_arrays, gtl):
    hd_t = utils.hausdorff_distance(np.stack(sh_arrays), gtl)
    print(f"__hd__ {hd_t}")
    return hd_t


def compute_swarm_size(sh_arrays):
    swarm_counts = {}
    for arr in sh_arrays:
        swarm_id = arr[0]
        if swarm_id in swarm_counts:
            swarm_counts[swarm_id] += 1
        else:
            swarm_counts[swarm_id] = 1
    return swarm_counts


def read_cliques_xlsx(path):
    df = pd.read_excel(path, sheet_name='cliques')
    return [np.array(eval(c)) for c in df["7 coordinates"]]


def read_groups(dir_experiment, file_name):
    groups = read_cliques_xlsx(os.path.join(dir_experiment, f'{file_name}.xlsx'))

    single_members = []
    single_indexes = []
    max_dist_singles = 0
    for k in range(len(groups)):
        if groups[k].shape[0] == 1:
            if len(single_indexes):
                max_dist_n = np.max(np.linalg.norm(np.stack(single_members) - groups[k][0], axis=1))
                max_dist_singles = max(max_dist_singles, max_dist_n)
            single_members.append(groups[k][0])
            single_indexes.append(k)

    # remove single nodes from groups
    for k in reversed(single_indexes):
        groups.pop(k)
        # radio_ranges.pop(k)

    # add single nodes as one group to the groups
    if len(single_members):
        groups.append(np.stack(single_members))
        # radio_ranges.append(max_dist_singles)

    return groups


if __name__ == '__main__':
    N = 1
    nid = 0
    experiment_name = str(int(time.time()))
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
        nid = int(sys.argv[2])
        experiment_name = sys.argv[3]

    IS_CLUSTER_SERVER = N != 1 and nid == 0
    IS_CLUSTER_CLIENT = N != 1 and nid != 0

    if IS_CLUSTER_SERVER:
        ServerSocket = socket.socket()
        ServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            try:
                ServerSocket.bind(Constants.SERVER_ADDRESS)
            except OSError:
                time.sleep(10)
                continue
            break
        ServerSocket.listen(N - 1)

        clients = []
        for i in range(N - 1):
            client, address = ServerSocket.accept()
            print(address)
            clients.append(client)

    if IS_CLUSTER_CLIENT:
        client_socket = socket.socket()
        while True:
            try:
                client_socket.connect(Constants.SERVER_ADDRESS)
            except OSError:
                time.sleep(10)
                continue
            break

    dir_name = None
    current_date_time = datetime.now().strftime("%H%M%S_%m%d%Y")
    if len(Config.FILE_NAME_KEYS):
        keys = join_config_properties(Config, Config.FILE_NAME_KEYS)
    else:
        keys = current_date_time
    file_name = f"{Config.SHAPE}_{keys}_{experiment_name}"

    if len(Config.DIR_KEYS):
        dir_name = join_config_properties(Config, Config.DIR_KEYS)

    main_dir = Config.RESULTS_PATH if dir_name is None else os.path.join(Config.RESULTS_PATH, Config.SHAPE, dir_name)
    results_directory = os.path.join(main_dir, file_name)
    shape_directory = main_dir
    print(main_dir)
    # exit()

    # results_directory = os.path.join(Config.RESULTS_PATH, Config.SHAPE, experiment_name)
    # shape_directory = os.path.join(Config.RESULTS_PATH, Config.SHAPE)
    if not os.path.exists(results_directory):
        os.makedirs(os.path.join(results_directory, 'json'), exist_ok=True)
        os.makedirs(os.path.join(results_directory, 'timeline'), exist_ok=True)
    point_cloud = np.loadtxt(f'assets/{Config.SHAPE}.txt', delimiter=',')

    if Config.SAMPLE_SIZE != 0:
        # np.random.shuffle(point_cloud)
        point_cloud = point_cloud[:Config.SAMPLE_SIZE]

    total_count = point_cloud.shape[0]
    h = np.log2(total_count)
    # print(h)

    gtl_point_cloud = np.random.uniform(0, 5, size=(total_count, 3))
    # x y z swarm_id is_failed
    sample = np.array([0.0])

    s = Config.SCALE
    node_point_idx = []
    for i in range(total_count):
        if i % N == nid:
            node_point_idx.append(i)
        gtl_point_cloud[i] = np.array([point_cloud[i][0]*s, point_cloud[i][1]*s, point_cloud[i][2]*s])

    count = len(node_point_idx)

    processes = []
    shared_arrays = []
    shared_memories = []

    local_gtl_point_cloud = []

    try:
        if Config.GROUP:
            groups = read_groups(Config.RESULTS_PATH, 'skateboard_K3')
            pid = 0
            count = 0
            stop_dispatcher = False
            for i in range(len(groups)):
                group = groups[i]
                member_count = group.shape[0]
                sum_x = np.sum(group[:, 0])
                sum_y = np.sum(group[:, 1])
                sum_z = np.sum(group[:, 2])
                stand_by_coord = np.array([
                    float(round(sum_x / member_count)),
                    float(round(sum_y / member_count)),
                    float(round(sum_z / member_count))
                ])

                # deploy group members
                for member_coord in group:
                    pid += 1
                    if (pid - 1) % N == nid:
                        count += 1
                        shm = shared_memory.SharedMemory(create=True, size=sample.nbytes)
                        shared_array = np.ndarray(sample.shape, dtype=sample.dtype, buffer=shm.buf)
                        shared_array[:] = sample[:]

                        shared_arrays.append(shared_array)
                        shared_memories.append(shm)
                        local_gtl_point_cloud.append(member_coord)
                        p = worker.WorkerProcess(count, pid, 1, member_coord, np.array([0, 0, 0]), shm.name,
                                                 results_directory, stand_by_coord)
                        p.start()
                        processes.append(p)
                    if pid == Config.SAMPLE_SIZE:
                        stop_dispatcher = True
                        break
                if stop_dispatcher:
                    break

        else:
            if Config.GROUP_TYPE == 'hierarchical' \
                    or Config.GROUP_TYPE == 'sequential' \
                    or Config.GROUP_TYPE == 'spanning' \
                    or Config.GROUP_TYPE == 'spanning_2' \
                    or Config.GROUP_TYPE == 'spanning_2_v2' \
                    or Config.GROUP_TYPE == 'spanning_2_v3' \
                    or Config.GROUP_TYPE == 'spanning_3':
                with open(f"assets/{Config.SHAPE}_localizer.json") as f:
                    localizer = json.load(f)
                # print(localizer)

                if Config.GROUP_TYPE == 'spanning_2' \
                        or Config.GROUP_TYPE == 'spanning_2_v2' \
                        or Config.GROUP_TYPE == 'spanning_2_v3':
                    with open(f"assets/{Config.SHAPE}_intra_localizer.json") as f:
                        intra_localizer = json.load(f)
                    gid_to_swarm_gtl = {}
                    for i, row in enumerate(point_cloud):
                        gid = row[3]
                        pid = row[4]
                        parent_id = intra_localizer[str(pid)] if str(pid) in intra_localizer else None
                        if gid in gid_to_swarm_gtl:
                            gid_to_swarm_gtl[gid][pid] = gtl_point_cloud[i]
                        else:
                            gid_to_swarm_gtl[gid] = {pid: gtl_point_cloud[i]}

                else:
                    intra_localizer = {}

                for i in node_point_idx:
                    if Config.GROUP_TYPE == 'spanning' or Config.GROUP_TYPE == 'spanning_2' \
                            or Config.GROUP_TYPE == 'spanning_2_v2' \
                            or Config.GROUP_TYPE == 'spanning_2_v3':
                        gid = int(point_cloud[i, 3])
                        swarm_gtl = gid_to_swarm_gtl[gid]
                    elif Config.GROUP_TYPE == 'spanning_3':
                        gid = [int(point_cloud[i, 3]), int(point_cloud[i, 5])]
                    else:
                        gid = [int(g) for g in point_cloud[i, 3:].tolist()]

                    if Config.GROUP_TYPE == 'spanning_2' or Config.GROUP_TYPE == 'spanning_3' \
                            or Config.GROUP_TYPE == 'spanning_2_v2' \
                            or Config.GROUP_TYPE == 'spanning_2_v3':
                        pid = int(point_cloud[i, 4])
                        idx = pid
                    else:
                        pid = i + 1
                        idx = i

                    anchor_for = {}
                    if Config.GROUP_TYPE == 'spanning_2_v3':
                        for loc, anc in intra_localizer.items():
                            if anc in anchor_for:
                                anchor_for[anc].append(int(loc))
                            else:
                                anchor_for[anc] = [int(loc)]

                    local_gtl_point_cloud.append(gtl_point_cloud[i])
                    p = worker.WorkerProcess(
                        count,
                        pid,
                        gid,
                        gtl_point_cloud[i],
                        np.array([0, 0, 0]),
                        None,
                        results_directory,
                        None,
                        localizer[str(idx)] if str(idx) in localizer else [],
                        intra_localizer=intra_localizer[str(idx)] if str(idx) in intra_localizer else None,
                        anchor=anchor_for[idx] if idx in anchor_for else [],
                        swarm_gtl=swarm_gtl
                    )
                    p.start()
                    processes.append(p)
            elif Config.GROUP_TYPE == 'bin_overlapping':
                with open(f"assets/{Config.SHAPE}_bin_overlapping.json") as f:
                    bin_groups = json.load(f)
                # print(localizer)
                for i in node_point_idx:
                    local_gtl_point_cloud.append(gtl_point_cloud[i])
                    p = worker.WorkerProcess(
                        count,
                        i + 1,
                        bin_groups[i],
                        gtl_point_cloud[i],
                        np.array([0, 0, 0]),
                        None,
                        results_directory,
                        None,
                        None,
                    )
                    p.start()
                    processes.append(p)
            elif Config.GROUP_TYPE == 'mst' or Config.GROUP_TYPE == 'universal':
                with open(f"assets/{Config.SHAPE}_localizer.json") as f:
                    data = json.load(f)
                    localizer = data["localizer"]
                    anchor = data["anchor"]

                for i in node_point_idx:
                    pid = int(point_cloud[i, 3])

                    local_gtl_point_cloud.append(gtl_point_cloud[i])
                    p = worker.WorkerProcess(
                        count,
                        pid,
                        pid,
                        gtl_point_cloud[i],
                        np.array([0, 0, 0]),
                        None,
                        results_directory,
                        None,
                        localizer[str(pid)] if str(pid) in localizer else None,
                        anchor=anchor[str(pid)] if str(pid) in anchor else [],
                        intra_localizer=None
                    )
                    p.start()
                    processes.append(p)
            else:
                for i in node_point_idx:
                    local_gtl_point_cloud.append(gtl_point_cloud[i])
                    p = worker.WorkerProcess(
                        count,
                        i + 1,
                        [int(g) for g in point_cloud[i, 3:].tolist()],
                        gtl_point_cloud[i],
                        np.array([0, 0, 0]),
                        None,
                        results_directory,
                        None,
                        None,
                    )
                    p.start()
                    processes.append(p)
    except OSError as e:
        print(e)
        for p in processes:
            p.terminate()
        for s in shared_memories:
            s.close()
            s.unlink()
        exit()

    print(count)
    gtl_point_cloud = local_gtl_point_cloud

    if nid == 0:
        threading.Timer(Config.DURATION, set_stop).start()

    print('waiting for processes ...')

    ser_sock = worker.WorkerSocket()
    old_swarmer = False

    if IS_CLUSTER_CLIENT:
        while True:
            server_msg = client_socket.recv(2048)
            server_msg = pickle.loads(server_msg)

            if server_msg.type == MessageTypes.QUERY_SWARM:
                swarms = compute_swarm_size(shared_arrays)
                response = Message(MessageTypes.REPLY_SWARM, args=(swarms, psutil.cpu_percent()))
                send_msg(client_socket, pickle.dumps(response))
            elif server_msg.type == MessageTypes.STOP:
                break
    else:
        if old_swarmer:
            reset = True
            last_thaw_time = time.time()
            round_duration = 0
            last_merged_flss = 0
            no_change_counter = 0
            server_cpu = []
            server_time = []
            while True:
                time.sleep(0.2)
                t = time.time()

                swarms = compute_swarm_size(shared_arrays)
                server_cpu.append([psutil.cpu_percent()])
                server_time.append(time.time())

                if IS_CLUSTER_SERVER:
                    for i in range(N - 1):
                        query_swarm_client(clients[i])

                    for i in range(N - 1):
                        client_swarms, client_cpu = pull_swarm_client(clients[i])
                        server_cpu[-1].append(client_cpu)
                        for sid in client_swarms:
                            if sid in swarms:
                                swarms[sid] += client_swarms[sid]
                            else:
                                swarms[sid] = client_swarms[sid]

                largest_swarm = max(swarms.values())
                num_swarms = len(swarms)
                thaw_condition = False

                if largest_swarm == total_count or num_swarms == 1 or (t - last_thaw_time >= h):
                    # if reset:
                    print(largest_swarm, num_swarms)
                    thaw_message = Message(MessageTypes.THAW_SWARM, args=(t,)).from_server().to_all()
                    ser_sock.broadcast(thaw_message)
                    if round_duration == 0 and largest_swarm == total_count:
                        round_duration = t - last_thaw_time
                    last_thaw_time = t

                if should_stop:
                    stop_all()
                    break
        else:
            time.sleep(Config.DURATION)
            stop_all()

    if IS_CLUSTER_SERVER:
        for i in range(N - 1):
            stop_client(clients[i])

        client_threads = []
        for client in clients:
            t = threading.Thread(target=wait_for_client, args=(client,))
            t.start()
            client_threads.append(t)
        for t in client_threads:
            t.join()

        ServerSocket.close()
        print("secondary nodes are done")

    for p in processes:
        p.join(20)
        if p.is_alive():
            continue

    for p in processes:
        if p.is_alive():
            print("timeout")
            p.terminate()

    # if Config.PROBABILISTIC_ROUND or Config.CENTRALIZED_ROUND:
    # utils.write_hds_time(hd_time, results_directory, nid)
    # else:
    #     utils.write_hds_round(hd_round, round_time, results_directory, nid)
    # if Config.DURATION < 660:
    #     utils.write_swarms(swarms_metrics, round_time, results_directory, nid)

    if nid == 0:
        utils.write_configs(results_directory)
        print("primary node is done")

    for s in shared_memories:
        s.close()
        s.unlink()

    if IS_CLUSTER_CLIENT:
        time.sleep(20)
        client_socket.send(struct.pack('b', True))
        client_socket.close()

    if nid == 0:
        pass
        # with open(f"{results_directory}/utilization.json", "w") as f:
        #     json.dump([server_time, server_cpu], f)
        # print("wait a fixed time for other nodes")
        # time.sleep(90)

        utils.create_csv_from_json(results_directory)
        utils.combine_csvs(results_directory, results_directory)
        utils.gen_sw_charts(results_directory, "*", file_name, False)
