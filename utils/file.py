import heapq
import math
import os
import json
import csv
import subprocess

import matplotlib as mpl

import numpy as np
from matplotlib import pyplot as plt, ticker
from scipy.spatial.distance import cdist

from config import Config
import pandas as pd
import glob

from utils import hausdorff_distance
from utils.chamfer import chamfer_distance_optimized
from utils.hausdorff import hausdorff_distance_optimized
from worker.metrics import TimelineEvents
import pandas as pd


def write_json(fid, results, directory):
    with open(os.path.join(directory, 'json', f"{fid:05}.json"), "w") as f:
        json.dump(results, f)


def create_csv_from_json(directory):
    if not os.path.exists(directory):
        return

    headers_set = set()
    rows = []

    json_dir = os.path.join(directory, 'json')
    filenames = os.listdir(json_dir)
    filenames.sort()

    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(json_dir, filename)) as f:
                try:
                    data = json.load(f)
                    headers_set = headers_set.union(set(list(data.keys())))
                except json.decoder.JSONDecodeError:
                    print(filename)

    headers = list(headers_set)
    headers.sort()
    rows.append(['fid'] + headers)

    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(json_dir, filename)) as f:
                try:
                    data = json.load(f)
                    fid = filename.split('.')[0]
                    row = [fid] + [data[h] if h in data else 0 for h in headers]
                    rows.append(row)
                except json.decoder.JSONDecodeError:
                    print(filename)

    with open(os.path.join(directory, 'metrics.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def write_hds_time(hds, directory, nid):
    if not os.path.exists(directory):
        return

    headers = ['timestamp(s)', 'relative_time(s)', 'hd']
    rows = [headers]

    for i in range(len(hds)):
        row = [hds[i][0], hds[i][0] - hds[0][0], hds[i][1]]
        rows.append(row)

    with open(os.path.join(directory, f'hd-n{nid}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def write_hds_round(hds, rounds, directory, nid):
    if not os.path.exists(directory):
        return

    headers = ['round', 'time(s)', 'hd']
    rows = [headers]

    for i in range(len(hds)):
        row = [i+1, rounds[i+1] - rounds[0], hds[i][1]]
        rows.append(row)

    with open(os.path.join(directory, f'hd-n{nid}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def write_swarms(swarms, rounds, directory, nid):
    headers = [
        'timestamp(s)',
        'relative times(s)',
        'num_swarms',
        'average_swarm_size',
        'largest_swarm',
        'smallest_swarm',
    ]

    rows = [headers]

    for i in range(len(swarms)):
        t = swarms[i][0] - rounds[0]
        num_swarms = len(swarms[i][1])
        sizes = swarms[i][1].values()

        row = [swarms[i][0], t, num_swarms, sum(sizes)/num_swarms, max(sizes), min(sizes)]
        rows.append(row)

    with open(os.path.join(directory, f'swarms-n{nid}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def write_configs(directory):
    headers = ['config', 'value']
    rows = [headers]

    for k, v in vars(Config).items():
        if not k.startswith('__'):
            rows.append([k, v])

    with open(os.path.join(directory, 'config.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def combine_csvs(directory, xslx_dir):
    from datetime import datetime
    current_datetime = datetime.now()
    current_date_time = current_datetime.strftime("%H:%M:%S_%m:%d:%Y")

    csv_files = glob.glob(f"{directory}/*.csv")

    with pd.ExcelWriter(os.path.join(xslx_dir, f'{Config.SHAPE}_{current_date_time}.xlsx')) as writer:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            sheet_name = csv_file.split('/')[-1][:-4]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    # shutil.rmtree(os.path.join(directory))


def read_timelines(path, fid='*'):
    json_files = sorted(glob.glob(f"{path}/timeline/timeline_{fid}.json"))
    timelines = []

    for jf in json_files:
        with open(jf) as f:
            timelines.append(json.load(f))

    start_time = min([tl[0][0] for tl in timelines if len(tl)])

    merged_timeline = merge_timelines(timelines)

    return {
        "start_time": start_time,
        "timeline": merged_timeline,
    }


def gen_sliding_window_chart_data(timeline, start_time, value_fn, sw=0.01, debug=False):  # 0.01
    xs = [0]
    hd = [-1]
    cd = [-1]
    current_points = {}
    current_swarms = {}
    gtl_points = {}
    total_failures = 0

    # for event in timeline:
    #     e_type = event[1]
    #     t = event[0] - start_time
    #     if t > 200:
    #         break
    #     if e_type == TimelineEvents.FAIL:
    #         total_failures += 1
    # print(total_failures)
    # return
    i = 0
    # skip = [12, 980, 94, 761, 129, 849, 1050, 1043, 236, 663, 959, 52, 860, 727, 500, 1170, 940, 878, 1028, 981, 532, 572, 140, 926, 892, 893, 1123, 916, 1250, 740]
    while i < len(timeline):
        event = timeline[i]
        e_type = event[1]
        e_fid = event[-1]


        # if e_fid in skip:
        #     continue


        t = event[0] - start_time

        if t > 300:
            break
        if xs[-1] <= t < xs[-1] + sw:
            if e_type == TimelineEvents.COORDINATE:
                current_points[e_fid] = event[2]
            elif e_type == TimelineEvents.FAIL:
                current_points.pop(e_fid)
                gtl_points.pop(e_fid)
            elif e_type == TimelineEvents.ILLUMINATE:
                gtl_points[e_fid] = event[2]
            i += 1
        else:
            # swarm_ys[-1] = len(set(current_swarms.values()))
            # print(len(current_swarms))
            if len(current_points) > 1 and len(gtl_points):
                a = np.stack(list(current_points.values()))
                b = np.stack(list(gtl_points.values()))
                # if t>30:
                if debug:
                    if t > 50:
                        # iss = hausdorff_distance_optimized(a, b, debug=debug)
                        hd = hausdorff_distance_optimized(a, b)
                        print(hd)
                        np.savetxt(f"figs/hd_palm_isr_g50.txt", a)
                        # print(np.array(list(current_points.keys()))[iss].tolist())
                        # print(np.array(list(gtl_points.keys()))[iss])
                        return
                else:
                    hd[-1] = hausdorff_distance_optimized(a, b)
                    cd[-1] = chamfer_distance_optimized(a, b)
                # ys[-1] = 1
            xs.append(xs[-1] + sw)
            hd.append(-1)
            cd.append(-1)

    return xs, hd, cd


def count_close_pairs(points, threshold):
    """Counts the number of pairs of points in a 3D array that are closer than a threshold.

    Args:
        points: A numpy array of shape (N, 3) where each row represents a 3D point.
        threshold: The maximum distance between two points to be considered close.

    Returns:
        An integer count of the number of close pairs.
    """

    distances_squared = cdist(points, points, metric='sqeuclidean')
    close_pairs_mask = distances_squared < threshold**2
    np.fill_diagonal(close_pairs_mask, False)  # Exclude self-pairs
    return int(close_pairs_mask.sum()) // 2  # Divide by 2 since each pair is counted twice


def count_collisions(path, sw=0.01, debug=False):
    data = read_timelines(path, "*")
    timeline = data['timeline']
    start_time = data['start_time']
    # print(len(timeline))
    # exit()
    # 0.01
    n = 1372
    xs = [0]
    coord_timeline = [{}]
    updates = {}

    # waypoints = np.zeros((n, 1000, 3))

    for k in range(n):
        coord_timeline[-1][k] = None
        updates[k] = []


    current_points = {}
    gtl_points = {}

    i = 0
    while i < len(timeline):
        # print(i)
        event = timeline[i]
        e_type = event[1]
        e_fid = event[-1]

        if e_fid > n - 1:
            i += 1
            continue

        t = event[0] - start_time

        j = len(xs)-1

        # if j >= 100:
        #     break

        # print(j, xs[-1], t, xs[-1] + sw)

        if xs[-1] <= t < xs[-1] + sw:
            if e_type == TimelineEvents.COORDINATE:
                # current_points[e_fid] = event[2]
                # waypoints[e_fid, j] = np.array(event[2])
                coord_timeline[j][e_fid] = np.array(event[2])
                updates[e_fid].append(j)
            elif e_type == TimelineEvents.FAIL:
                current_points.pop(e_fid)
                gtl_points.pop(e_fid)
            elif e_type == TimelineEvents.ILLUMINATE:
                gtl_points[e_fid] = event[2]
            i += 1
        else:
            xs.append(xs[-1] + sw)
            coord_timeline.append({})
            for k in range(n):
                coord_timeline[-1][k] = coord_timeline[-2][k]

    # interpolate
    for fid in range(n):
        for j in range(len(updates[fid])-1):
            u = updates[fid][j]
            v = updates[fid][j+1]
            vector = coord_timeline[v][fid] - coord_timeline[u][fid]
            # vector = waypoints[u][fid] - waypoints[v][fid]
            if v - u == 0:
                continue
            dv = vector / (v - u)
            for k in range(u+1, v):
                coord_timeline[k][fid] = coord_timeline[k-1][fid] + dv
                # waypoints[fid][k] = waypoints[fid][k-1] + dv

    # count collisions
    collisions = []
    for i in range(len(xs)):
        points = np.array(list(filter(lambda j: j is not None, coord_timeline[i].values())))
        # points = waypoints[:,i,:]
        # points = points[~np.all(points == [.0, .0, .0], axis=1)]
        # points = points[~np.all(np.abs(points) < 10e-6, axis=1)]
        # if i == 1000:
        #     fig = plt.figure(figsize=(5, 5))
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
        #     ax.set_aspect('equal')
        #     plt.show()
        # plt.close()
        # print(len(points))
        if len(points):
            collisions.append(count_close_pairs(points, 3))
        else:
            collisions.append(0)

    with open(f"{path}/collisions.json", "w") as f:
        json.dump({"t": xs, "collisions": collisions}, f)
    plt.plot(xs, collisions)
    plt.savefig(f"{path}/collisions.png", dpi=300)
    plt.show()
    plt.close()


def merge_timelines(timelines):
    lists = timelines
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heap.append((lst[0][0], i, 0))
    heapq.heapify(heap)

    merged = []
    while heap:
        val, lst_idx, elem_idx = heapq.heappop(heap)
        merged.append(lists[lst_idx][elem_idx] + [lst_idx])
        if elem_idx + 1 < len(lists[lst_idx]):
            next_elem = lists[lst_idx][elem_idx + 1][0]
            heapq.heappush(heap, (next_elem, lst_idx, elem_idx + 1))
    return merged


def gen_sw_charts(path, fid, name, read_from_file=True, debug=False):
    # fig = plt.figure(figsize=(5, 2.5))
    # ax = fig.add_subplot()

    if read_from_file:
        with open(f"{path}/charts.json") as f:
            chart_data = json.load(f)
            # r_xs = chart_data[0]
            # t_idx = next(i for i, v in enumerate(r_xs) if v > 300)
            t = chart_data['t']
            hd = chart_data['hd']
            cd = chart_data['cd']
    else:
        data = read_timelines(path, fid)
        if debug:
            gen_sliding_window_chart_data(data['timeline'], data['start_time'], lambda x: x[2], debug=debug)
            return
        else:
            t, hd, cd = gen_sliding_window_chart_data(data['timeline'], data['start_time'], lambda x: x[2])

        with open(f"{path}/charts.json", "w") as f:
            json.dump({'t': t, 'hd': hd, 'cd': cd}, f)

    # s_xs, s_ys = gen_sliding_window_chart_data(data['sent_bytes'], data['start_time'], lambda x: x[2])
    # h_xs, h_ys = gen_sliding_window_chart_data(data['heuristic'], data['start_time'], lambda x: 1)
    # ax.step(r_xs, s_ys, where='post', label="Number of swarms", color="tab:purple")
    # ax.step(r_xs, l_ys, where='post', label="Number of expired leases")
    while True:
        if hd[0] == -1:
            hd.pop(0)
            cd.pop(0)
            t.pop(0)
        else:
            break

    # ax.step(s_xs, s_ys, where='post', label="Sent bytes", color="black")
    # ax.step(h_xs, h_ys, where='post', label="Heuristic invoked")
    # ax.legend()
    # ax.legend()
    # ax.set_ylabel('Number of swarms', loc='top', rotation=0, labelpad=-90)
    # ax.set_xlabel('Time (Second)', loc='right')
    # ax.spines['top'].set_color('white')
    # ax.spines['right'].set_color('white')
    # plt.xlim([0, 60])
    # plt.show()
    # plt.savefig(f'{path}/{name}_{fid}.png', dpi=300)

    fig, ax = plt.subplots(figsize=(5, 2.5), layout="constrained")
    ax.step(t, hd, where='post', label="Hausdorff distance", color="tab:cyan")
    ax.step(t, cd, where='post', label="Chamfer distance", color="tab:orange")
    ax.text(30, 10, f"HD: {hd[-50]*10:.2f}mm, CD: {cd[-50]*10:.2f}mm")
    ax.legend()
    # ax.set_ylabel(f'HD, {name}', loc='top', rotation=0, labelpad=-133)
    ax.set_title(f'HD, CD {name}', fontsize=10, loc="left")
    ax.set_xlabel('Time (Second)', loc='right')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    # y_locator = ticker.FixedLocator(list(range(0, int(max_y), 10)) + [math.floor(max_y)])

    ax.set_xlim(0, 60)
    # plt.tight_layout()
    # plt.ylim([10e-13, 10e3])
    plt.yscale('log')
    plt.savefig(f'{path}/{name}_{fid}h.png', dpi=300)
    plt.close()


def gen_util_chart(path):
    fig = plt.figure()
    ax = fig.add_subplot()

    with open(f"{path}/utilization.json") as f:
        chart_data = json.load(f)
        t = chart_data[0]
        ys = chart_data[1]

    for i in range(1):
        ax.step(t, [y[i] for y in ys], where='post', label=f"server-{i+1}")

    ax.legend()

    # plt.show()
    plt.savefig(f'{path}/cpu_utilization.png', dpi=300)


def gen_shape_comp_hd(paths, labels, poses, colors, dest):
    lss = ['solid', 'dashdot', 'dashed', 'dotted']
    lss += lss
    fig = plt.figure(figsize=(5, 2.4))
    ax = fig.add_subplot()
    # ax2 = fig.add_axes([0.57, 0.48, 0.38, 0.42])
    ax2 = fig.add_axes([0.57, 0.58, 0.38, 0.32])
    max_y = 80
    t_e2s = []
    for path, label, pos, color, ls in zip(paths, labels, poses, colors, lss):
        with open(f"{path}/charts.json") as f:
            chart_data = json.load(f)
            t = chart_data['t']
            ys = chart_data['hd']
            t_100 = 0
            t_e2 = 0
            while True:
                t_100 += 1
                if t[t_100] >= 100:
                    break

            while True:
                if ys[0] == -1:
                    ys.pop(0)
                    t.pop(0)
                else:
                    break
            while True:
                t_e2 += 1
                if ys[t_e2] < 1e-2:
                    break
            ax.plot(t, ys, linestyle=ls, linewidth=1.4, color=color, label=label)
            ax2.plot(t, ys, linestyle=ls, linewidth=1.4, color=color, label=label)
            max_y = max(max_y, max(ys[:t_100]))
            if ys[t_e2] < 1e-2 and ys[t_e2] != -1:
                print(ys[t_e2])
                t_e2s.append(t[t_e2])
            else:
                t_e2s.append(-1)
            # plt.text(pos[0], pos[1], label, color=color, fontweight='bold')

    ax.set_ylabel('Hausdorff distance (Display cell)', loc='top', rotation=0, labelpad=-133)
    ax.set_xlabel('Time (Second)', loc='right')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_ylim(0, max_y + 10)
    # y_locator = ticker.FixedLocator(list(range(0, int(max_y), 10)) + [math.floor(max_y)])
    y_locator = ticker.FixedLocator(list(range(0, int(max_y), 10)))
    ax.yaxis.set_major_locator(y_locator)
    ax.set_xlim(0, 200)
    plt.tight_layout()


    # plt.yscale('log')
    # plt.ylim([1e-3, 149])
    # plt.xlim([0, 100])
    # y_locator = ticker.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100])
    # ax.yaxis.set_major_locator(y_locator)
    # y_formatter = ticker.FixedFormatter(["0.001", "0.01", "0.1", "1", "10", "100"])
    # ax.yaxis.set_major_formatter(y_formatter)

    # ax2.legend()
    ax2.set_ylabel('Log scale', loc='top', rotation=0, labelpad=-50)
    # ax2.set_xlabel('Time (Second)', loc='right')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    # plt.tight_layout()
    ax2.set_yscale('log')
    ax2.set_ylim([1e-3, 200])
    ax2.set_xlim([0, 100])
    y_locator = ticker.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100])
    # y_locator = ticker.FixedLocator([1e-2, 1e-1, 1, 10, 100])
    ax2.yaxis.set_major_locator(y_locator)
    y_formatter = ticker.FixedFormatter(["0.001", "0.01", "0.1", "1", "10", "100"])
    # y_formatter = ticker.FixedFormatter(["0.01", "0.1", "1", "10", "100"])
    ax2.yaxis.set_major_formatter(y_formatter)
    ax2.yaxis.grid(True, which='minor')
    # ax2.legend(loc="upper right", fontsize="small")
    ax.legend(loc="upper left", fontsize="small", bbox_to_anchor=(0.05, .88))
    # ax.legend(loc="upper left", fontsize="small", bbox_to_anchor=(0.08, .8))

    plt.savefig(dest, dpi=300)
    return t_e2s


def gen_shape_comp_hd_2(paths, labels, colors, dest, ylim=100):
    lss = ['solid', 'dashdot', 'dashed', 'dotted']
    fig = plt.figure(figsize=(5, 2.4))
    ax = fig.add_subplot()
    # ax2 = fig.add_axes([0.57, 0.48, 0.38, 0.42])
    for path, label, color in zip(paths, labels, colors):
        with open(f"{path}/charts.json") as f:
            chart_data = json.load(f)
            t = chart_data[0]
            ys = chart_data[1]
            while True:
                if ys[0] == -1:
                    ys.pop(0)
                    t.pop(0)
                else:
                    break
            ax.plot(t, ys, linewidth=1.4, color=color, label=label)
            # ax2.plot(t, ys, linewidth=1.4, color=color, label=label)
            # plt.text(pos[0], pos[1], label, color=color, fontweight='bold')

    ax.set_ylabel('Hausdorff distance (Display cell)', loc='top', rotation=0, labelpad=-133)
    ax.set_xlabel('Time (Second)', loc='right')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    # ax.set_yscale('log')
    # ax.set_ylim([1e-3, 200])
    ax.set_xlim([0, ylim])
    plt.tight_layout()
    y_locator = ticker.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.yaxis.set_major_locator(y_locator)
    y_formatter = ticker.FixedFormatter(["0.001", "0.01", "0.1", "1", "10", "100"])
    ax.yaxis.set_major_formatter(y_formatter)


    # plt.yscale('log')
    # plt.ylim([1e-3, 149])
    # plt.xlim([0, 100])
    # y_locator = ticker.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100])
    # ax.yaxis.set_major_locator(y_locator)
    # y_formatter = ticker.FixedFormatter(["0.001", "0.01", "0.1", "1", "10", "100"])
    # ax.yaxis.set_major_formatter(y_formatter)

    # ax2.legend()
    # ax2.set_ylabel('Log scale', loc='top', rotation=0, labelpad=-50)
    # # ax2.set_xlabel('Time (Second)', loc='right')
    # ax2.spines['top'].set_color('white')
    # ax2.spines['right'].set_color('white')
    # # plt.tight_layout()
    # ax2.set_yscale('log')
    # ax2.set_ylim([1e-3, 200])
    # ax2.set_xlim([0, 50])
    # y_locator = ticker.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100])
    # ax2.yaxis.set_major_locator(y_locator)
    # y_formatter = ticker.FixedFormatter(["0.001", "0.01", "0.1", "1", "10", "100"])
    # ax2.yaxis.set_major_formatter(y_formatter)
    # ax2.yaxis.grid(True, which='minor')
    # ax2.legend(loc="upper right", fontsize="small")
    ax.legend(loc="upper right", fontsize="small")

    plt.savefig(dest, dpi=300)


def find_time_by_hd(path):
    print(path)
    with open(f"{path}/charts.json") as f:
        chart_data = json.load(f)
        t = chart_data[0]
        ys = chart_data[1]

    for i, y in enumerate(ys):
        if 0.9 < y <= 10:
            print(i, t[i], y)
        elif 0.09 < y <= 0.15:
            print(i, t[i], y)
        elif 0.009 < y <= 0.015:
            print(i, t[i], y)
        elif 0.0009 < y <= 0.0019:
            print(i, t[i], y)


def find_time_to_reach_hd(path, hd=1e-2):
    with open(f"{path}/charts.json") as f:
        chart_data = json.load(f)
        t = chart_data[0]
        ys = chart_data[1]

    t_i = 0
    while True:
        t_i += 1
        if t_i >= len(ys):
            break
        if ys[t_i] != -1 and ys[t_i] < hd:
            return t[t_i], ys[t_i]

    return -1, -1


def gen_shape_fig_by_time(path, target, sw=0.01):
    data = read_timelines(path, "*")
    timeline = data['timeline']
    start_time = data['start_time']
    xs = [0]
    ys = [-1]
    current_points = {}
    gtl_points = {}

    i = 0
    while i < len(timeline):
        event = timeline[i]
        e_type = event[1]
        e_fid = event[-1]
        t = event[0] - start_time
        # if t < 15.65:
        #     timeline.pop(0)
        #     continue
        # print(t)
        if len(xs) == target - 1:
            break
        if xs[-1] <= t < xs[-1] + sw:
            if e_type == TimelineEvents.COORDINATE:
                current_points[e_fid] = event[2]
            elif e_type == TimelineEvents.FAIL:
                current_points.pop(e_fid)
                gtl_points.pop(e_fid)
            elif e_type == TimelineEvents.ILLUMINATE:
                gtl_points[e_fid] = event[2]
            # elif e_type == TimelineEvents.SWARM:
            #     current_swarms[e_fid] = event[2]
            # elif e_type == TimelineEvents.LEASE_EXP:
            #     lease_exp_ys[-1] += 1
            i += 1
        else:
            pass
            # swarm_ys[-1] = len(set(current_swarms.values()))
            # print(len(current_swarms))
            # if len(current_points) > 1 and len(gtl_points):
            #     ys[-1] = hausdorff_distance(np.stack(list(current_points.values())),
            #                                 np.stack(list(gtl_points.values())))
                # ys[-1] = 1
            xs.append(xs[-1] + sw)
            # ys.append(-1)
            # swarm_ys.append(-1)
            # lease_exp_ys.append(0)
    # print(hausdorff_distance(np.stack(list(current_points.values())), np.stack(list(gtl_points.values()))))
    ptcld = np.stack(list(current_points.values()))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ptcld[:, 0], ptcld[:, 1], ptcld[:, 2])
    # plt.show()
    return ptcld, round(hausdorff_distance(ptcld, np.stack(list(gtl_points.values()))), 4)


def quad(ptlds, hds, dest):
    # pylab.rcParams['xtick.major.pad'] = '1'
    # pylab.rcParams['ytick.major.pad'] = '1'
    # pylab.rcParams['ztick.major.pad'] = '8'
    # title_offset = [-.25, -.25, -.02, -.02]
    shapes_labels = [f'HD={hd}' for hd in hds]
    fig = plt.figure(figsize=(5, 4), layout='constrained')

    for i, ptcld in enumerate(ptlds):
        # mat = scipy.io.loadmat(f'../assets/{shapes[i]}.mat')
        # ptcld = mat['p']

        ticks_gap = 10
        length = math.ceil(np.max(ptcld[:, 0]) / ticks_gap) * ticks_gap
        width = math.ceil(np.max(ptcld[:, 1]) / ticks_gap) * ticks_gap
        height = math.ceil(np.max(ptcld[:, 2]) / ticks_gap) * ticks_gap
        ax = fig.add_subplot(2, 2, i + 1, projection='3d', proj_type='ortho')
        ax.scatter(ptcld[:, 0], ptcld[:, 1], ptcld[:, 2], c='blue', s=1, alpha=1, edgecolors='none')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((0, 0, 0, 0.025))
        # ax.view_init(elev=16, azim=137, roll=0)
        ax.view_init(elev=16, azim=-120, roll=0)
        ax.axes.set_xlim3d(left=1, right=length)
        ax.axes.set_ylim3d(bottom=1, top=width)
        ax.axes.set_zlim3d(bottom=1, top=height)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(length))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(width))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(height))
        ax.set_aspect('equal')
        ax.grid(False)
        # ax.set_xticks(range(0, length + 1, length))
        # ax.set_yticks(range(0, width + 1, width))
        # ax.set_zticks(range(0, height + 1, height))
        ax.tick_params(pad=-2)
        ax.tick_params(axis='x', pad=-6)
        ax.tick_params(axis='y', pad=-6)
        ax.set_title(shapes_labels[i], y=-.01)

    plt.margins(x=0, y=0)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(dest, dpi=300)


def get_table_vals(st_path):
    df = pd.read_csv(f"{st_path}/metrics.csv")
    return {
        # "Avg Xmit Bytes": df['A4_bytes_sent'].mean(),
        "Total Xmit bytes": format(df['A4_bytes_sent'].sum(), ','),
        # "Avg Xmit Bytes": df['A4_bytes_sent'].mean(),
        "Total # Localizations": format(df['A3_num_localize'].sum(), ','),
        "Total # Anchors": format(df['A3_num_anchor'].sum(), ','),
        "Total # times swarms thawed": format(df['C0_num_received_THAW_SWARM'].sum(), ','),
        "Total # times leases expired": f"{df['A2_num_expired_leases'].sum()} ({round(df['A2_num_expired_leases'].sum() / df['A2_num_granted_leases'].sum() * 100, 2)}\%)",
        "Avg distance traveled": round(df['A0_total_distance'].mean(), 2),
    }


def report_distance_traveled(st_path):
    json_dir = os.path.join(st_path, 'json')
    filenames = os.listdir(json_dir)
    filenames.sort()

    total_dist = 0
    count = 0
    dists = []
    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(json_dir, filename)) as f:
                try:
                    data = json.load(f)
                    total_dist += data['A0_total_distance']
                    dists.append(data['A0_total_distance'])
                    count += 1
                except json.decoder.JSONDecodeError:
                    print(filename)
    dists = np.array(dists)*2.5
    print(f"min:{np.min(dists)}, max:{np.max(dists)}, avg:{np.mean(dists)}")
    return total_dist / count

