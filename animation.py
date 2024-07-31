import itertools
import json
import math
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl

from utils.file import read_timelines
from worker.metrics import TimelineEvents

ticks_gap = 5
S = 4

start_time = 0
plt.style.use('dark_background')

# t30_d1_g0	t30_d1_g20	t30_d5_g0	t30_d5_g20	t600_d1_g0	t600_d1_g20	t600_d5_g0	t600_d5_g20
output_name = "testd"

# COLORS = ['#b083f0', '#fc8dc7', '#f47068', '#e0823e', '#c69027', '#57ab5a']

COLORS = [
    '#d56156',
    '#9e9ab7',
    '#6eb1a6',
    '#dbca4e',
    '#aac8a3',
    '#9a639c',
    '#93bc49',
    '#d99343',
]
gtl_color = '#529bf5'

side_color = '#b083f0'


def set_axis(ax, length, width, height, title=""):
    ax.axes.set_xlim3d(left=0, right=length)
    ax.axes.set_ylim3d(bottom=0, top=width)
    ax.axes.set_zlim3d(bottom=0, top=height)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks(range(0, length + 1, ticks_gap))
    ax.set_yticks(range(0, width + 1, ticks_gap))
    ax.set_zticks(range(0, height + 1, ticks_gap))
    ax.set_title(title, y=.9)
    # ax.set_title(title)


def set_axis_2d(ax, length, width, title):
    ax.axes.set_xlim(0, length)
    ax.axes.set_ylim(0, width)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off')
    ax.set_title(title)


def set_text(tx, t, hd):
    hs_f = "{:.3f}".format(round(hd, 3)) if hd > 0.001 else "{:.2e}".format(hd)
    tx.set(text=f"Elapsed time: {int(t)} (Second)\nHausdorff distance: {hs_f} (cm)")
    # tx.set(text=f"Elapsed time: {int(t)} seconds")


def draw_figure():
    px = 1 / plt.rcParams['figure.dpi']
    fig_width = 1920 * px
    fig_height = 1080 * px
    fig = plt.figure(figsize=(fig_width, fig_height))
    spec = fig.add_gridspec(3, 6, left=0.04, right=0.96, top=0.92, bottom=0.08)
    ax = fig.add_subplot(spec[0:2, 0:3], projection='3d', proj_type='ortho')

    ax1 = fig.add_subplot(spec[0:2, 3:6], projection='3d', proj_type='ortho')

    ax2 = fig.add_subplot(spec[2, 0:2])
    ax3 = fig.add_subplot(spec[2, 2:4])
    ax4 = fig.add_subplot(spec[2, 4:6])

    # ax = fig.add_subplot(2, 1, 1, projection='3d', proj_type='ortho')
    # ax2 = fig.add_subplot(2, 1, 2, projection='3d', proj_type='ortho')
    tx = fig.text(0.05, 0.88, s="", fontsize=16)
    line1 = ax.scatter([], [], [])
    return fig, ax, ax1, ax2, ax3, ax4, tx


def read_point_cloud(path):
    data = read_timelines(path, "*")
    timeline = data['timeline']
    start_time = data['start_time']

    height = 0
    width = 0
    length = 0
    filtered_events = []
    gtl = []
    for e in timeline:
        if e[1] == TimelineEvents.FAIL or e[1] == TimelineEvents.SWARM:
            e[0] -= start_time
            filtered_events.append(e)
        elif e[1] == TimelineEvents.COORDINATE:
            e[0] -= start_time
            filtered_events.append(e)
            length = max(int(e[2][0]), length)
            width = max(int(e[2][1]), width)
            height = max(int(e[2][2]), height)
        elif e[1] == TimelineEvents.ILLUMINATE:
            e[0] -= start_time
            filtered_events.append(e)
            gtl.append(e[2])
    length = math.ceil(length / ticks_gap) * ticks_gap
    width = math.ceil(width / ticks_gap) * ticks_gap
    height = math.ceil(height / ticks_gap) * ticks_gap

    return filtered_events, length, width, height, np.stack(gtl)


def init(ax, ax1, ax2, ax3, ax4):
    ax.xaxis.set_pane_color((.0, .0, .0, 1.0))
    ax.yaxis.set_pane_color((.0, .0, .0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    ax.view_init(elev=14, azim=-136, roll=0)
    ax1.xaxis.set_pane_color((.0, .0, .0, 1.0))
    ax1.yaxis.set_pane_color((.0, .0, .0, 1.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    ax1.view_init(elev=14, azim=-136, roll=0)
    # ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax2.view_init(elev=0, azim=0, roll=0)
    # ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax3.view_init(elev=0, azim=90, roll=0)
    # ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax4.view_init(elev=90, azim=90, roll=0)
    # return line1,


def update(frame):
    t = start_time + frame * frame_rate
    while len(filtered_events):
        # print(t)
        event_time = filtered_events[0][0]
        if event_time <= t:
            event = filtered_events.pop(0)
            event_type = event[1]
            fls_id = event[-1]
            if fls_id > 1371:
                continue
            if event_type == TimelineEvents.COORDINATE:
                points[fls_id] = event[2]
            elif event_type == TimelineEvents.FAIL:
                points.pop(fls_id)
            elif event_type == TimelineEvents.SWARM:
                swarms[fls_id] = event[2]
        else:
            t += frame_rate
            break
    coords = points.values()
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    colors = [COLORS[(swarms[fid]) % len(COLORS)] for fid in points.keys()]
    ax.clear()
    ln = ax.scatter(xs, ys, zs, c=colors, s=S, alpha=1)
    set_axis(ax, length, width, height)

    ax1.clear()
    ln1 = ax1.scatter(gtl[:, 0], gtl[:, 1], gtl[:, 2], c=gtl_color, s=S, alpha=1)
    set_axis(ax1, length, width, height, "Ground Truth")

    ax2.clear()
    if name[0].startswith('skateboard'):
        ln2 = ax2.scatter(ys, xs, c=side_color, s=S, alpha=1)
        set_axis_2d(ax2, width, length, "Top")

    else:
        ln2 = ax2.scatter(xs, ys, c=side_color, s=S, alpha=1)
        set_axis_2d(ax2, length, width, "Top")

    ax3.clear()
    ln3 = ax3.scatter(xs, zs, c=side_color, s=S, alpha=1)
    set_axis_2d(ax3, length, height, "Front")

    ax4.clear()
    ln4 = ax4.scatter(ys, zs, c=side_color, s=S, alpha=1)
    set_axis_2d(ax4, width, height, "Side")

    idx = find_nearest(time_stamps, t)
    hd = hds[idx]
    set_text(tx, t, hd)
    return [ln, ln1, ln2, ln3, ln4]


def show_last_frame(events, t=30):
    movements = dict()
    swarm = dict()
    swarm_size = dict()
    l_points = dict()

    for event in events:
        event_time = event[0]
        if event_time > t:
            break
        event_type = event[1]
        fls_id = event[-1]
        if event_type == TimelineEvents.COORDINATE:
            movements[fls_id] = np.linalg.norm(np.array(event[2]) - np.array(l_points[fls_id]))
            l_points[fls_id] = event[2]
            if movements[fls_id] > 5:
                print(event_time, fls_id, swarm[fls_id], movements[fls_id], l_points[fls_id])
        elif event_type == TimelineEvents.FAIL:
            l_points.pop(fls_id)
        elif event_type == TimelineEvents.ILLUMINATE:
            l_points[fls_id] = event[2]
        elif event_type == TimelineEvents.SWARM:
            swarm[fls_id] = event[2]
            if event[2] in swarm_size:
                swarm_size[event[2]] += 1
            else:
                swarm_size[event[2]] = 1
        # else:
        #     points.pop(fls_id)
    print(swarm_size)
    coords = l_points.values()
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    return xs, ys, zs


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == '__main__':
    duration = 20
    fps = 10
    frame_rate = 1 / fps

    for r, p in [
        (
            "root",
            "path"
        )
    ]:
        names = p.split("_")
        name = names[0]
        # name = 'skateboard'
        n_points = names[1]
        # n_points = 1372
        path = os.path.join(r, p)
        scale = 0.02
        if name == 'chess':
            scale = 0.4
        print(p)

        filtered_events, length, width, height, _ = read_point_cloud(path)

        # gtl = np.loadtxt(f'assets/{name}.xyz', delimiter=' ') * 100 * scale
        # gtl = np.loadtxt(f'assets/{name}_{n_points}.xyz', delimiter=' ')*100*scale
        # gtl[:, [1, 2, 0]] = gtl[:, [0, 1, 2]]
        # gtl = np.loadtxt(f'assets/{name}_{n_points}.txt', delimiter=',')
        gtl = np.array([[0, 0, 0]])

        with open(f"{path}/charts.json") as f:
            chart_data = json.load(f)
            time_stamps = chart_data['t']
            hds = chart_data['hd']
            while True:
                if hds[0] == -1:
                    hds.pop(0)
                    time_stamps.pop(0)
                else:
                    break
        fig, ax, ax1, ax2, ax3, ax4, tx = draw_figure()
        points = dict()
        swarms = dict()
        ani = FuncAnimation(
            fig, partial(update, ),
            frames=fps * duration,
            init_func=partial(init, ax, ax1, ax2, ax3, ax4))
        #
        # plt.show()
        writer = FFMpegWriter(fps=fps)
        ani.save(f"{path}/{p}.mp4", writer=writer)
