import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from multiprocessing import shared_memory
import numpy as np


def plot_point_cloud(ptcld):
    mpl.use('macosx')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    graph = ax.scatter(ptcld[:, 0], ptcld[:, 1], ptcld[:, 2])
    # count = ptcld.shape[0]
    # ani = animation.FuncAnimation(fig, update, fargs=[graph, shm_name, count], frames=100, interval=50, blit=True)
    plt.show()


def update(num, graph, shm_name, count):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray((count, 3), dtype=np.float64, buffer=shared_mem.buf)
    # print(graph._offsets3d)
    graph._offsets3d = (shared_array[:, 0], shared_array[:, 1], shared_array[:, 2])
    return graph,

def add_dead_reckoning_error(vector, alpha=0):
    alpha = alpha / 180 * np.pi
    if vector[0] or vector[1]:
        i = np.array([vector[1], -vector[0], 0])
    elif vector[2]:
        i = np.array([vector[2], 0, -vector[0]])
    else:
        return vector

    if alpha == 0:
        return vector

    j = np.cross(vector, i)
    norm_i = np.linalg.norm(i)
    norm_j = np.linalg.norm(j)
    norm_v = np.linalg.norm(vector)
    i = i / norm_i
    j = j / norm_j
    phi = np.random.uniform(0, 2 * np.pi)
    error = np.sin(phi) * i + np.cos(phi) * j
    r = np.linalg.norm(vector) * np.tan(alpha)

    erred_v = vector + np.random.uniform(0, r) * error
    return norm_v * erred_v / np.linalg.norm(erred_v)


def plot_points(shape, A, azim, elev, x=None, y=None, z=None, alpha=0):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    A_e = []
    for v in A:
        A_e.append(add_dead_reckoning_error(v, alpha))

    A_e = np.vstack(A_e)

    ax.scatter3D(A_e[:, 0], A_e[:, 1], A_e[:, 2], color=root_color, depthshade=False, s=1)
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if x:
        ax.axes.set_xlim3d(left=0, right=x)
    if y:
        ax.axes.set_ylim3d(bottom=0, top=y)
    if z:
        ax.axes.set_zlim3d(bottom=0, top=z)
    ax.set_aspect('equal')
    ax.view_init(azim=azim, elev=elev)

    ax.axis('off')
    # plt.show()
    # return
    post_fix = 'gtl' if alpha == 0 else f'alpha{alpha}'
    plt.savefig(f'figs/gtl/{shape}_{post_fix}.png', dpi=300)


if __name__ == '__main__':
    mpl.use('macosx')
    root_color = '#0871f5'

    shapes = ["chess_408", "skateboard_1372", "dragon_1147", "palm_725", "racecar_3720", "kangaroo_972"][3:4]
    scales = [.4, 1, 1, 1, 1, 1][3:4]
    azims = [-110, -150, -150, 20, -20, -45][3:4]
    elevs = [20, 30, 20, 20, 42, 20][3:4]
    kargs = [{"z":100}, {"x": 100}, {"y": 100}, {"z": 100}, {"x": 100}, {"x": 100}][3:4]

    # A = np.loadtxt(f'figs/hd_palm_isr_g50.txt', delimiter=' ')
    # plot_points("palm_725", A, azims[0], elevs[0], **kargs[0])
    # exit()

    for shape, scale, azim, elev, karg in zip(shapes, scales, azims, elevs, kargs):
        A = np.loadtxt(f'../assets/{shape}.xyz', delimiter=' ') * 100 * scale
        if shape != "kangaroo_972":
            A[:, [1, 2, 0]] = A[:, [0, 1, 2]]
        plot_points(shape, A, azim, elev, alpha=2, **karg)


