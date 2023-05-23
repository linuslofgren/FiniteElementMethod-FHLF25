import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np
from clamping_problem import ClampingProblem

def vis_mesh(problem: ClampingProblem):
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    cfv.draw_mesh(problem.coords, problem.edof, 1, 2)
    cfv.show_and_wait()

def vis_temp(a_stat, problem: ClampingProblem):
    """Visualizes the static temperature distribution"""

    print(f"Maximum temperature {np.amax(a_stat):.2f} (°C)")

    cfv.figure(fig_size=(10, 10))
    # Mirror in the x and y plane
    draw_nodal_values_shaded(a_stat, problem.coords, problem.edof,
                             dofs_per_node=1, el_type=2, draw_elements=True)
    draw_nodal_values_shaded(a_stat, [0, problem.L]+[1, -1]*problem.coords, problem.edof,
                             dofs_per_node=1, el_type=2, draw_elements=True)
    draw_nodal_values_shaded(a_stat, [2*problem.L, problem.L]+[-1, -1]*problem.coords, problem.edof,
                             dofs_per_node=1, el_type=2, draw_elements=True)
    draw_nodal_values_shaded(a_stat, [2*problem.L, 0]+[-1, 1]*problem.coords, problem.edof,
                             dofs_per_node=1, el_type=2, draw_elements=True)
    draw_nodal_values_shaded(a_stat, [2*problem.L, 0]+[-1, 1]*problem.coords, problem.edof, title=(f"Maximum temperature {np.amax(a_stat):.2f} °C"),
                             dofs_per_node=1, el_type=2, draw_elements=True)

    cfv.colorbar()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    # Set the colorbar label
    for c in cfv.gca().collections:
        if c.colorbar:
            c.colorbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
            cfv.show_and_wait()


def vis_transient(snapshots, snapshot_time, problem: ClampingProblem):
    """Visualizes the transient heat distribution by showing five snapshots from 0 to 3% progress
    snapshots: heat distributions from 0 to 90% of maximum static temperature
    snapshot_time: timestamp for each snapshot
    """

    total_time = snapshot_time[-1]
    print(f"total time to 90% {total_time:.2f} seconds")

    # Select and show five snapshots equally spaced from 0 to 3% of total time
    time_90_perc = total_time
    time_3_perc = 0.03 * time_90_perc
    time_step = (time_3_perc / 4)

    total_time = 0
    time_step_index = 0

    vis_snapshots = []
    vis_snapshot_time = []
    for snapshot, timestamp in zip(snapshots, snapshot_time):
        if timestamp >= time_step_index * time_step:
            time_step_index += 1
            vis_snapshots.append(snapshot)
            vis_snapshot_time.append(timestamp)

        if time_step_index == 5:
            break

    fig, axes = plt.subplots(3, 2)
    fig.tight_layout()
    axes.flatten()[-1].axis('off')

    vmax = np.max(vis_snapshots[-1])
    for snapshot, time, ax in zip(vis_snapshots, vis_snapshot_time, axes.flatten()):
        plt.sca(ax)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Mirror in x and y plane
        draw_nodal_values_shaded(snapshot, problem.coords, problem.edof,
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=problem.T_0, vmax=vmax)
        draw_nodal_values_shaded(snapshot, [0, problem.L]+[1, -1]*problem.coords, problem.edof,
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=problem.T_0, vmax=vmax)
        draw_nodal_values_shaded(snapshot, [2*problem.L, problem.L]+[-1, -1]*problem.coords, problem.edof,
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=problem.T_0, vmax=vmax)
        draw_nodal_values_shaded(snapshot, [2*problem.L, 0]+[-1, 1]*problem.coords, problem.edof, title=(f"t={time:.2f}s, max temp {np.amax(snapshot):.2f} °C"),
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=problem.T_0, vmax=vmax)

    fig.subplots_adjust(right=0.8)
    plt.colorbar(ax=axes.ravel().tolist())

    for c in cfv.gca().collections:
        if c.colorbar:
            c.colorbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    cfv.show_and_wait()


def vis_displacement(von_mises_node, a, problem: ClampingProblem):
    magnification = 10.0
    cfv.figure(fig_size=(10, 10))

    flip_y = np.array([([1, -1]*int(a.size/2))]).T
    flip_x = np.array([([-1, 1]*int(a.size/2))]).T

    coords_list = [problem.coords, [0, problem.L]+[1, -1]*problem.coords, [2*problem.L,
                                                                           problem.L]+[-1, -1]*problem.coords, [2*problem.L, 0]+[-1, 1]*problem.coords]
    displacement_list = [a, np.multiply(flip_y, a), np.multiply(
        flip_y*flip_x, a), np.multiply(flip_x, a)]
    for coords, displacements in zip(coords_list, displacement_list):
        if displacements is not None:
            if displacements.shape[1] != coords.shape[1]:
                displacements = np.reshape(
                    displacements, (-1, coords.shape[1]))
                coords_disp = np.asarray(
                    coords + magnification * displacements)
        cfv.draw_mesh(coords, problem.edof, 1,
                      problem.el_type, color=(0, 0, 0, 0.1))
        draw_nodal_values_shaded(von_mises_node, coords_disp, problem.edof, title=(f"Maximum von Mises stress {np.amax(von_mises_node):.1E} [Pa]"),
                                 dofs_per_node=1, el_type=2, draw_elements=False)

        cfv.draw_mesh(coords_disp, problem.edof, 1,
                      problem.el_type, color=(0, 1, 0, 0.1))

    cfv.colorbar()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    for c in cfv.gca().collections:
        if c.colorbar:
            c.colorbar.set_label('von Mises stress (Pa)',
                                 rotation=270, labelpad=20)
    cfv.show_and_wait()

# Shim cfv.draw_nodal_values_shaded to allow arguments to be passed down to plt.tripcolor


def draw_nodal_values_shaded(values, coords, edof, title=None, dofs_per_node=None, el_type=None, draw_elements=False, **kwargs):
    """Draws element nodal values as shaded triangles. Element topologies
    supported are triangles, 4-node quads and 8-node quads."""

    edof_tri = cfv.topo_to_tri(edof)

    ax = plt.gca()
    ax.set_aspect('equal')

    x, y = coords.T
    v = np.asarray(values)
    plt.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud", **kwargs)

    if draw_elements:
        if dofs_per_node != None and el_type != None:
            cfv.draw_mesh(coords, edof, dofs_per_node,
                          el_type, color=(0, 1, 0, 0.1))
        else:
            cfv.info(
                "dofs_per_node and el_type must be specified to draw the mesh.")

    if title != None:
        ax.set(title=title)
