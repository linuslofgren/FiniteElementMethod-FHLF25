import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import calfem.core as cfc
import numpy as np

# Contitutional parameters
k_cu = 385  # [W/(mK)]
D_cu = np.diag([k_cu, k_cu])

k_ny = 0.26  # [W/(mK)]
D_ny = np.diag([k_ny, k_ny])

T_inf = 18  # [C]
h = -1e5  # [W/m**2]


thickness = [0.001] # [m]
t = thickness[0]

alpha_c = 40 # [W/(m^2K)]
# Definie Geometry

# Mesh markers
HEAT_FLUX_BOUNDARY = 6
CONVECTION_BOUNDARY = 8

COPPER_SURFACE = 1
NYLON_SURFACE = 2

def define_geometry():
    L = 0.005 # [m]
    a = 0.1*L
    b = 0.1*L
    c = 0.3*L
    d = 0.05*L
    h = 0.15*L
    t = 0.05*L

    g = cfg.geometry()

    points = [[0, 0.5*L], [a, 0.5*L], [a, 0.5*L-b], [a+c, 0.5*L-b], [a+c+d, 0.5*L-b-d],  # 0-4
              [a+c+d, d], [L-2*d, 0.3*L], [L, 0.3*L], [L,
                                                       0.3*L-d], [L-2*d, 0.3*L-d],  # 5-9
              [a+c+d, 0], [c+d, 0], [c+d, 0.5*L-b-a], [a+t,
                                                       0.5*L-b-a], [a+t, 0.5*L-b-a-h],  # 10-14
              [a, 0.5*L-b-a-h], [a, 0.5*L-b-a], [0, 0.5*L-b-a], [0, 0.5*L-b], [0, 0]]  # 15-19

    for xp, yp in points:
        g.point([xp, yp])


    for s in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],  # 0-4
               [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]:
        g.spline(s, marker=CONVECTION_BOUNDARY)
    for s in [ # 5-9
               [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],  # 10-14
               [15, 16], [16, 17], [17, 18]]:
        g.spline(s)

    for s in  [[18, 0]]:
        g.spline(s, marker=HEAT_FLUX_BOUNDARY)

    for s in [[17, 19], [19, 11]]:
        g.spline(s)

    g.surface(list(range(0, 19)), marker=COPPER_SURFACE)
    g.surface([11, 12, 13, 14, 15, 16, 19, 20], marker=NYLON_SURFACE)

    return g


def generate_mesh(g, dof=1):
    mesh = cfm.GmshMesh(g, 2, dof)
    mesh.return_boundary_elements = True
    return mesh.create()


g = define_geometry()
# cfv.draw_geometry(g)


coords, edof, dofs, bdofs, element_markers, boundary_elements = generate_mesh(g)
ex, ey = cfc.coord_extract(edof, coords, dofs)

K = np.zeros((np.size(dofs), np.size(dofs)))
for eldof, elx, ely, material_index in zip(edof, ex, ey, element_markers):
    Ke = cfc.flw2te(elx, ely, thickness, D_cu if material_index == COPPER_SURFACE else D_ny)
    cfc.assem(eldof, K, Ke)



f_h = np.zeros([np.size(dofs), 1])

def N_transpose(node_pair_list, f_sub, factor):
    for node_pair in node_pair_list:
        r1 = None
        r1_dof_index = None
        r2 = None
        r2_dof_index = None
        for i, (coord, dof) in enumerate(zip(coords, dofs)):
            if dof == node_pair[0]:
                r1 = coord
                r1_dof_index = i
            if dof == node_pair[1]:
                r2 = coord
                r2_dof_index = i
        
        distance = np.linalg.norm(np.array(r1) - np.array(r2))
        print(h)
        f_sub[r1_dof_index] += factor * distance
        f_sub[r2_dof_index] += factor * distance

def N_N_transpose(node_pair_list, k_sub):
    for node_pair in node_pair_list:
        r1 = None
        r1_dof_index = None
        r2 = None
        r2_dof_index = None
        for i, (coord, dof) in enumerate(zip(coords, dofs)):
            if dof == node_pair[0]:
                r1 = coord
                r1_dof_index = i
            if dof == node_pair[1]:
                r2 = coord
                r2_dof_index = i
        
        distance = np.linalg.norm(np.array(r1) - np.array(r2))
        print(distance)
        k_sub[r1_dof_index][r1_dof_index] += 1/3 * alpha_c * t * distance
        k_sub[r2_dof_index][r2_dof_index] += 1/3 * alpha_c * t * distance
        k_sub[r1_dof_index][r2_dof_index] += 1/6 * alpha_c * t * distance
        k_sub[r2_dof_index][r1_dof_index] += 1/6 * alpha_c * t * distance



node_lists = []
for b in boundary_elements[HEAT_FLUX_BOUNDARY]:
    node_lists.append(b.get("node-number-list"))
N_transpose(node_lists, f_h, -h*t * 1/2)
print(f_h)
f_c = np.zeros([np.size(dofs), 1])

node_lists_conv = []
for b in boundary_elements[CONVECTION_BOUNDARY]:
    node_lists_conv.append(b.get("node-number-list"))
N_transpose(node_lists_conv, f_c, T_inf * alpha_c * t)
# print(f_c)

K_sub = np.zeros((np.size(dofs), np.size(dofs)))
N_N_transpose(node_lists_conv, K_sub)
cfv.figure(fig_size=(10, 10))
# cfv.plt.spy(K_sub)
K += K_sub

bc = np.array([], 'i')
bc_val = np.array([], 'i')

# cfv.draw_mesh(coords, edof, 1, 2)

f = f_h + f_c

a_stat = np.linalg.solve(K, f)

max_temp=np.amax(a_stat)

print(a_stat.shape)
# cfv.figure(fig_size=(10, 10))
# cfv.draw_nodal_values_shaded(a, coords, edof, title="Temperature",
#                             dofs_per_node=1, el_type=2, draw_elements=True)

# MARK: Del B let's go!!!

from plantml import plantml

rho_cu = 8930 # [kg/m^3]
rho_ny = 1100 # [kg/m^3]

c_p_cu = 386 # [J/(kgK)]
c_p_ny = 1500 # [J/(kgK)]

C = np.zeros((np.size(dofs), np.size(dofs)))
for eldof, elx, ely, material_index in zip(edof, ex, ey, element_markers):
    Ce = plantml(elx, ely, (rho_cu * c_p_cu) if material_index == COPPER_SURFACE else (rho_ny * c_p_ny))
    cfc.assem(eldof, C, Ce)


theta = 1.0
delta_t = 1

a = np.full(dofs.shape, 18)
print(a.shape)
from time import sleep
import matplotlib.pyplot as plt

# cfv.figure(fig_size=(10, 10))
# cfv.show()
print("a shape", a.shape)
for _ in np.arange(0, 80000, delta_t):
    # print("No shift, right", ((C-delta_t*K*(1-theta))@a).shape)
    # print("No shift (f)", (delta_t*f).shape)
    # print("No shift (big)", (delta_t*f+(C-delta_t*K*(1-theta))@a).shape)
# f_shift = f.flatten()
# print("With shift (f)", f_shift.shape)
# print("With shift (big)", (delta_t*f_shift+(C-delta_t*K*(1-theta))@a).shape)
    A = C+delta_t*theta*K
    b = delta_t*f+(C-delta_t*K*(1-theta))@a
    a = np.linalg.solve(A, b)
    print("C-delta", ((C-delta_t*K*(1-theta))@a).shape)
    print("f", f.shape)
    print("A", A.shape)
    print("b", b.shape)
    print("a", a.shape)
    # cfv.draw_nodal_values_shaded(a, coords, edof, title="Temperature",
    #                         dofs_per_node=1, el_type=2, draw_elements=True)
    # cfv.colorbar()
    # plt.show(block=False)
    # plt.pause(0.2)
    # plt.close()
    # cfv.gcf().canvas.draw()
    # cfv.canvas.draw()

    # cfv.gcf().canvas.flush_events()
    # sleep(0)
    # plt.close()

print(np.amax(a_stat-a))
# cfv.show_and_wait()
