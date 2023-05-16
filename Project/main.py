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


thickness = [0.005] # [m]
t = thickness[0]

alpha_c = 40 # [W/(m^2K)]
# Definie Geometry

# Mesh markers
HEAT_FLUX_BOUNDARY = 6
CONVECTION_BOUNDARY = 8
FIXED_BOUNDARY = 10
X_FIXED_BOUNDARY = 11
Y_FIXED_BOUNDARY = 12

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

    NUM = None

    for s in [[0, 1]]:
        g.spline(s, marker=Y_FIXED_BOUNDARY, el_on_curve=NUM)

    for s in [[1, 2], [2, 3], [3, 4], [4, 5],  # 0-4
               [5, 6], [6, 7]]:
        g.spline(s, marker=CONVECTION_BOUNDARY, el_on_curve=NUM)

    for s in [[7, 8]]:
        g.spline(s, marker=X_FIXED_BOUNDARY, el_on_curve=NUM)

    for s in [[8, 9], [9, 10]]:
        g.spline(s, marker=CONVECTION_BOUNDARY, el_on_curve=NUM)

    for s in [ # 5-9
               [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],  # 10-14
               [15, 16], [16, 17]]:
        g.spline(s, el_on_curve=NUM)

    for s in [[17, 18]]:
        g.spline(s, marker=FIXED_BOUNDARY, el_on_curve=NUM)

    for s in  [[18, 0]]:
        g.spline(s, marker=HEAT_FLUX_BOUNDARY, el_on_curve=NUM)

    for s in [[17, 19]]:
        g.spline(s, marker=FIXED_BOUNDARY, el_on_curve=NUM)


    for s in [[19, 11]]:
        g.spline(s, el_on_curve=NUM)

    g.surface(list(range(0, 19)), marker=COPPER_SURFACE)
    g.surface([11, 12, 13, 14, 15, 16, 19, 20], marker=NYLON_SURFACE)

    return g


def generate_mesh(g, dof=1):
    mesh = cfm.GmshMesh(g, 2, dof)
    mesh.return_boundary_elements = True
    return mesh.create()


g = define_geometry()
cfv.draw_geometry(g)


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
# cfv.figure(fig_size=(10, 10))
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
    Ce = plantml(elx, ely, (rho_cu * c_p_cu * t) if material_index == COPPER_SURFACE else (rho_ny * c_p_ny * t))
    cfc.assem(eldof, C, Ce)


theta = 1.0
delta_t = 0.05

a = np.full(dofs.shape, 18)
print(a.shape)
from time import sleep
import matplotlib.pyplot as plt

# cfv.figure(fig_size=(10, 10))
# cfv.show()
print("a shape", a.shape)

total_time = 0
time_90_perc = None
for _ in np.arange(0, 100, delta_t):
    A = C+delta_t*theta*K
    b = delta_t*f+(C-delta_t*K*(1-theta))@a
    a = np.linalg.solve(A, b)
    total_time += delta_t
    # cfv.draw_nodal_values_shaded(a, coords, edof, title="Temperature",
    #                         dofs_per_node=1, el_type=2, draw_elements=True)
    # cfv.colorbar()
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.close()
    if np.amax(a) >= np.amax(a_stat) * 0.9:
        print("total time", total_time)
        time_90_perc = total_time
        break
    # print(np.amax(a_stat-a))
print(time_90_perc)
time_3_perc = 0.03 * time_90_perc
time_step = (time_3_perc / 4)


a = np.full(dofs.shape, 18)

total_time = 0
time_90_perc = None
time_step_index = 0

snapshots = []
for _ in np.arange(0, 100, delta_t):
    if total_time >= time_step_index * time_step:
        time_step_index += 1
        snapshots.append(a)
        
    if time_step_index == 5:
        break
    A = C+delta_t*theta*K
    b = delta_t*f+(C-delta_t*K*(1-theta))@a
    a = np.linalg.solve(A, b)
    total_time += delta_t
    
print(len(snapshots))
# for snapshot in snapshots:
#     cfv.figure(fig_size=(10, 10))
#     cfv.draw_nodal_values_shaded(snapshot, coords, edof, title=("Temperature " + str(np.amax(snapshot))),
#                             dofs_per_node=1, el_type=2, draw_elements=True)
#     cfv.colorbar()

# cfv.show_and_wait()



# MARK: Del C

print(edof.shape)
stress_edof = np.full((edof.shape[0], edof.shape[1]*2), 0)
print(stress_edof.shape)
for (a, b, c ), stress_a in zip(edof, stress_edof):
    # print(a,b,c)
    stress_a[0] = 2*a-1
    stress_a[1] = 2*a
    stress_a[2] = 2*b-1
    stress_a[3] = 2*b
    stress_a[4] = 2*c-1
    stress_a[5] = 2*c
# cfc.plants
print(stress_edof.shape)

E_cu = 128 # [GPa]
E_ny = 3 # [GPa]

v_cu = 0.36
v_ny = 0.36


K = np.zeros((np.size(dofs)*2, np.size(dofs)*2))
print(K.shape)
for eldof, elx, ely, material_index in zip(stress_edof, ex, ey, element_markers):
    D = cfc.hooke(4, E_cu if material_index == COPPER_SURFACE else E_ny, v_cu if material_index == COPPER_SURFACE else v_ny)[np.ix_([0, 1, 3], [0, 1, 3])]
    Ce = cfc.plante(elx, ely, [2, t], D)
    cfc.assem(eldof, K, Ce)

f_0 = np.zeros((np.size(dofs)*2,1))

ptype = 2

alpha_cu = 17.6e-6
alpha_ny = 80e-6

D_list = []
for eldof, temp_eldof, elx, ely, material_index in zip(stress_edof, edof, ex, ey, element_markers):
    E = E_cu if material_index == COPPER_SURFACE else E_ny
    v = v_cu if material_index == COPPER_SURFACE else v_ny
    D = cfc.hooke(ptype, E, v)[np.ix_([0, 1, 3], [0, 1, 3])]
    D_list.append(D)
    alpha = alpha_cu if material_index == COPPER_SURFACE else alpha_ny
    print(a_stat)
    dt = ((a_stat[temp_eldof[0]-1] + a_stat[temp_eldof[1]-1] + a_stat[temp_eldof[2]-1])/3)[0] - T_inf
    internal_force = cfc.plantf(elx, ely, [ptype, t], D*alpha*dt@np.array([1,1,0]).T)
    for i_f, dof in zip(internal_force, eldof):
        f_0[dof-1] = i_f
    # print(internal_force)

# bc, bcVal = cfu.applybc(bdofs, bc, bcVal, 5, 0.0, 0)
temp_fixed_bdofs = list(set(bdofs[FIXED_BOUNDARY]+bdofs[HEAT_FLUX_BOUNDARY]))
strain_fixed_dofs = np.array([[2*dof-1, 2*dof] for dof in temp_fixed_bdofs]).flatten()
x_strain_fixed = [2*dof-1 for dof in bdofs[X_FIXED_BOUNDARY]]
y_strain_fixed = [2*dof for dof in bdofs[Y_FIXED_BOUNDARY]]
strain_fixed_dofs = list(set(list(strain_fixed_dofs) + x_strain_fixed + y_strain_fixed))
print(strain_fixed_dofs)
# bc, bcVal = cfu.applybc(strain_fixed_dofs, bc, bcVal, 5, 0.0, 0)
a, _ = cfc.solveq(K, f_0, np.array(strain_fixed_dofs), np.zeros_like(np.array(strain_fixed_dofs)))
# plt.spy(K)
print(a)
ed = cfc.extractEldisp(stress_edof, a)



# cfv.figure(fig_size=(10,10))
# cfv.draw_displacements(a, coords, stress_edof, 2, 2,
# draw_undisplaced_mesh=True, title="Displacements",
# magnfac=1)
ed = cfc.extractEldisp(stress_edof, a)

mises = []
for dof, temp_edof, elx, ely, disp, D, material_index in zip(stress_edof, edof, ex, ey, ed, D_list, element_markers):
            E = E_cu if material_index == COPPER_SURFACE else E_ny
            v = v_cu if material_index == COPPER_SURFACE else v_ny
            alpha = alpha_cu if material_index == COPPER_SURFACE else alpha_ny
            # Determine element stresses and strains in the element.
            dt = ((a_stat[temp_eldof[0]-1] + a_stat[temp_eldof[1]-1] + a_stat[temp_eldof[2]-1])/3)[0] - T_inf

            sigma, epsilon = cfc.plants(elx, ely, [ptype, t], D, disp)
            sigma = sigma.flatten()
            epsilon = epsilon.flatten()
            print(sigma, epsilon)
            # Sida 255
            sigma = sigma-alpha*E*dt/(1-2*v)
            sigma_zz = E*v/((1+v)*(1-2*v))*(epsilon[0]+epsilon[1])-alpha*E*dt/(1-2*v)
            
            
            # Calc and append effective stress to list.

            mises.append((sigma[0]**2+sigma[1]**2+sigma_zz**2-sigma[0]*sigma[1]-sigma[0]*sigma_zz+3*sigma[2]**2)**(1/2))
print(mises)
# plt.show()
cfv.figure(fig_size=(10,10))
cfv.draw_element_values(mises, coords, stress_edof, 2, 2, a,
    draw_elements=True, draw_undisplaced_mesh=True,
    title="Effective Stress", magnfac=1.0)
cfv.colorbar()
cfv.show_and_wait()