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

Tinf = 18  # [C]
h = 1e5  # [W/m**2]


# Definie Geometry


def define_geometry():
    L = 0.005
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
               [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],  # 5-9
               [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],  # 10-14
               [15, 16], [16, 17], [17, 18]]:
        g.spline(s)

    for s in  [[18, 0]]:
        g.spline(s, marker=2)

    for s in [[17, 19], [19, 11]]:
        g.spline(s)

    g.surface(list(range(0, 19)), marker=1)
    g.surface([11, 12, 13, 14, 15, 16, 19, 20], marker=2)

    return g


def generate_mesh(g, dof=1):
    mesh = cfm.GmshMesh(g, 2, dof)
    mesh.return_boundary_elements = True
    return mesh.create()


g = define_geometry()
cfv.draw_geometry(g)


coords, edof, dofs, bdofs, elementmarkers, boundary_elements = generate_mesh(g)

node_lists = []
for b in boundary_elements[2]:
    node_lists.append(b.get("node-number-list"))
print(node_lists)

ex, ey = cfc.coord_extract(edof, coords, dofs)

K = np.zeros((np.size(dofs), np.size(dofs)))
thickness = [0.001]

for eldof, elx, ely, materialindex in zip(edof, ex, ey, elementmarkers):
    Ke = cfc.flw2te(elx, ely, thickness, D_cu if materialindex == 1 else D_ny)
    cfc.assem(eldof, K, Ke)
print(K)



bc = np.array([], 'i')
bc_val = np.array([], 'i')
f = np.array([1/2, 0, 1/2])
cfv.draw_mesh(coords, edof, 1, 2)

f = np.zeros([np.size(dofs), 1])

heat_boundary_dofs = bdofs[2]
print("--"*10)

for i, (eldof, elx, ely) in enumerate(zip(edof, ex, ey)):
    if eldof[0] in heat_boundary_dofs:
        print(eldof, elx, ely)

heat_bounds_dof = []
t = 0.02
print("-"*10)

print(heat_boundary_dofs)
for i, (dof, coord) in enumerate(zip(dofs, coords)):
    if dof in heat_boundary_dofs:
        f[i] = 0
        print(coord)
                # heat_bounds_dof.append([j, coord[1]])
        f[i] = -h*t/2
# print(f)
print(f)
a, r = cfc.solveq(K, f, np.array(bdofs.get(0)))


cfv.figure(fig_size=(10, 10))
cfv.draw_nodal_values_shaded(a, coords, edof, title="Temperature",
                            dofs_per_node=1, el_type=2, draw_elements=True)



cfv.showAndWait()
