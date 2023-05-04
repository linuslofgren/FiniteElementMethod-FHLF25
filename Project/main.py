import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import calfem.core as cfc
import numpy as np

L = 0.005
a = 0.1*L
b = 0.1*L
c = 0.3*L
d = 0.05*L
h = 0.15*L
t = 0.05*L

# t = 0.005
v = 0.35
E = 2.1e9
ptype = 1
ep = [ptype, t]
D = cfc.hooke(ptype, E, v)


g = cfg.geometry()


s2 = 1/np.sqrt(2)


points = [[0, 0.5*L], [a, 0.5*L], [a, 0.5*L-b], [a+c, 0.5*L-b], [a+c+d, 0.5*L-b-d],  # 0-4
          [a+c+d, d], [L-2*d, 0.3*L], [L, 0.3*L], [L, 0.3*L-d], [L-2*d, 0.3*L-d],  # 5-9
          [a+c+d, 0], [c+d, 0], [c+d, 0.5*L-b-a], [a+t, 0.5*L-b-a], [a+t, 0.5*L-b-a-h],  # 10-14
          [a, 0.5*L-b-a-h], [a, 0.5*L-b-a], [0, 0.5*L-b-a], [0, 0]]  # 15-18

for xp, yp in points:
    g.point([xp, yp])

splines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],  # 0-4
           [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],  # 5-9
           [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],  # 10-14
           [15, 16], [16, 17], [17, 0], [18, 17], [18, 11]]

for s in splines:
    g.spline(s)#, el_on_curve=10)

cfv.draw_geometry(g, draw_points=True, label_curves=True, label_points=True)
cfv.showAndWait()

# g.curve_marker(ID=4,  marker=7) #Assign marker 7 to the splines on the right.
# g.curve_marker(ID=5,  marker=7) # We will apply a force on nodes with marker 7.
# g.curve_marker(ID=10, marker=5) #Assign marker 5 to the splines on the left.
# g.curve_marker(ID=11, marker=5) # The nodes with marker 5 will be locked in place.


# Points in circle arcs are [start, center, end]

# circlearcs = [[2, 23, 3], [3, 23, 4], [4, 23, 5], [5, 23, 6],           #26-29
#               [16, 24, 17], [17, 24, 18], [18, 24, 19], [19, 24, 20]]   #30-33

# for c in circlearcs:
#     g.circle(c, el_on_curve=10)


# g.struct_surf([11,12,13,0]) #0
# g.struct_surf([14, 12, 10, 9])
# g.struct_surf([8, 30, 24, 14])
# g.struct_surf([24, 31, 17, 15])
# g.struct_surf([15, 16, 27, 22]) #4
# g.struct_surf([22, 26, 1, 13])
# g.struct_surf([16, 18, 23, 28])
# g.struct_surf([19, 2, 29, 23])
# g.struct_surf([19, 21, 4, 3]) #8
# g.struct_surf([20, 6, 5, 21])
# g.struct_surf([25, 20, 7, 33])
# g.struct_surf([32, 17, 18, 25]) #11


mesh = cfm.GmshMesh(g)

mesh.el_type = 3
mesh.dofs_per_node = 2

coords, edof, dofs, bdofs, elementmarkers = mesh.create()


nDofs = np.size(dofs)
ex, ey = cfc.coordxtr(edof, coords, dofs)
K = np.zeros([nDofs,nDofs])

for eltopo, elx, ely in zip(edof, ex, ey):
    Ke = cfc.planqe(elx, ely, ep, D)
    cfc.assem(eltopo, K, Ke)


f = np.zeros([nDofs,1])

bc = np.array([],'i')
bcVal = np.array([],'f')

bc, bcVal = cfu.applybc(bdofs, bc, bcVal, 5, 0.0, 0)

cfu.applyforce(bdofs, f, 7, 10e5, 1)

a,r = cfc.solveq(K,f,bc,bcVal)



ed = cfc.extract_eldisp(edof,a)

vonMises = []

# For each element:
for i in range(edof.shape[0]):

    # Determine element stresses and strains in the element.
    es, et = cfc.planqs(ex[i,:], ey[i,:], ep, D, ed[i,:])

    # Calc and append effective stress to list.
    vonMises.append(np.sqrt(np.power(es[0],2) - es[0]*es[1] + np.power(es[1],2) + 3*es[2] ) )

    ## es: [sigx sigy tauxy]


cfv.figure(fig_size=(10,10))
cfv.draw_element_values(vonMises, coords, edof, mesh.dofs_per_node, mesh.el_type, None, draw_elements=False, draw_undisplaced_mesh=False, title="Example 6 - Effective stress")

cfv.showAndWait()