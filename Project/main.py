import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import calfem.core as cfc
import numpy as np

# Contitutional parameters
k_cu=385 # [W/(mK)]
D_cu=np.diag([k_cu,k_cu])

k_ny=0.26# [W/(mK)]
D_ny=np.diag([k_ny,k_ny])

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
            [a, 0.5*L-b-a-h], [a, 0.5*L-b-a], [0, 0.5*L-b-a], [0, 0]]  # 15-18

    for xp, yp in points:
        g.point([xp, yp])

    splines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],  # 0-4
            [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],  # 5-9
            [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],  # 10-14
            [15, 16], [16, 17], [17, 0], [18, 17], [18, 11]]

    for s in splines:
        g.spline(s)
    g.surface(list(range(0,18)),marker=1)
    g.surface(list(range(11,17))+list(range(18,21)),marker=2)
    return g

cfv.draw_geometry(define_geometry(), draw_points=True, label_curves=True, label_points=True)
cfv.showAndWait()
