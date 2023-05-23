import calfem.geometry as cfg
import calfem.mesh as cfm


class ClampingBoundaryKeys:
    HEAT_FLUX_BOUNDARY = 6
    CONVECTION_BOUNDARY = 8
    FIXED_BOUNDARY = 10
    X_FIXED_BOUNDARY = 11
    Y_FIXED_BOUNDARY = 12


class ClampingMaterialKeys:
    COPPER = 1
    NYLON = 2


def gripper_geometry(L):
    a = 0.1*L
    b = 0.1*L
    c = 0.3*L
    d = 0.05*L
    h = 0.15*L
    t = 0.05*L

    g = cfg.geometry()

    points = [
        [0, 0.5*L],
        [a, 0.5*L],
        [a, 0.5*L-b],
        [a+c, 0.5*L-b],
        [a+c+d, 0.5*L-b-d],  # 0-4
        [a+c+d, d],
        [L-2*d, 0.3*L],
        [L, 0.3*L],
        [L, 0.3*L-d],
        [L-2*d, 0.3*L-d],  # 5-9
        [a+c+d, 0],
        [c+d, 0],
        [c+d, 0.5*L-b-a],
        [a+t, 0.5*L-b-a],
        [a+t, 0.5*L-b-a-h],  # 10-14
        [a, 0.5*L-b-a-h],
        [a, 0.5*L-b-a],
        [0, 0.5*L-b-a],
        [0, 0.5*L-b],
        [0, 0]  # 15-19
    ]

    for xp, yp in points:
        g.point([xp, yp])

    NUM = None

    for s in [[0, 1]]:
        g.spline(s, marker=ClampingBoundaryKeys.Y_FIXED_BOUNDARY, el_on_curve=NUM)

    for s in [[1, 2], [2, 3], [3, 4], [4, 5],
              [5, 6], [6, 7]]:
        g.spline(s, marker=ClampingBoundaryKeys.CONVECTION_BOUNDARY,
                 el_on_curve=NUM)

    for s in [[7, 8]]:
        g.spline(s, marker=ClampingBoundaryKeys.X_FIXED_BOUNDARY, el_on_curve=NUM)

    for s in [[8, 9], [9, 10]]:
        g.spline(s, marker=ClampingBoundaryKeys.CONVECTION_BOUNDARY,
                 el_on_curve=NUM)

    for s in [[10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
              [15, 16], [16, 17]]:
        g.spline(s, el_on_curve=NUM)

    for s in [[17, 18]]:
        g.spline(s, marker=ClampingBoundaryKeys.FIXED_BOUNDARY, el_on_curve=NUM)

    for s in [[18, 0]]:
        g.spline(s, marker=ClampingBoundaryKeys.HEAT_FLUX_BOUNDARY,
                 el_on_curve=NUM)

    for s in [[17, 19]]:
        g.spline(s, marker=ClampingBoundaryKeys.FIXED_BOUNDARY, el_on_curve=NUM)

    for s in [[19, 11]]:
        g.spline(s, el_on_curve=NUM)

    print(ClampingMaterialKeys.COPPER)
    g.surface(list(range(0, 19)), marker=ClampingMaterialKeys.COPPER)
    g.surface(list(set(range(11, 21)) - set([17, 18])),
              marker=ClampingMaterialKeys.NYLON)

    return g


def gripper_mesh(L, el_type, dof=1, size_factor=None):
    geometry = gripper_geometry(L)
    mesh = cfm.GmshMesh(geometry, el_type, dof)
    mesh.return_boundary_elements = True
    if size_factor is not None:
        mesh.el_size_factor = size_factor
    return mesh.create()
