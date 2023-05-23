import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import calfem.core as cfc
import numpy as np
from collections import namedtuple
from time import sleep
import matplotlib.pyplot as plt
from plantml import plantml
from operator import itemgetter


class IsomorphicMaterial:
    def __init__(self, k, E, v, alpha, name=None) -> None:
        self.k = k  # [W/(mK)]
        self.D = np.diag([k, k])
        self.E = E  # [Pa]
        self.v = v
        self.alpha = alpha
        self.name = name


class ClampingProblem:

    COPPER = 1
    NYLON = 2

    HEAT_FLUX_BOUNDARY = 6
    CONVECTION_BOUNDARY = 8
    FIXED_BOUNDARY = 10
    X_FIXED_BOUNDARY = 11
    Y_FIXED_BOUNDARY = 12

    def __init__(self, L=0.005) -> None:
        self.materials = {
            ClampingProblem.COPPER: IsomorphicMaterial(385, 128e9, 0.36, 17.6e-6, "Copper"),
            ClampingProblem.NYLON: IsomorphicMaterial(0.26, 3e9, 0.39, 80e-6, "Nylon")
        }
        self.T_inf = 18  # [C]
        self.T_0 = 18 # [C]
        self.h = -1e5  # [W/m**2]
        self.thickness = [0.005]  # [m]
        self.alpha_c = 40  # [W/(m^2K)]
        self.L = L  # [M]

        # CALFEM specific
        self.ep = [2, self.thickness[0]]

        self.geometry = self.define_geometry()
        self.coords, self.edof, self.dofs, self.bdofs, self.element_markers, self.boundary_elements = self.generate_mesh()

    def define_geometry(self):
        a = 0.1*self.L
        b = 0.1*self.L
        c = 0.3*self.L
        d = 0.05*self.L
        h = 0.15*self.L
        t = 0.05*self.L

        g = cfg.geometry()

        points = [
            [0, 0.5*self.L],
            [a, 0.5*self.L],
            [a, 0.5*self.L-b],
            [a+c, 0.5*self.L-b],
            [a+c+d, 0.5*self.L-b-d],  # 0-4
            [a+c+d, d],
            [self.L-2*d, 0.3*self.L],
            [self.L, 0.3*self.L],
            [self.L, 0.3*self.L-d],
            [self.L-2*d, 0.3*self.L-d],  # 5-9
            [a+c+d, 0],
            [c+d, 0],
            [c+d, 0.5*self.L-b-a],
            [a+t, 0.5*self.L-b-a],
            [a+t, 0.5*self.L-b-a-h],  # 10-14
            [a, 0.5*self.L-b-a-h],
            [a, 0.5*self.L-b-a],
            [0, 0.5*self.L-b-a],
            [0, 0.5*self.L-b],
            [0, 0]  # 15-19
        ]

        for xp, yp in points:
            g.point([xp, yp])

        NUM = 2

        for s in [[0, 1]]:
            g.spline(s, marker=ClampingProblem.Y_FIXED_BOUNDARY, el_on_curve=NUM)

        for s in [[1, 2], [2, 3], [3, 4], [4, 5],  # 0-4
                  [5, 6], [6, 7]]:
            g.spline(s, marker=ClampingProblem.CONVECTION_BOUNDARY,
                     el_on_curve=NUM)

        for s in [[7, 8]]:
            g.spline(s, marker=ClampingProblem.X_FIXED_BOUNDARY, el_on_curve=NUM)

        for s in [[8, 9], [9, 10]]:
            g.spline(s, marker=ClampingProblem.CONVECTION_BOUNDARY,
                     el_on_curve=NUM)

        for s in [  # 5-9
                [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],  # 10-14
                [15, 16], [16, 17]]:
            g.spline(s, el_on_curve=NUM)

        for s in [[17, 18]]:
            g.spline(s, marker=ClampingProblem.FIXED_BOUNDARY, el_on_curve=NUM)

        for s in [[18, 0]]:
            g.spline(s, marker=ClampingProblem.HEAT_FLUX_BOUNDARY,
                     el_on_curve=NUM)

        for s in [[17, 19]]:
            g.spline(s, marker=ClampingProblem.FIXED_BOUNDARY, el_on_curve=NUM)

        for s in [[19, 11]]:
            g.spline(s, el_on_curve=NUM)

        g.surface(list(range(0, 19)), marker=ClampingProblem.COPPER)
        g.surface(list(set(range(11, 21)) - set([17, 18])),
                  marker=ClampingProblem.NYLON)

        return g

    def generate_mesh(self, dof=1):
        mesh = cfm.GmshMesh(self.geometry, 2, dof)
        mesh.return_boundary_elements = True
        return mesh.create()

    def N_transpose(self, node_pair_list, f_sub, factor):
        # Calculates the integral N^tN over an element edge
        for node_pair in node_pair_list:
            r1 = None
            r1_dof_index = None
            r2 = None
            r2_dof_index = None
            for i, (coord, dof) in enumerate(zip(self.coords, self.dofs)):
                if dof == node_pair[0]:
                    r1 = coord
                    r1_dof_index = i
                if dof == node_pair[1]:
                    r2 = coord
                    r2_dof_index = i

            distance = np.linalg.norm(np.array(r1) - np.array(r2))

            f_sub[r1_dof_index] += factor * distance
            f_sub[r2_dof_index] += factor * distance

    def N_N_transpose(self, node_pair_list, k_sub):
        # Calculates the integral N^tN over an element
        for node_pair in node_pair_list:
            r1 = None
            r1_dof_index = None
            r2 = None
            r2_dof_index = None
            for i, (coord, dof) in enumerate(zip(self.coords, self.dofs)):
                if dof == node_pair[0]:
                    r1 = coord
                    r1_dof_index = i
                if dof == node_pair[1]:
                    r2 = coord
                    r2_dof_index = i

            distance = np.linalg.norm(np.array(r1) - np.array(r2))

            k_sub[r1_dof_index][r1_dof_index] += 1/3 * \
                self.alpha_c * self.thickness[0] * distance
            k_sub[r2_dof_index][r2_dof_index] += 1/3 * \
                self.alpha_c * self.thickness[0] * distance
            k_sub[r1_dof_index][r2_dof_index] += 1/6 * \
                self.alpha_c * self.thickness[0] * distance
            k_sub[r2_dof_index][r1_dof_index] += 1/6 * \
                self.alpha_c * self.thickness[0] * distance

    def solve_static(self, show_figure=False):
        # cfv.draw_geometry(self.geometry)
        if show_figure:
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            cfv.draw_mesh(self.coords, self.edof, 1, 2)

        ex, ey = cfc.coord_extract(self.edof, self.coords, self.dofs)

        K = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        for eldof, elx, ely, material_index in zip(self.edof, ex, ey, self.element_markers):
            Ke = cfc.flw2te(elx, ely, self.thickness,
                            self.materials[material_index].D)
            cfc.assem(eldof, K, Ke)

        f = np.zeros([np.size(self.dofs), 1])

        # Add f_h
        node_lists = node_lists_conv = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[ClampingProblem.HEAT_FLUX_BOUNDARY]
        ))
        self.N_transpose(node_lists, f, -self.h*self.thickness[0] * 1/2)

        # Add f_c
        node_lists_conv = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[ClampingProblem.CONVECTION_BOUNDARY]
        ))
        self.N_transpose(node_lists_conv, f, self.T_inf *
                         self.alpha_c * self.thickness[0] * 1/2)

        # MARK: Sida 220
        K_c = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        self.N_N_transpose(node_lists_conv, K_c)
        K += K_c

        a_stat = np.linalg.solve(K, f)

        max_temp = np.amax(a_stat)

        print(f"Maximum temperature {max_temp:.2f} (°C)")


        if show_figure:
            cfv.figure(fig_size=(10, 10))
            cfv.draw_nodal_values_shaded(a_stat, self.coords, self.edof, title=(f"Max temp {np.amax(a_stat):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=True)
            cfv.draw_nodal_values_shaded(a_stat, [0, self.L]+[1, -1]*self.coords, self.edof, title=(f"Max temp {np.amax(a_stat):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=True)
            cfv.draw_nodal_values_shaded(a_stat, [2*self.L, self.L]+[-1, -1]*self.coords, self.edof, title=(f"Max temp {np.amax(a_stat):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=True)
            cfv.draw_nodal_values_shaded(a_stat, [2*self.L, 0]+[-1, 1]*self.coords, self.edof, title=(f"Max temp {np.amax(a_stat):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=True)
            cfv.draw_nodal_values_shaded(a_stat, [2*self.L, 0]+[-1, 1]*self.coords, self.edof, title=(f"Maximum temperature {np.amax(a_stat):.2f} °C"),
                                    dofs_per_node=1, el_type=2, draw_elements=True)
            
            cfv.colorbar()
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            for c in cfv.gca().collections:
                
                if c.colorbar:
                    c.colorbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
            cfv.show_and_wait()

        return a_stat, K, f

    def solve_transient(self):
        # ---------------------
        # MARK: Del B
        # ---------------------

        ex, ey = cfc.coord_extract(self.edof, self.coords, self.dofs)

        a_stat, K, f = self.solve_static()

        rho_cu = 8930  # [kg/m^3]
        rho_ny = 1100  # [kg/m^3]

        c_p_cu = 386  # [J/(kgK)]
        c_p_ny = 1500  # [J/(kgK)]

        C = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        for eldof, elx, ely, material_index in zip(self.edof, ex, ey, self.element_markers):
            Ce = plantml(elx, ely, (rho_cu * c_p_cu * self.thickness[0]) if material_index ==
                         ClampingProblem.COPPER else (rho_ny * c_p_ny * self.thickness[0]))
            cfc.assem(eldof, C, Ce)

        theta = 1.0
        delta_t = 0.05

        a = np.full(self.dofs.shape, self.T_0)

        total_time = 0
        time_90_perc = None
        while np.amax(a) < ((np.amax(a_stat)-self.T_0) * 0.9)+self.T_0:
            A = C+delta_t*theta*K
            b = delta_t*f+(C-delta_t*K*(1-theta))@a
            a = np.linalg.solve(A, b)
            total_time += delta_t
        
        print(f"total time to 90% {total_time:.2f} seconds")
        time_90_perc = total_time
        time_3_perc = 0.03 * time_90_perc
        time_step = (time_3_perc / 4)

        a = np.full(self.dofs.shape, self.T_0)

        total_time = 0
        time_90_perc = None
        time_step_index = 0

        snapshots = []
        snapshot_time = []
        for _ in np.arange(0, 100, delta_t):
            if total_time >= time_step_index * time_step:
                time_step_index += 1
                snapshots.append(a)
                snapshot_time.append(total_time)

            if time_step_index == 5:
                break
            A = C+delta_t*theta*K
            b = delta_t*f+(C-delta_t*K*(1-theta))@a
            a = np.linalg.solve(A, b)
            total_time += delta_t

        fig, axes  = plt.subplots(3,2)
        fig.tight_layout()
        axes.flatten()[-1].axis('off')
        # vmax = np.max(a_stat)
        vmax = np.max(snapshots[-1])
        for snapshot, time, ax in zip(snapshots, snapshot_time, axes.flatten()):
            plt.sca(ax)
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            
            # cfv.figure(fig, fig_size=(10, 10))
            cfv.draw_nodal_values_shaded(snapshot, self.coords, self.edof, title=(f"Max temp {np.amax(snapshot):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=False, vmin=self.T_0, vmax=vmax)
            cfv.draw_nodal_values_shaded(snapshot, [0, self.L]+[1, -1]*self.coords, self.edof, title=(f"Max temp {np.amax(snapshot):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=False, vmin=self.T_0, vmax=vmax)
            cfv.draw_nodal_values_shaded(snapshot, [2*self.L, self.L]+[-1, -1]*self.coords, self.edof, title=(f"Max temp {np.amax(snapshot):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=False, vmin=self.T_0, vmax=vmax)
            cfv.draw_nodal_values_shaded(snapshot, [2*self.L, 0]+[-1, 1]*self.coords, self.edof, title=(f"Max temp {np.amax(snapshot):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=False, vmin=self.T_0, vmax=vmax)
            cfv.draw_nodal_values_shaded(snapshot, [2*self.L, 0]+[-1, 1]*self.coords, self.edof, title=(f"t={time:.2f}s, max temp {np.amax(snapshot):.2f} °C"),
                                    dofs_per_node=1, el_type=2, draw_elements=False, vmin=self.T_0, vmax=vmax)


        fig.subplots_adjust(right=0.8)
        plt.colorbar(ax=axes.ravel().tolist())
        
        for c in cfv.gca().collections:
            
            if c.colorbar:
                c.colorbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
        cfv.show_and_wait()
    def solve_displacement(self, temp_values=None):

        # MARK: Del C (plane strain)

        ex, ey = cfc.coord_extract(self.edof, self.coords, self.dofs)

        if temp_values is None:
            a_stat, _, _ = self.solve_static()
        else:
            a_stat = temp_values

        # Create new degrees of freedom
        stress_edof = np.full((self.edof.shape[0], self.edof.shape[1]*2), 0)

        for (a, b, c), stress_a in zip(self.edof, stress_edof):
            # Map each old dof to a new dof via i -> (2*i-1, 2*i)
            stress_a[0] = 2*a-1
            stress_a[1] = 2*a
            stress_a[2] = 2*b-1
            stress_a[3] = 2*b
            stress_a[4] = 2*c-1
            stress_a[5] = 2*c

        # cfc.plants
        
        ptype = 2

        K = np.zeros((np.size(self.dofs)*2, np.size(self.dofs)*2))

        for eldof, elx, ely, material_index in zip(stress_edof, ex, ey, self.element_markers):
            D = cfc.hooke(ptype, self.materials[material_index].E,
                          self.materials[material_index].v)
            Ce = cfc.plante(elx, ely, self.ep, D)
            cfc.assem(eldof, K, Ce)

        f_0 = np.zeros((np.size(self.dofs)*2, 1))

        D_list = []
        for eldof, temp_eldof, elx, ely, material_index in zip(stress_edof, self.edof, ex, ey, self.element_markers):
            E = self.materials[material_index].E
            v = self.materials[material_index].v
            alpha = self.materials[material_index].alpha

            D = cfc.hooke(ptype, E, v)[np.ix_([0, 1, 3], [0, 1, 3])]
            D_list.append(D)

            dt = ((a_stat[temp_eldof[0]-1] + a_stat[temp_eldof[1] -
                  1] + a_stat[temp_eldof[2]-1])/3)[0] - self.T_inf

            internal_force = cfc.plantf(
                elx, ely, self.ep, D*alpha*dt@np.array([1, 1, 0]).T)

            for i_f, dof in zip(internal_force, eldof):
                f_0[dof-1] += i_f

        temp_fixed_bdofs = list(set(
            self.bdofs[ClampingProblem.FIXED_BOUNDARY] +
            self.bdofs[ClampingProblem.HEAT_FLUX_BOUNDARY]
        ))
        strain_fixed_dofs = np.array(
            [[2*dof-1, 2*dof] for dof in temp_fixed_bdofs]
        ).flatten()
        x_strain_fixed = [
            2*dof-1 for dof in self.bdofs[ClampingProblem.X_FIXED_BOUNDARY]
        ]
        y_strain_fixed = [
            2*dof for dof in self.bdofs[ClampingProblem.Y_FIXED_BOUNDARY]
        ]
        strain_fixed_dofs = list(
            set(list(strain_fixed_dofs) + x_strain_fixed + y_strain_fixed)
        )

        a, _ = cfc.solveq(
            K,
            f_0,
            np.array(strain_fixed_dofs),
            np.zeros_like(np.array(strain_fixed_dofs))
        )

        ed = cfc.extract_eldisp(stress_edof, a)

        von_mises = []
        for dof, temp_edof, elx, ely, disp, _, material_index in zip(stress_edof, self.edof, ex, ey, ed, D_list, self.element_markers):
            
            E = self.materials[material_index].E
            v = self.materials[material_index].v
            alpha = self.materials[material_index].alpha
            # Determine element stresses and strains in the element.
            dt = np.mean(a_stat[temp_edof-1]) - self.T_0

            # MARK: This is probably wrong, se sida 255
            D = cfc.hooke(ptype, E, v)
            [[sigx, sigy, sigz, tauxy]], [[epsx, epsy, epsz, gamxy]] = cfc.plants(elx, ely, self.ep, D, disp)
            sigx -= alpha*E*dt/(1-2*v)
            sigy -= alpha*E*dt/(1-2*v)

            sigz -= alpha*E*dt/(1-2*v)
            # sigz = v*(sigx + sigy) - (alpha * E * dt)

            stress = (sigx**2+sigy**2+sigz**2-sigx *
                      sigy-sigx*sigz+3*tauxy**2)**(1/2)

            von_mises.append(stress)

        magnification = 10.0

        cfv.figure(fig_size=(10, 10))
        
        flip_y = np.array([([1, -1]*int(a.size/2))]).T
        flip_x = np.array([([-1, 1]*int(a.size/2))]).T
        cfv.draw_element_values(von_mises, self.coords, stress_edof, 2, 2, a,
                                draw_elements=False, draw_undisplaced_mesh=True,
                                title="Effective Stress", magnfac=magnification)
        cfv.draw_element_values(von_mises, [0, self.L]+[1, -1]*self.coords, stress_edof, 2, 2, np.multiply(flip_y, a),
                                draw_elements=False, draw_undisplaced_mesh=True,
                                title="Effective Stress", magnfac=magnification)
        cfv.draw_element_values(von_mises, [2*self.L, self.L]+[-1, -1]*self.coords, stress_edof, 2, 2, np.multiply(flip_y*flip_x, a),
                                draw_elements=False, draw_undisplaced_mesh=True,
                                title="Effective Stress", magnfac=magnification)
        cfv.draw_element_values(von_mises, [2*self.L, 0]+[-1, 1]*self.coords, stress_edof, 2, 2, np.multiply(flip_x, a),
                                draw_elements=False, draw_undisplaced_mesh=True,
                                title="Effective stress and displacement", magnfac=magnification)
        cfv.colorbar()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        for c in cfv.gca().collections:
            
            if c.colorbar:
                c.colorbar.set_label('von Mises stress (Pa)', rotation=270, labelpad=20)
        cfv.show_and_wait()

if __name__ == "__main__":
    problem = ClampingProblem()
    # problem.solve_static(show_figure=True)
    # problem.solve_transient()
    problem.solve_displacement()
