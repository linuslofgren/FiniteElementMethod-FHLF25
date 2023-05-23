
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import calfem.core as cfc
import numpy as np
import matplotlib.pyplot as plt
from plantml import plantml
from operator import itemgetter
from geometry import gripper_mesh, ClampingBoundaryKeys, ClampingMaterialKeys


class IsomorphicMaterial:
    """Defines an isomorphic material
    k: Thermal conductivity [W/(mK)]
    E: Young's modulus [Pa]
    v: Poisson's ratio
    alpha: expansion coefficient [1/K]
    density: [kg/m^3]
    spec_heat: [J/(kg K)]
    name: optional material name
    """
    def __init__(self, k, E, v, alpha, density, spec_heat, name=None) -> None:
        self.k = k 
        self.D = np.diag([k, k])
        self.E = E
        self.v = v
        self.alpha = alpha
        self.density = density
        self.spec_heat = spec_heat
        self.name = name




class ClampingProblem:

    def __init__(self) -> None:
        self.materials = {
            ClampingMaterialKeys.COPPER: IsomorphicMaterial(385, 128e9, 0.36, 17.6e-6, 8930, 386, "Copper"),
            ClampingMaterialKeys.NYLON: IsomorphicMaterial(0.26, 3e9, 0.39, 80e-6, 1100, 1500, "Nylon")
        }

        self.T_inf = 18  # [C]
        self.T_0 = 18 # [C]

        # Thermal load
        self.h = -1e5  # [W/m**2]

        # Convection constant
        self.alpha_c = 40  # [W/(m^2K)]

        # Three-point triangle element
        self.el_type = 2

        self.thickness = [0.005]  # [m]
        self.L = 0.005  # [M]
        self.coords, self.edof, self.dofs, self.bdofs, self.element_markers, self.boundary_elements = gripper_mesh(self.L, self.el_type, 1)
        print(self.element_markers)
        self.ex, self.ey = cfc.coord_extract(self.edof, self.coords, self.dofs)

        # CALFEM specific
        self.ep = [2, self.thickness[0]]

    def integrate_boundary_load(self, node_pair_list, f_sub, factor):
        # Calculates the integral N^tN over an element edge
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            f_sub[np.isin(self.dofs, [p1, p2])] += factor * distance

    def integrate_boundary_convection(self, node_pair_list, k_sub):
        # Calculates the integral N^tN over an element
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            k_e = np.array([[1/3, 1/6],
                            [1/6, 1/3]]) * \
                self.alpha_c * self.thickness[0] * distance

            cfc.assem(np.array([p1, p2]), k_sub, k_e)

    def solve_static(self, show_figure=False):
        # cfv.draw_geometry(self.geometry)
        if show_figure:
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            cfv.draw_mesh(self.coords, self.edof, 1, 2)

        

        K = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        for eldof, elx, ely, material_index in zip(self.edof, self.ex, self.ey, self.element_markers):
            Ke = cfc.flw2te(elx, ely, self.thickness,
                            self.materials[material_index].D)
            cfc.assem(eldof, K, Ke)

        f = np.zeros([np.size(self.dofs), 1])

        # Add f_h
        f_h_nodes = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[ClampingBoundaryKeys.HEAT_FLUX_BOUNDARY]
        ))
        self.integrate_boundary_load(f_h_nodes, f, -self.h*self.thickness[0] * 1/2)

        # Add f_c
        f_c_nodes = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[ClampingBoundaryKeys.CONVECTION_BOUNDARY]
        ))
        self.integrate_boundary_load(f_c_nodes, f, self.T_inf *
                         self.alpha_c * self.thickness[0] * 1/2)

        # Add K_c (from page 220)
        K_c = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        self.integrate_boundary_convection(f_c_nodes, K_c)
        K += K_c

        a_stat = np.linalg.solve(K, f)

        print(f"Maximum temperature {np.amax(a_stat):.2f} (°C)")


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

        C = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        for eldof, elx, ely, material_index in zip(self.edof, ex, ey, self.element_markers):
            rho = self.materials[material_index].density
            c_v = self.materials[material_index].spec_heat
            Ce = plantml(elx, ely, (rho * c_v * self.thickness[0]))
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

        vmax = np.max(snapshots[-1])
        for snapshot, time, ax in zip(snapshots, snapshot_time, axes.flatten()):
            plt.sca(ax)
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            
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

        for eldof, elx, ely, material_index in zip(stress_edof, self.ex, self.ey, self.element_markers):
            D = cfc.hooke(ptype, self.materials[material_index].E,
                          self.materials[material_index].v)
            Ce = cfc.plante(elx, ely, self.ep, D)
            cfc.assem(eldof, K, Ce)

        f_0 = np.zeros((np.size(self.dofs)*2, 1))

        D_list = []
        for eldof, temp_eldof, elx, ely, material_index in zip(stress_edof, self.edof, self.ex, self.ey, self.element_markers):
            E = self.materials[material_index].E
            v = self.materials[material_index].v
            alpha = self.materials[material_index].alpha

            D = cfc.hooke(ptype, E, v)[np.ix_([0, 1, 3], [0, 1, 3])]
            D_list.append(D)
            dt = np.mean(a_stat[temp_eldof-1]) - self.T_0

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

        von_mises_element = []
        for dof, temp_edof, elx, ely, disp, _, material_index in zip(stress_edof, self.edof, self.ex, self.ey, ed, D_list, self.element_markers):
            
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

            von_mises_element.append(stress)
        von_mises_element = np.array(von_mises_element)
        von_mises_node = []
        for node in zip(self.dofs):
            s = np.mean(von_mises_element[np.any(np.isin(self.edof, node), axis=1)])
            von_mises_node.append(s)
            # print(node)
        # von_mises = von_mises2
        magnification = 10.0

        cfv.figure(fig_size=(10, 10))
        
        flip_y = np.array([([1, -1]*int(a.size/2))]).T
        flip_x = np.array([([-1, 1]*int(a.size/2))]).T
        
        # print(np.min(von_mises2))
        magnfac = magnification
        coords_list = [self.coords, [0, self.L]+[1, -1]*self.coords, [2*self.L, self.L]+[-1, -1]*self.coords, [2*self.L, 0]+[-1, 1]*self.coords]
        # coords = self.coords
        displacement_list = [a, np.multiply(flip_y, a), np.multiply(flip_y*flip_x, a), np.multiply(flip_x, a)]
        for coords, displacements in zip(coords_list, displacement_list):
            if displacements is not None:
                if displacements.shape[1] != coords.shape[1]:
                    displacements = np.reshape(displacements, (-1, coords.shape[1]))
                    coords_disp = np.asarray(coords + magnfac * displacements)
            cfv.draw_mesh(coords, self.edof, 1, self.el_type, color=(0, 0, 0, 0.1))
            cfv.draw_nodal_values_shaded(von_mises_node, coords_disp, self.edof, title=(f"Max temp {np.amax(von_mises_element):.2f} (C)"),
                                    dofs_per_node=1, el_type=2, draw_elements=False)
            
            cfv.draw_mesh(coords_disp, self.edof, 1, self.el_type, color=(0, 1, 0, 0.1))

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
