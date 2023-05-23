import calfem.core as cfc
import numpy as np
from plantml import plantml
from operator import itemgetter
from geometry import gripper_mesh, ClampingBoundaryKeys, ClampingMaterialKeys


class ClampingProblem:
    """This class solves the static and transient heat distribution problem and heat stress/displacement problem of a gripper."""

    def __init__(self, size_factor=None) -> None:
        """
        size_factor: mesh element size (optional)
        """
        # Material parameters, found in project description
        self.materials = {
            ClampingMaterialKeys.COPPER: IsomorphicMaterial(385, 128e9, 0.36, 17.6e-6, 8930, 386, "Copper"),
            ClampingMaterialKeys.NYLON: IsomorphicMaterial(
                0.26, 3e9, 0.39, 80e-6, 1100, 1500, "Nylon")
        }

        self.T_inf = 18  # [C]
        self.T_0 = 18  # [C]

        # Thermal load
        self.h = -1e5  # [W/m^2]

        # Convection constant
        self.alpha_c = 40  # [W/(m^2 K)]

        # Three-point triangle element
        self.el_type = 2

        self.thickness = [0.005]  # [m]
        # Geometry defined in terms of L
        self.L = 0.005  # [M]

        self.coords, self.edof, self.dofs, self.bdofs, self.element_markers, self.boundary_elements = gripper_mesh(
            self.L, self.el_type, 1, size_factor)
        self.ex, self.ey = cfc.coord_extract(self.edof, self.coords, self.dofs)

        # CALFEM specific shorthand
        self.ep = [2, self.thickness[0]]

    def integrate_boundary_load(self, node_pair_list, f_sub, factor):
        # Calculates the integral N^tN over an element edge
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            f_sub[np.isin(self.dofs, [p1, p2])] += factor * distance

    def integrate_boundary_convection(self, node_pair_list, k_sub):
        # Calculates the integral N^tN over an element edge
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            # Specific for our three pint triangular element
            k_e = np.array([[1/3, 1/6],
                            [1/6, 1/3]]) * \
                self.alpha_c * self.thickness[0] * distance

            cfc.assem(np.array([p1, p2]), k_sub, k_e)

    def solve_static(self):
        # Solves the static heat problem
        # (K + K_c) a = f_h + f_c

        # Assemble K matrix
        K = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        for eldof, elx, ely, material_index in zip(self.edof, self.ex, self.ey, self.element_markers):
            Ke = cfc.flw2te(elx, ely, self.thickness,
                            self.materials[material_index].D)
            cfc.assem(eldof, K, Ke)

        # Create force vector
        f = np.zeros([np.size(self.dofs), 1])

        # Add f_h
        f_h_nodes = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[ClampingBoundaryKeys.HEAT_FLUX_BOUNDARY]
        ))
        self.integrate_boundary_load(
            f_h_nodes, f, -self.h*self.thickness[0] * 1/2)

        # Add f_c
        f_c_nodes = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[ClampingBoundaryKeys.CONVECTION_BOUNDARY]
        ))
        self.integrate_boundary_load(f_c_nodes, f, self.T_inf *
                                     self.alpha_c * self.thickness[0] * 1/2)

        # Add K_c
        K_c = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        self.integrate_boundary_convection(f_c_nodes, K_c)
        K += K_c

        a_stat = np.linalg.solve(K, f)

        return a_stat, K, f

    def solve_transient(self):
        # Solve the transient heat problem
        # (C + ∆tθK)a_{n+1} = ∆fn + (C − ∆tK(1 − θ))a_n with a_0 = T_0

        # Use the K and f matrices (+ the static heat distribution) from the static problem
        a_stat, K, f = self.solve_static()

        # Assemble the C matrix
        C = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        for eldof, elx, ely, material_index in zip(self.edof, self.ex, self.ey, self.element_markers):
            rho = self.materials[material_index].density
            c_v = self.materials[material_index].spec_heat
            Ce = plantml(elx, ely, (rho * c_v * self.thickness[0]))
            cfc.assem(eldof, C, Ce)

        # Determine time-stepping method
        theta = 1.0
        # Times step
        delta_t = 0.05

        # a_0 is given as T_0
        a = np.full(self.dofs.shape, self.T_0)

        # Keep track of the time and save snapshots of the temperature distribution
        total_time = 0
        snapshots = [a]
        snapshot_time = [total_time]

        # Advance time until at 90% of maximum temperature
        while np.amax(a) < ((np.amax(a_stat)-self.T_0) * 0.9)+self.T_0:
            # Solve the heat problem as stated above
            A = C+delta_t*theta*K
            b = delta_t*f+(C-delta_t*K*(1-theta))@a
            a = np.linalg.solve(A, b)

            snapshots.append(a)
            snapshot_time.append(total_time)
            total_time += delta_t

        return snapshots, snapshot_time

    def solve_displacement(self):
        # Displacement solver, assumes plain strain
        # Plane strain --> ptype = 2
        ptype = 2

        # Get the static temperature distribution
        a_stat, _, _ = self.solve_static()

        # Create new degrees of freedom for the stress problem: each node has (x, y) displacement
        # This means twice the number of dofs as the heat problem
        # Each dof is located at index dof-1 in the matrix formulation
        stress_edof = np.full((self.edof.shape[0], self.edof.shape[1]*2), 0)

        # Map each old dof to a new dof via i -> (2*i-1, 2*i)
        def to_xdof(dof): return 2*dof-1
        def to_ydof(dof): return 2*dof
        def to_xydof(dof): return [to_xdof(dof), to_ydof(dof)]

        for (a, b, c), stress_a in zip(self.edof, stress_edof):
            stress_a[0], stress_a[1] = to_xydof(a)
            stress_a[2], stress_a[3] = to_xydof(b)
            stress_a[4], stress_a[5] = to_xydof(c)

        # Construct K-matrix (twice as many dofs as for the heat problem)
        K = np.zeros((np.size(self.dofs)*2, np.size(self.dofs)*2))
        element_constitutive_D = []
        for eldof, elx, ely, material_index in zip(stress_edof, self.ex, self.ey, self.element_markers):
            E = self.materials[material_index].E
            v = self.materials[material_index].v
            D = cfc.hooke(ptype, E, v)
            element_constitutive_D.append(D)
            Ke = cfc.plante(elx, ely, self.ep, D)
            cfc.assem(eldof, K, Ke)

        # Calculate the heat force vector
        f_0 = np.zeros((np.size(self.dofs)*2, 1))

        for eldof, temp_eldof, elx, ely, D, material_index in zip(stress_edof, self.edof, self.ex, self.ey, element_constitutive_D, self.element_markers):
            E = self.materials[material_index].E
            v = self.materials[material_index].v
            alpha = self.materials[material_index].alpha
            dt = np.mean(a_stat[temp_eldof-1]) - self.T_0

            # Use plantf to calculate the integral of the force vector
            internal_force = cfc.plantf(
                elx, ely, self.ep, D[np.ix_([0, 1, 3], [0, 1, 3])]*alpha*dt@np.array([1, 1, 0]).T).reshape(6, 1)

            f_0[eldof-1] += internal_force

        # Get dofs fixed in xy direction (in the heat problem)
        heat_fixed_bdofs = list(set(
            self.bdofs[ClampingBoundaryKeys.FIXED_BOUNDARY] +
            self.bdofs[ClampingBoundaryKeys.HEAT_FLUX_BOUNDARY]
        ))

        # Convert fixed heat dofs into stress dofs
        xy_strain_fixed = [
            direction_dof for dof in heat_fixed_bdofs for direction_dof in to_xydof(dof)]
        # Get dofs fixed in x direction (in the heat problem) and convert into stress dofs
        x_strain_fixed = list(
            map(to_xdof, self.bdofs[ClampingBoundaryKeys.X_FIXED_BOUNDARY]))
        # Get dofs fixed in y direction (in the heat problem) and convert into stress dofs
        y_strain_fixed = list(
            map(to_ydof, self.bdofs[ClampingBoundaryKeys.Y_FIXED_BOUNDARY]))

        # Remove duplicate dofs
        strain_fixed_dofs = np.unique(
            xy_strain_fixed + x_strain_fixed + y_strain_fixed)

        # Solve for displacement
        displacement, _ = cfc.solveq(
            K,
            f_0,
            strain_fixed_dofs,
            np.zeros_like(strain_fixed_dofs)
        )

        # Extract displacement per node
        ed = cfc.extract_eldisp(stress_edof, displacement)

        # Calculate the von Mises stress per element
        von_mises_element = []
        for temp_edof, elx, ely, disp, D, material_index in zip(self.edof, self.ex, self.ey, ed, element_constitutive_D, self.element_markers):
            E = self.materials[material_index].E
            v = self.materials[material_index].v
            alpha = self.materials[material_index].alpha

            # Average temperature of surrounding nodes
            dt = np.mean(a_stat[temp_edof-1]) - self.T_0

            # Determine element stresses and strains in the element.
            [[sigx, sigy, sigz, tauxy]], _ = cfc.plants(
                elx, ely, self.ep, D, disp)

            # Compensate for temperature stress
            e_temp = alpha*E*dt/(1-2*v)
            sigx -= e_temp
            sigy -= e_temp
            sigz -= e_temp

            # calculate von Mises stress per definition
            effective_stress = (sigx**2+sigy**2+sigz**2-sigx *
                                sigy-sigx*sigz+3*tauxy**2)**(1/2)

            von_mises_element.append(effective_stress)

        # Convert to np-array to help with mean stress calculation
        von_mises_element = np.array(von_mises_element)

        # Calculate the effective stress per node by averaging over surrounding element stresses
        von_mises_node = [np.mean(von_mises_element[np.any(
            np.isin(self.edof, node), axis=1)]) for node in self.dofs]

        return von_mises_node, displacement


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
