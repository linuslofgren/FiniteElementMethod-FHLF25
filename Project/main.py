from clamping_problem import ClampingProblem
from clamping_visualization import vis_mesh, vis_temp, vis_transient, vis_displacement

if __name__ == "__main__":
    problem = ClampingProblem(0.04)
    
    vis_mesh(problem)

    a_stat, _, _ = problem.solve_static()
    vis_temp(a_stat, problem)

    snapshots, snapshot_time = problem.solve_transient()
    vis_transient(snapshots, snapshot_time, problem)

    von_mises, displacement = problem.solve_displacement()
    vis_displacement(von_mises, displacement, problem)
