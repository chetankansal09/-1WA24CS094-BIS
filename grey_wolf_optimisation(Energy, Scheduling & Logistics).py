import numpy as np


# ==============================================================================
# CORE REUSABLE GREY WOLF OPTIMIZER (GWO) ENGINE
# ==============================================================================
def run_gwo(obj_func, dim, lb, ub, num_wolves=35, max_iter=200):
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    # Initialize wolf population randomly within bounds
    positions = np.random.uniform(lb, ub, (num_wolves, dim))

    # Initialize Alpha, Beta, and Delta leaders
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")

    beta_pos = np.zeros(dim)
    beta_score = float("inf")

    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    for t in range(max_iter):
        # Evaluate fitness and update pack hierarchy
        for i in range(num_wolves):
            positions[i] = np.clip(positions[i], lb, ub)
            fitness = obj_func(positions[i])

            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i].copy()

        # Linearly decrease parameter 'a' from 2 to 0
        a = 2.0 - t * (2.0 / max_iter)

        # Update positions of all wolves
        for i in range(num_wolves):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])

            positions[i] = (X1 + X2 + X3) / 3.0

    return alpha_pos, alpha_score


# ==============================================================================
# 1. ENERGY: ECONOMIC LOAD DISPATCH (ELD)
# ==============================================================================
def solve_energy_eld():
    # Generator specs: [a, b, c, P_min (MW), P_max (MW)]
    generators = np.array([
        [0.00156, 7.92, 561.0, 100.0, 600.0],
        [0.00194, 7.85, 310.0, 100.0, 400.0],
        [0.00482, 7.97, 78.0, 50.0, 200.0],
    ])
    total_demand = 850.0  # Target power demand in MW

    def eld_fitness(power_outputs):
        a, b, c = generators[:, 0], generators[:, 1], generators[:, 2]
        fuel_cost = np.sum(a * (power_outputs**2) + b * power_outputs + c)
        mismatch = abs(np.sum(power_outputs) - total_demand)
        penalty = 1000.0 * (mismatch**2)
        return fuel_cost + penalty

    best_power, min_cost = run_gwo(
        obj_func=eld_fitness,
        dim=len(generators),
        lb=generators[:, 3],
        ub=generators[:, 4],
        num_wolves=40,
        max_iter=300,
    )

    print("==========================================")
    print("1. ENERGY: ECONOMIC LOAD DISPATCH")
    print("==========================================")
    for idx, p in enumerate(best_power):
        print(f"Generator {idx + 1} Output : {p:.2f} MW")
    print(
        f"Total Power Generated: {np.sum(best_power):.2f} MW (Target:"
        f" {total_demand} MW)"
    )
    print(f"Minimum Fuel Cost    : ${min_cost:.2f}/hr\n")


# ==============================================================================
# 2. SCHEDULING: CLOUD TASK SCHEDULING (MAKESPAN MINIMIZATION)
# ==============================================================================
def solve_cloud_scheduling():
    # Base execution time (in seconds) for 10 tasks
    task_lengths = np.array(
        [12.0, 45.0, 23.0, 67.0, 34.0, 89.0, 15.0, 52.0, 41.0, 28.0]
    )
    # Speed multipliers of 3 Virtual Machines (VMs)
    vm_speeds = np.array([1.0, 1.5, 2.0])

    num_tasks = len(task_lengths)
    num_vms = len(vm_speeds)

    def scheduling_fitness(position):
        vm_assignments = np.floor(position).astype(int)
        vm_loads = np.zeros(num_vms)
        for task_idx, vm_idx in enumerate(vm_assignments):
            vm_loads[vm_idx] += task_lengths[task_idx] / vm_speeds[vm_idx]
        return np.max(vm_loads)  # Minimize makespan (busiest VM time)

    best_pos, min_makespan = run_gwo(
        obj_func=scheduling_fitness,
        dim=num_tasks,
        lb=np.zeros(num_tasks),
        ub=np.full(num_tasks, num_vms - 1e-5),
        num_wolves=35,
        max_iter=200,
    )

    best_assignment = np.floor(best_pos).astype(int)
    print("==========================================")
    print("2. SCHEDULING: CLOUD TASK ALLOCATION")
    print("==========================================")
    print(f"Task-to-VM Mapping   : {best_assignment}")
    for vm_id in range(num_vms):
        assigned_tasks = np.where(best_assignment == vm_id)[0]
        load = np.sum(task_lengths[assigned_tasks]) / vm_speeds[vm_id]
        print(
            f"VM {vm_id} (Speed {vm_speeds[vm_id]}x)   : Tasks"
            f" {assigned_tasks.tolist()} -> Finish Time: {load:.2f}s"
        )
    print(f"Minimum Makespan     : {min_makespan:.2f} seconds\n")


# ==============================================================================
# 3. LOGISTICS: DELIVERY ROUTE OPTIMIZATION (TRAVELING SALESMAN PROBLEM)
# ==============================================================================
def solve_logistics_routing():
    # (X, Y) coordinates in km for 8 delivery stops (Stop 0 = Depot/Warehouse)
    cities = np.array([
        [0.0, 0.0],
        [12.0, 45.0],
        [34.0, 18.0],
        [67.0, 52.0],
        [23.0, 78.0],
        [81.0, 15.0],
        [49.0, 63.0],
        [15.0, 22.0],
    ])
    num_cities = len(cities)

    def route_distance(position):
        # Random-key encoding: argsort turns continuous values into a valid route
        route = np.argsort(position)
        total_dist = 0.0
        for i in range(num_cities):
            current_city = cities[route[i]]
            next_city = cities[route[(i + 1) % num_cities]]
            total_dist += np.linalg.norm(current_city - next_city)
        return total_dist

    best_pos, shortest_dist = run_gwo(
        obj_func=route_distance,
        dim=num_cities,
        lb=np.zeros(num_cities),
        ub=np.ones(num_cities),
        num_wolves=50,
        max_iter=300,
    )

    optimal_route = np.argsort(best_pos)
    # Rotate route so it starts and ends at Stop 0 (Depot)
    depot_idx = np.where(optimal_route == 0)[0][0]
    ordered_route = np.roll(optimal_route, -depot_idx)
    full_trip = np.append(ordered_route, 0)

    print("==========================================")
    print("3. LOGISTICS: DELIVERY ROUTE OPTIMIZATION")
    print("==========================================")
    print(f"Optimal Route Order  : {' -> '.join(map(str, full_trip))}")
    print(f"Shortest Total Dist. : {shortest_dist:.2f} km\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)  # Fixed seed for reproducible results
    solve_energy_eld()
    solve_cloud_scheduling()
    solve_logistics_routing()
