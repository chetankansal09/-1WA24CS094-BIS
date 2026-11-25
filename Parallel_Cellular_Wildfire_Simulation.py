from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np

# ==============================================================================
# CONSTANTS
# ==============================================================================
EMPTY = 0
FOREST = 1
BURNING = 2
BURNED = 3


# ==============================================================================
# PARALLEL WORKER FUNCTION (EVALUATES A SLICE OF ROWS SIMULTANEOUSLY)
# ==============================================================================
def _update_row_slice(args):
    start_row, end_row, grid, elevation, wind_vector, base_prob, seed = args
    rng = np.random.default_rng(seed)
    rows, cols = grid.shape
    chunk_next = grid[start_row:end_row].copy()

    # 8 Moore neighborhood offsets: (dr, dc)
    moore_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for local_r, r in enumerate(range(start_row, end_row)):
        for c in range(cols):
            if grid[r, c] == BURNING:
                # Fuel exhausted; cell transitions to burned out
                chunk_next[local_r, c] = BURNED

            elif grid[r, c] == FOREST:
                p_ignite = 0.0

                # Inspect 8 Moore neighbors within grid bounds
                for dr, dc in moore_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr, nc] == BURNING:
                            # Spread direction from burning neighbor (nr, nc) to (r, c)
                            spread_r = r - nr
                            spread_c = c - nc

                            # WindFactor <- 1.0 + DotProduct(SpreadDirection, WindVector)
                            dot_product = (
                                spread_r * wind_vector[0]
                                + spread_c * wind_vector[1]
                            )
                            wind_factor = 1.0 + dot_product

                            # SlopeFactor <- 1.0 + 0.03 * (Elevation[r][c] - Elevation[nr][nc])
                            slope_factor = 1.0 + 0.03 * (
                                elevation[r, c] - elevation[nr, nc]
                            )

                            # P_local <- Clip(P_base * WindFactor * SlopeFactor, 0.0, 0.95)
                            p_local = np.clip(
                                base_prob * wind_factor * slope_factor,
                                0.0,
                                0.95,
                            )

                            # Combined independent ignition probability
                            p_ignite = 1.0 - (1.0 - p_ignite) * (1.0 - p_local)

                # Stochastic ignition check
                if rng.random() < p_ignite:
                    chunk_next[local_r, c] = BURNING

    return start_row, end_row, chunk_next


# ==============================================================================
# MAIN PARALLEL CELLULAR WILDFIRE SIMULATION FUNCTION
# ==============================================================================
def parallel_cellular_wildfire_simulation(
    grid_size=(50, 50),
    max_steps=40,
    elevation=None,
    wind_vector=(0.25, 0.25),
    base_prob=0.22,
    num_workers=None,
):
    rows, cols = grid_size
    num_workers = num_workers or min(4, os.cpu_count() or 1)
    rng = np.random.default_rng(42)

    # Step 1: Initialize Terrain Grid (85% FOREST, 15% EMPTY firebreaks)
    grid = rng.choice([FOREST, EMPTY], size=(rows, cols), p=[0.85, 0.15])

    # Create default uphill elevation map if none provided
    if elevation is None:
        elevation = np.tile(np.linspace(0, 100, cols), (rows, 1))

    # Set initial fire cells in Grid to BURNING
    grid[5:8, 5:8] = BURNING

    # Split rows across parallel workers
    row_splits = np.array_split(np.arange(rows), num_workers)
    row_ranges = [
        (int(s[0]), int(s[-1]) + 1) for s in row_splits if len(s) > 0
    ]

    print(
        f"Starting Parallel Cellular Simulation on {rows}x{cols} grid using"
        f" {len(row_ranges)} workers..."
    )

    # Step 2: Time-Stepped Parallel Transition Loop
    with ProcessPoolExecutor(max_workers=len(row_ranges)) as executor:
        for step in range(1, max_steps + 1):
            next_grid = grid.copy()

            tasks = [
                (
                    r_start,
                    r_end,
                    grid,
                    elevation,
                    wind_vector,
                    base_prob,
                    1000 * step + idx,
                )
                for idx, (r_start, r_end) in enumerate(row_ranges)
            ]

            for r_start, r_end, updated_chunk in executor.map(
                _update_row_slice, tasks
            ):
                next_grid[r_start:r_end] = updated_chunk

            # Synchronous Grid Update
            grid = next_grid

            burning_count = int(np.sum(grid == BURNING))
            if step % 5 == 0 or burning_count == 0 or step == 1:
                print(
                    f"Step {step:02d} | Healthy Forest: {np.sum(grid == FOREST):4d} | "
                    f"Burning: {burning_count:4d} | Burned Out: {np.sum(grid == BURNED):4d}"
                )

            # Early termination if no active fire remains
            if burning_count == 0:
                print(f"Simulation ended early at step {step}: No active fire remaining.")
                break

    return grid


# ==============================================================================
# RUN SIMULATION
# ==============================================================================
if __name__ == "__main__":
    final_grid = parallel_cellular_wildfire_simulation(
        grid_size=(50, 50),
        max_steps=40,
        wind_vector=(0.25, 0.25),  # South-East wind
        base_prob=0.22,
    )
