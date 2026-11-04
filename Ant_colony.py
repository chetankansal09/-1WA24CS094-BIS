import numpy as np
import matplotlib.pyplot as plt
import random
import math

# 10x10 Grid Layout (0 = Empty, 1 = Wall)
grid = np.zeros((10, 10))
grid[1, 2:6] = 1
grid[2:6, 5] = 1
grid[7, 3:8] = 1

start = (0, 0)
goal = (9, 9)

# ACO Parameters
num_ants = 15
num_iterations = 20
evaporation_rate = 0.4
alpha = 1.0  # Pheromone importance
beta = 2.0   # Distance importance

# Initialize pheromones equally across the grid
pheromones = np.ones((10, 10, 8)) 
motions = [
    (1, 0), (0, 1), (-1, 0), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

all_explored_paths = []
best_path = None
shortest_distance = float('inf')

# Run Simulation
for iteration in range(num_iterations):
    for ant in range(num_ants):
        current = start
        path = [current]
        visited = {current}
        distance = 0
        
        # Max steps to avoid infinite loops while wandering
        for _ in range(50):
            if current == goal:
                break
                
            # Calculate probabilities for next step
            valid_moves = []
            probabilities = []
            
            for idx, (dx, dy) in enumerate(motions):
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < 10 and 0 <= ny < 10 and grid[nx, ny] == 0 and (nx, ny) not in visited:
                    move_cost = math.hypot(dx, dy)
                    valid_moves.append(((nx, ny), idx, move_cost))
                    
                    # ACO Formula elements: pheromone^alpha * (1/distance)^beta
                    tau = pheromones[current[0], current[1], idx] ** alpha
                    eta = (1.0 / move_cost) ** beta
                    probabilities.append(tau * eta)
            
            if not valid_moves:
                break # Dead end
                
            # Choose move based on probability distribution
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]
            next_choice = random.choices(valid_moves, weights=probabilities)[0]
            
            current = next_choice[0]
            path.append(current)
            visited.add(current)
            distance += next_choice[2]
            
        if path[-1] == goal:
            all_explored_paths.append(path)
            if distance < shortest_distance:
                shortest_distance = distance
                best_path = path

    # Evaporate pheromones
    pheromones *= (1.0 - evaporation_rate)

# --- Plotting the Results ---
plt.figure(figsize=(7, 7))

# Draw Walls
for r in range(10):
    for c in range(10):
        if grid[r, c] == 1:
            plt.fill_between([c, c+1], r, r+1, color='black')

# 1. Plot all wandering ant tracks (Faint orange lines)
for i, path in enumerate(all_explored_paths):
    px, py = zip(*path)
    label = "Discovered Ant Trails" if i == 0 else ""
    plt.plot([y + 0.5 for y in py], [x + 0.5 for x in px], color='orange', alpha=0.25, linewidth=1, linestyle='--', label=label)

# 2. Plot final chosen optimal path (Thick blue line)
if best_path:
    bx, by = zip(*best_path)
    plt.plot([y + 0.5 for y in by], [x + 0.5 for x in bx], color='blue', linewidth=3.5, label="Final Converged Path")

plt.scatter(start[1]+0.5, start[0]+0.5, color='green', s=200, zorder=5, label="Start")
plt.scatter(goal[1]+0.5, goal[0]+0.5, color='red', s=200, zorder=5, label="Goal")

plt.xlim(0, 10)
plt.ylim(10, 0)
plt.grid(True, color='lightgray')
plt.title("Ant Colony Optimization Path Discovery")
plt.legend()
plt.show()
