# Generate trajectories for Diffuser training

import numpy as np
from pathlib import Path
from tqdm import tqdm
import heapq
import argparse

from maze2d import Maze2DEnv
from dataset import TrajectoryDataset


def continuous_to_grid(pos, resolution=50):
    # Convert continuous position to grid coordinates
    x = int(np.clip(pos[0] * resolution, 0, resolution - 1))
    y = int(np.clip(pos[1] * resolution, 0, resolution - 1))
    return (x, y)


def grid_to_continuous(grid_pos, resolution=50):
    # Convert grid coordinates to continuous position.
    return np.array([(grid_pos[0] + 0.5) / resolution, (grid_pos[1] + 0.5) / resolution])


def is_valid_grid_pos(pos, env, resolution=50):
    """Check if a grid position is valid (not in wall)."""
    continuous_pos = grid_to_continuous(pos, resolution)
    # Check if position collides with any wall
    for wall in env.walls:
        x1, y1, x2, y2 = wall
        # Simple box check for wall proximity
        cx, cy = continuous_pos
        margin = 0.03

        # vertical walls
        if abs(x1 - x2) < 0.01:  
            if abs(cx - x1) < margin and min(y1, y2) - margin < cy < max(y1, y2) + margin:
                return False
        # horizontal walls
        elif abs(y1 - y2) < 0.01:
            if abs(cy - y1) < margin and min(x1, x2) - margin < cx < max(x1, x2) + margin:
                return False
    return True


def astar(start, goal, env, resolution=50):
    """
    A* pathfinding on discretized maze.

    Returns list of grid positions from start to goal.
    """
    def heuristic(a, b):    # manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 8-directional movement
    neighbors = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check bounds
            if not (0 <= neighbor[0] < resolution and 0 <= neighbor[1] < resolution):
                continue

            # Check validity
            if not is_valid_grid_pos(neighbor, env, resolution):
                continue

            # Diagonal movement costs more
            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # no path found


def generate_trajectory_from_path(path, env, horizon=128, resolution=50, kp=0.5):
    """
    Generate a smooth trajectory following a discrete path.

    Uses proportional control to follow waypoints. Returns a (horizon, transition_dim) trajectory array
    """
    # Convert path to continuous waypoints
    waypoints = [grid_to_continuous(p, resolution) for p in path]

    trajectory = []
    state = waypoints[0].copy()
    waypoint_idx = 1

    for t in range(horizon):
        if waypoint_idx < len(waypoints):
            target = waypoints[waypoint_idx]

            # P control towards waypoint
            error = target - state
            dist = np.linalg.norm(error)

            if dist < 0.02:  # Close enough to waypoint
                waypoint_idx += 1
                if waypoint_idx >= len(waypoints):
                    target = waypoints[-1]
                else:
                    target = waypoints[waypoint_idx]
                error = target - state

            # Compute action
            action = kp * error
            action = np.clip(action, -env.max_action, env.max_action)
        else:
            # At goal, zero action
            action = np.zeros(2)

        trajectory.append(np.concatenate([state, action]))

        # Update state
        next_state = state + action
        next_state = np.clip(next_state, 0.01, 0.99)
        state = next_state

    return np.array(trajectory, dtype=np.float32)


def sample_random_valid_position(env, rng, resolution=50):
    """Sample a random valid position anywhere in the maze."""
    for _ in range(1000):
        pos = rng.uniform(0.05, 0.95, size=2)
        grid_pos = continuous_to_grid(pos, resolution)
        if is_valid_grid_pos(grid_pos, env, resolution):
            return pos
    raise RuntimeError("Could not find valid position")


def generate_expert_trajectories(env, num_trajectories=1000, horizon=128, resolution=50, seed=42):
    """
    Generate expert trajectories using A* + proportional control.
    Start/goal pairs are sampled randomly anywhere in the maze.

    - In:
        - env: Maze environment
        - num_trajectories: Number of trajectories to generate
        - horizon: Trajectory length
        - resolution: Grid resolution for A*
        - seed: Random seed
    - Out:
        - (N, transition_dim, horizon) array of trajectories
    """
    rng = np.random.default_rng(seed)
    trajectories = []

    pbar = tqdm(total=num_trajectories, desc="Generating trajectories")
    attempts = 0
    max_attempts = num_trajectories * 10

    while len(trajectories) < num_trajectories and attempts < max_attempts:
        attempts += 1

        # Sample random start and goal anywhere in the maze
        env.seed(int(rng.integers(0, 1e6)))
        start = sample_random_valid_position(env, rng, resolution)
        goal = sample_random_valid_position(env, rng, resolution)

        # Ensure start and goal are sufficiently far apart
        if np.linalg.norm(start - goal) < 0.3:
            continue

        # Convert to grid
        start_grid = continuous_to_grid(start, resolution)
        goal_grid = continuous_to_grid(goal, resolution)

        # Find path with A*
        path = astar(start_grid, goal_grid, env, resolution)

        if path is None:
            continue

        # Generate smooth trajectory
        traj = generate_trajectory_from_path(path, env, horizon, resolution)

        # Verify trajectory reaches goal
        final_pos = traj[-1, :2]
        if np.linalg.norm(final_pos - goal) < 0.1:
            trajectories.append(traj)
            pbar.update(1)

    pbar.close()

    if len(trajectories) < num_trajectories:
        print(f"Warning: Only generated {len(trajectories)} trajectories")

    # Stack and transpose to (N, transition_dim, horizon)
    trajectories = np.stack(trajectories)  # (N, horizon, transition_dim)
    trajectories = trajectories.transpose(0, 2, 1)  # (N, transition_dim, horizon)

    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Generate random expert trajectories")
    parser.add_argument("--maze-type", type=str, default="umaze", choices=["umaze", "simple", "smaze"])
    parser.add_argument("--num-trajectories", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--output", type=str, default="dataset/umaze_5000.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.num_trajectories} trajectories for \"{args.maze_type}\" maze.\n")

    env = Maze2DEnv(maze_type=args.maze_type)

    trajectories = generate_expert_trajectories(
        env,
        num_trajectories=args.num_trajectories,
        horizon=args.horizon,
        seed=args.seed,
    )

    print(f"\nGenerated trajectories shape: {trajectories.shape}")
    print(f"  - Number of trajectories: {trajectories.shape[0]}")
    print(f"  - Transition dimension: {trajectories.shape[1]}")
    print(f"  - Horizon: {trajectories.shape[2]}")

    TrajectoryDataset.save(trajectories, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
