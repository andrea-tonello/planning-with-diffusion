import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from models.temporal_unet import TemporalUNet
from models.scheduler import CosineNoiseSchedule
from models.gaussian_diffusion import GaussianDiffusion
from utils.trajectory import DatasetNormalizer
from maze2d import Maze2DEnv
from utils.visualization import plot_maze, plot_trajectory, plot_diffusion_process


def load_model(checkpoint_path, device, diffusion_steps=100):
    # Load trained model from checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dim = checkpoint.get("dim", 32)
    dim_mults = checkpoint.get("dim_mults", [1, 2, 4])
    model = TemporalUNet(transition_dim=4, dim=dim, dim_mults=dim_mults)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    normalizer = DatasetNormalizer()
    normalizer.mean = checkpoint["normalizer_mean"]
    normalizer.std = checkpoint["normalizer_std"]
    normalizer._fitted = True

    schedule = CosineNoiseSchedule(num_timesteps=diffusion_steps)
    diffusion = GaussianDiffusion(model=model, schedule=schedule, state_dim=2, action_dim=2)
    diffusion = diffusion.to(device)

    return model, diffusion, normalizer


def check_trajectory_valid(trajectory, walls):
    """
    Check if a trajectory crosses any walls.

    - In:
        - trajectory: (horizon, 4) array with states in first 2 columns
        - walls: list of wall segments (x1, y1, x2, y2)
    - Out:
        - True if trajectory is valid (no wall crossings)
    """
    def segments_intersect(p1, p2, p3, p4):
        # Check if segment (p1,p2) intersects segment (p3,p4)
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    # Check each consecutive pair of states
    for t in range(len(trajectory) - 1):
        p1 = trajectory[t, :2]
        p2 = trajectory[t + 1, :2]

        for wall in walls:
            x1, y1, x2, y2 = wall
            p3 = np.array([x1, y1])
            p4 = np.array([x2, y2])

            if segments_intersect(p1, p2, p3, p4):
                return False

    return True


def plan_trajectory(diffusion, normalizer, start, goal, horizon=128, device=torch.device("cpu"),
                    inpaint_boundary_steps=4, walls=None, num_samples=8):
    """
    Generate a planned trajectory from start to goal using rejection sampling.

    - In:
        - diffusion: trained diffusion model
        - normalizer: dataset normalizer
        - start: start position (x, y)
        - goal: goal position (x, y)
        - horizon: planning horizon
        - device: device
        - inpaint_boundary_steps: number of timesteps at each boundary
          to apply soft inpainting
        - walls: list of wall segments for validity checking
        - num_samples: number of samples to generate for rejection sampling
    - Out:
        - (horizon, 4) trajectory array [states, actions]
    """
    # Normalize start and goal
    start_norm = (start - normalizer.mean[:2]) / normalizer.std[:2]
    goal_norm = (goal - normalizer.mean[:2]) / normalizer.std[:2]

    start_tensor = torch.from_numpy(start_norm).float().to(device)
    goal_tensor = torch.from_numpy(goal_norm).float().to(device)

    # Generate multiple trajectory samples
    with torch.no_grad():
        trajectories = diffusion.plan(
            start_state=start_tensor,
            goal_state=goal_tensor,
            horizon=horizon,
            batch_size=num_samples,
            inpaint_boundary_steps=inpaint_boundary_steps,
        )

    # trajectories shape: (num_samples, horizon, transition_dim)
    trajectories = trajectories.cpu().numpy()

    # Unnormalize all samples
    unnormed_trajectories = []
    for i in range(num_samples):
        traj = trajectories[i].T  # (transition_dim, horizon)
        traj = normalizer.unnormalize(traj)
        traj = traj.T  # (horizon, transition_dim)
        unnormed_trajectories.append(traj)

    # If walls provided, do rejection sampling
    if walls is not None:
        for traj in unnormed_trajectories:
            if check_trajectory_valid(traj, walls):
                return traj
        # no valid trajectory found, return the first one anyway

    return unnormed_trajectories[0]


def execute_trajectory(env, trajectory, start, goal, use_planned_endpoint=False):
    """
    Execute a planned trajectory in the environment.

    - In:
        - env: maze environment
        - trajectory: (horizon, 4) planned trajectory
        - start: start position
        - goal: goal position
        - use_planned_endpoint: if True, measure success from planned states rather than executed states
    - Out:
        - Dictionary with execution results
    """
    env.reset(start=start, goal=goal)

    total_reward = 0
    states_visited = [env.state.copy()]
    done = False

    for t in range(len(trajectory)):
        # Use planned states as waypoints, compute action to track them
        target_state = trajectory[t, :2]  # target position
        error = target_state - env.state
        # Proportional controller with gain that respects max_action
        action = np.clip(error, -env.max_action, env.max_action)

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        states_visited.append(next_state.copy())

        if done:
            break

    # Distance from executed endpoint to goal
    executed_distance = np.linalg.norm(env.state - goal)
    # Distance from planned endpoint to goal
    planned_endpoint = trajectory[-1, :2]
    planned_distance = np.linalg.norm(planned_endpoint - goal)

    # Choose which metric to use for success
    if use_planned_endpoint:
        final_distance = planned_distance
    else:
        final_distance = executed_distance

    success = final_distance < env.goal_threshold

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": t + 1,
        "final_distance": final_distance,
        "executed_distance": executed_distance,
        "planned_distance": planned_distance,
        "states_visited": np.array(states_visited),
    }


def sample_random_valid_position(env, rng):
    """Sample a random valid position anywhere in the maze, used with --random-goals flag"""
    for _ in range(1000):
        pos = rng.uniform(0.05, 0.95, size=2).astype(np.float32)
        # Validity check, not too close to walls
        valid = True
        for wall in env.walls:
            x1, y1, x2, y2 = wall
            # Check distance to wall segment
            if abs(x1 - x2) < 0.01:  # vertical wall
                if abs(pos[0] - x1) < 0.05 and min(y1, y2) - 0.02 < pos[1] < max(y1, y2) + 0.02:
                    valid = False
                    break
            elif abs(y1 - y2) < 0.01:  # horizontal wall
                if abs(pos[1] - y1) < 0.05 and min(x1, x2) - 0.02 < pos[0] < max(x1, x2) + 0.02:
                    valid = False
                    break
        if valid:
            return pos


def evaluate(diffusion, normalizer, env, num_episodes=10, horizon=128, device=torch.device("cpu"),
             output_dir="results", use_planned_endpoint=True, inpaint_boundary_steps=4, random_goals=False
    ):
    """
    Evaluate the model on multiple episodes.

    - In:
        - diffusion: trained diffusion model
        - normalizer: dataset normalizer
        - env: maze environment
        - num_episodes: number of evaluation episodes
        - horizon: planning horizon
        - device: device
        - output_dir: output directory for visualizations
        - use_planned_endpoint: whether to use planned or executed endpoints
        - inpaint_boundary_steps: number of timesteps at each boundary
          to apply soft inpainting
        - random_goals: whether to sample from anywhere in the maze (for generalization)
    - Out:
        - Dictionary with evaluation metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    all_trajectories = []

    rng = np.random.default_rng(42)

    for episode in range(num_episodes):
        # Sample start and goal

        if random_goals: # sample from anywhere in the maze (to test generalization)
            start = sample_random_valid_position(env, rng)
            goal = sample_random_valid_position(env, rng)
            # Ensure start and goal are far apart enough
            while np.linalg.norm(start - goal) < 0.3:
                goal = sample_random_valid_position(env, rng)
        else:
            # Sample from predefined regions
            start = env._sample_from_region(env.start_region)
            goal = env._sample_from_region(env.goal_region)

        # Plan trajectory with rejection sampling
        planned_traj = plan_trajectory(
            diffusion, normalizer, start, goal, horizon, device,
            inpaint_boundary_steps=inpaint_boundary_steps,
            walls=env.walls,
            num_samples=16,  # generate 16 samples, keep first valid
        )
        all_trajectories.append((planned_traj, start, goal))

        # Execute
        result = execute_trajectory(env, planned_traj, start, goal, use_planned_endpoint)
        result["episode"] = episode
        results.append(result)
        all_trajectories[-1] = (planned_traj, start, goal, result["states_visited"]) # store for visualization

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Success={result['success']}, "
              f"Steps={result['steps']}, "
              f"Planned dist={result['planned_distance']:.3f}, "
              f"Executed dist={result['executed_distance']:.3f}")

    # Metrics
    successes = [r["success"] for r in results]
    planned_distances = [r["planned_distance"] for r in results]
    executed_distances = [r["executed_distance"] for r in results]

    metrics = {
        "success_rate": np.mean(successes),
        "mean_planned_distance": np.mean(planned_distances),
        "std_planned_distance": np.std(planned_distances),
        "mean_executed_distance": np.mean(executed_distances),
        "std_executed_distance": np.std(executed_distances),
        "num_episodes": num_episodes,
        "use_planned_endpoint": use_planned_endpoint,
    }

    print(f"\n{'='*60}")
    print(f"Evaluation results ({num_episodes} episodes):")
    print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Success metric: {'planned' if use_planned_endpoint else 'executed'} endpoint")
    print(f"  Mean planned distance:  {metrics['mean_planned_distance']:.4f} +/- {metrics['std_planned_distance']:.4f}")
    print(f"  Mean executed distance: {metrics['mean_executed_distance']:.3f} +/- {metrics['std_executed_distance']:.3f}")
    print(f"{'='*60}\n")

    # Visualizations
    
    # If --random-goals, don't show the bigger region boxes
    start_region = None if random_goals else env.start_region
    goal_region = None if random_goals else env.goal_region

    # 1) Plot all trajectories on one map
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_maze(env.walls, start_region, goal_region, ax=ax)

    for i, (traj, start, goal, executed_states) in enumerate(all_trajectories):
        color = "green" if results[i]["success"] else "red"
        # Plot planned trajectory (solid line)
        plot_trajectory(traj, state_dim=2, ax=ax, color=color, alpha=0.5, show_endpoints=False)

        # Show executed trajectory only when --use-executed-trajectory is passed
        if not use_planned_endpoint:
            # Plot executed trajectory
            ax.plot(executed_states[:, 0], executed_states[:, 1], color=color, linestyle="--", alpha=0.3, linewidth=1)
            # Mark executed endpoint with X
            ax.scatter(executed_states[-1, 0], executed_states[-1, 1], c=color, s=30, marker="x", zorder=6, alpha=0.7)

        ax.scatter(start[0], start[1], c="blue", s=50, marker="o", zorder=5)
        ax.scatter(goal[0], goal[1], c="orange", s=50, marker="*", zorder=5)
        # Draw goal threshold circle
        circle = Circle((goal[0], goal[1]), env.goal_threshold, fill=False, edgecolor="orange", linestyle="--", alpha=0.5)
        ax.add_patch(circle)

    title = f"Trajectories (success rate: {metrics['success_rate']*100:.1f}%)"
    title += f" - {'Planned' if use_planned_endpoint else 'Executed'} endpoint"
    if random_goals:
        title += " [Random goals]"
    ax.set_title(title)
    plt.savefig(output_path / "all_trajectories.png", dpi=150, bbox_inches="tight")
    print(f"Saved trajectory visualization to {output_path / 'all_trajectories.png'}")


    # 2) Plot six individual examples
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:min(3, len(all_trajectories))]):
        traj, start, goal, executed_states = all_trajectories[i]
        plot_maze(env.walls, start_region, goal_region, ax=ax)
        color = "green" if results[i]["success"] else "red"
        # Plot planned trajectory (solid line)
        plot_trajectory(traj, state_dim=2, ax=ax, color=color, show_endpoints=False, label="Planned")

        # Show executed trajectory only when --use-executed-trajectory is passed
        if not use_planned_endpoint:
            # Plot executed trajectory
            ax.plot(executed_states[:, 0], executed_states[:, 1], color="purple", linestyle="--", alpha=0.7, linewidth=1.5, label="Executed")
            # Mark executed endpoint with X
            ax.scatter(executed_states[-1, 0], executed_states[-1, 1], c="purple", s=100, marker="x", zorder=6, linewidths=2, label="Executed end")

        ax.scatter(start[0], start[1], c="blue", s=100, marker="o", zorder=5, label="Start")
        ax.scatter(goal[0], goal[1], c="orange", s=100, marker="*", zorder=5, label="Goal")
        # Draw goal threshold circle
        circle = Circle((goal[0], goal[1]), env.goal_threshold, fill=False, edgecolor="orange", linestyle="--", linewidth=2, zorder=4)
        ax.add_patch(circle)
        # Show distances in title
        planned_dist = results[i]["planned_distance"]
        status = "Success" if results[i]["success"] else "Fail"
        if use_planned_endpoint:
            ax.set_title(f"Ep {i+1}: {status}\nPlanned dist: {planned_dist:.3f}")
        else:
            executed_dist = results[i]["executed_distance"]
            ax.set_title(f"Ep {i+1}: {status}\nPlanned: {planned_dist:.3f}, Executed: {executed_dist:.3f}")
        ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path / "individual_trajectories.png", dpi=150, bbox_inches="tight")
    print(f"Saved individual examples to {output_path / 'individual_trajectories.png'}")


    # 3) Diffusion process visualization
    # Use the first episode
    _, start, goal, _ = all_trajectories[0]
    start_norm = (start - normalizer.mean[:2]) / normalizer.std[:2]
    goal_norm = (goal - normalizer.mean[:2]) / normalizer.std[:2]
    start_tensor = torch.from_numpy(start_norm).float().to(device)
    goal_tensor = torch.from_numpy(goal_norm).float().to(device)
    # Plan with intermediates (single sample)
    trajectory, intermediates = diffusion.plan(
        start_state=start_tensor,
        goal_state=goal_tensor,
        horizon=horizon,
        batch_size=1,
        inpaint_boundary_steps=inpaint_boundary_steps,
        return_intermediates=True,
    )
    # Unnormalize each intermediate: each is (1, horizon, transition_dim)
    unnormed_intermediates = []
    for inter in intermediates:
        traj_np = inter[0].cpu().numpy()  # (horizon, transition_dim)
        traj_np = normalizer.unnormalize(traj_np.T).T  # unnormalize expects (transition_dim, horizon)
        unnormed_intermediates.append(traj_np)

    fig = plot_diffusion_process(unnormed_intermediates, state_dim=2, num_steps=6, walls=env.walls)
    fig.suptitle("Denoising process intermediate steps", y=1.02)
    fig.savefig(output_path / "diffusion_process.png", dpi=150, bbox_inches="tight")
    print(f"Saved diffusion process visualization to {output_path / 'diffusion_process.png'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffuser model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt")
    parser.add_argument("--maze-type", type=str, default="umaze", choices=["umaze", "umaze-inv", "smaze", "smaze-inv", "simple"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-executed-endpoint", action="store_true",
                        help="Measure success from executed (not planned) endpoint")
    parser.add_argument("--inpaint-boundary-steps", type=int, default=4,
                        help="Timesteps at each boundary for soft inpainting (default: 4)")
    parser.add_argument("--random-goals", action="store_true",
                        help="Sample start/goal from anywhere in maze (tests generalization)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, diffusion, normalizer = load_model(args.checkpoint, device, args.diffusion_steps)
    print(f"Model loaded successfully")

    # Create env
    env = Maze2DEnv(maze_type=args.maze_type)

    metrics = evaluate(
        diffusion=diffusion,
        normalizer=normalizer,
        env=env,
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        device=device,
        output_dir=args.output_dir,
        use_planned_endpoint=not args.use_executed_endpoint,
        inpaint_boundary_steps=args.inpaint_boundary_steps,
        random_goals=args.random_goals
    )


if __name__ == "__main__":
    main()
