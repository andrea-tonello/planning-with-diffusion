import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


def plot_maze(walls, start_region=None, goal_region=None, ax=None, figsize=(8, 8)):
    """
    Plot the maze layout.
    - In:
        - walls: list of wall segments (x1, y1, x2, y2)
        - start_region: (x_min, y_min, x_max, y_max) start region
        - goal_region: (x_min, y_min, x_max, y_max) goal region
        - ax: optional existing axes
        - figsize: figure size if creating new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot walls
    for wall in walls:
        x1, y1, x2, y2 = wall
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=2)

    # Plot start region
    if start_region is not None:
        x_min, y_min, x_max, y_max = start_region
        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=True, facecolor="lightgreen", edgecolor="green", alpha=0.5, label="Start",
        )
        ax.add_patch(rect)

    # Plot goal region
    if goal_region is not None:
        x_min, y_min, x_max, y_max = goal_region
        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=True, facecolor="lightcoral", edgecolor="red", alpha=0.5, label="Goal",
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return ax


def plot_trajectory(trajectory, state_dim=2, ax=None, color="blue", alpha=0.7, show_endpoints=True, label=None):
    """
    - In:
        - trajectory: (horizon, transition_dim) or (transition_dim, horizon) array
        - state_dim: number of state dimensions
        - ax: optional existing axes
        - color: line color
        - alpha: transparency
        - show_endpoints: Whether to mark start and end points
        - label: optional label for legend
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Handle both (horizon, dim) and (dim, horizon) formats
    if trajectory.shape[0] == state_dim or trajectory.shape[0] == state_dim * 2:
        # Shape is (transition_dim, horizon), transpose
        trajectory = trajectory.T

    # Convert torch tensor if needed
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()

    # Extract positions
    positions = trajectory[:, :state_dim]

    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=alpha, linewidth=1.5, label=label)

    if show_endpoints:
        # start (circle)
        ax.scatter(positions[0, 0], positions[0, 1], c=color, s=100, marker="o", edgecolors="black", zorder=5)
        # end (star)
        ax.scatter(positions[-1, 0], positions[-1, 1], c=color, s=150, marker="*", edgecolors="black", zorder=5)

    return ax


def plot_diffusion_process(intermediates, state_dim=2, num_steps=6, walls=None, figsize=(20, 4)):
    """
    Visualize intermediate denoising steps.

    - In:
        - intermediates: list of trajectory arrays at each denoising step.
          Each element is (transition_dim, horizon) or (horizon, transition_dim).
        - state_dim: number of state dimensions (should always be 2 anyway)
        - num_steps: number of intermediate steps to show
        - walls: optional list of wall segments (x1, y1, x2, y2) to overlay on each subplot
        - figsize: figure size
    """
    fig, axes = plt.subplots(1, num_steps, figsize=figsize)

    # Evenly spaced steps across the full denoising process
    total_steps = len(intermediates)
    indices = np.linspace(0, total_steps - 1, num_steps, dtype=int)

    for ax, idx in zip(axes, indices):
        traj = intermediates[idx]
        if isinstance(traj, torch.Tensor):
            traj = traj.cpu().numpy()

        # Handle batch dimension
        if traj.ndim == 3:
            traj = traj[0]  # take first sample

        if walls is not None:
            plot_maze(walls, ax=ax)

        plot_trajectory(traj, state_dim=state_dim, ax=ax, show_endpoints=True)

        ax.set_title(f"Step {idx}/{total_steps - 1}")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")

    plt.tight_layout()
    return fig


