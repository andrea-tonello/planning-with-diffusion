# Gaussian diffusion process for trajectory planning.

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for trajectory planning.

    Implements:
        - Forward process: q(x_t | x_0) - adding noise to data
        - Reverse process: p(x_{t-1} | x_t) - denoising
        - Training loss: simplified epsilon-prediction objective
        - Conditional sampling via inpainting
    """

    def __init__(self, model, schedule, state_dim=2, action_dim=2):
        """
        - In:
            - model: Denoising network (TemporalUNet)
            - schedule: Noise schedule
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
        """
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_timesteps = schedule.num_timesteps

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: sample x_t given x_0.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        - In :
            - x_0: (batch, transition_dim, horizon) clean trajectory
            - t: (batch,) timesteps
            - noise: Optional pre-sampled noise
        - Out:
            - (batch, transition_dim, horizon) noisy trajectory
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.schedule.get_schedule_values(t, self.schedule.sqrt_alphas_cumprod)
        sqrt_one_minus_alpha_bar = self.schedule.get_schedule_values(t, self.schedule.sqrt_one_minus_alphas_cumprod)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def p_mean_variance(self, x_t, t):
        """
        Compute mean and variance for reverse process step.

        Uses epsilon-prediction parameterization:
            mu = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_pred)

        - In:
            - x_t: (batch, transition_dim, horizon) noisy trajectory
            - t: (batch,) timesteps
        - Out:
            - mean: (batch, transition_dim, horizon)
            - variance: (batch, 1, 1) for broadcasting
        """
        # Predict noise
        epsilon_pred = self.model(x_t, t)

        # Get schedule values
        sqrt_recip_alpha = self.schedule.get_schedule_values(t, self.schedule.sqrt_recip_alphas) # 1/sq(a_t)
        beta = self.schedule.get_schedule_values(t, self.schedule.betas) # b_t
        sqrt_one_minus_alpha_bar = self.schedule.get_schedule_values(t, self.schedule.sqrt_one_minus_alphas_cumprod) # sq(1-a_bar_t)
        posterior_variance = self.schedule.get_schedule_values(t, self.schedule.posterior_variance)

        # Compute mean
        mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha_bar * epsilon_pred)

        return mean, posterior_variance

    def p_sample(self, x_t, t):
        """
        Single reverse diffusion step: sample x_{t-1} given x_t,
        from the gaussian x_{t-1} ~ N( mu_theta(x_t), sigma^2_theta(x_t)*I )

        - In:
            - x_t: (batch, transition_dim, horizon) noisy trajectory
            - t: (batch,) timesteps
        - Out:
            - (batch, transition_dim, horizon) less noisy trajectory
        """
        mean, variance = self.p_mean_variance(x_t, t)

        # Add noise except at t=0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1)

        return mean + nonzero_mask * torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, start_state=None, goal_state=None, device=torch.device("cpu"), inpaint_boundary_steps=4):
        """
        Main inference loop, reverse process. Generate trajectory from noise.

        Applies inpainting at each step to condition on start/goal.

        - In:
            - shape: (batch, transition_dim, horizon) output shape
            - start_state: (batch, state_dim) start state for conditioning
            - goal_state: (batch, state_dim) goal state for conditioning
            - device: device to run on
            - inpaint_boundary_steps: number of timesteps at each boundary
              to apply soft inpainting
        - Out:
            - (batch, transition_dim, horizon) generated trajectory
        """
        batch_size, transition_dim, horizon = shape

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion
        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # Denoise
            x = self.p_sample(x, t)

            # Here we tried to avoid sharp turns near start and goal by "spreading" inpainting near start/goal
            # So for pos 0 -> 100% constraint, pos1 --> 75% constraint 25% model pred, pos2 --> 50% constraint 50% model pred...
            if start_state is not None:
                # hard constraint at t=0
                x[:, :self.state_dim, 0] = start_state
                # soft constraints, blend for nearby timesteps
                for k in range(1, min(inpaint_boundary_steps, horizon)):
                    weight = 1.0 - (k / inpaint_boundary_steps)
                    x[:, :self.state_dim, k] = weight * start_state + (1 - weight) * x[:, :self.state_dim, k]

            if goal_state is not None:
                # hard constraint at t=-1 (127)
                x[:, :self.state_dim, -1] = goal_state
                # soft constraints, blend for nearby timesteps
                for k in range(1, min(inpaint_boundary_steps, horizon)):
                    weight = 1.0 - (k / inpaint_boundary_steps)
                    x[:, :self.state_dim, -(k + 1)] = weight * goal_state + (1 - weight) * x[:, :self.state_dim, -(k + 1)]

        return x

    def loss(self, x_0):
        """
        Standard MSE-like loss between true epsilon / pred epsilon

        L = E_{t, epsilon} [ ||epsilon - epsilon_theta(x_t, t)||^2 ]

        - In:
            - x_0: (batch, transition_dim, horizon) clean trajectory
            - t: (batch,) timesteps (sampled uniformly if not provided)
        - Out:
            - Scalar loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps to check and noise
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x_0)

        # Sample noisy trajectory and predict noise
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.model(x_t, t)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def plan(self, start_state, goal_state, horizon=128, batch_size=1, inpaint_boundary_steps=4):
        """
        Generate a planned trajectory from start to goal.

        - In:
            - start_state: (state_dim,) or (batch, state_dim) start state
            - goal_state: (state_dim,) or (batch, state_dim) goal state
            - horizon: planning horizon
            - batch_size: number of trajectories to sample
            - inpaint_boundary_steps: number of timesteps at each boundary
              to apply soft inpainting
        - Out:
            - trajectory: (batch, horizon, transition_dim) planned trajectory
        """
        device = next(self.model.parameters()).device

        # If I pass a 1D tesnor with no batch dimensions (e.g. only a point (x,y)), add it: (2,) -> (1, 2)
        if start_state.dim() == 1:
            start_state = start_state.unsqueeze(0).expand(batch_size, -1)
        if goal_state.dim() == 1:
            goal_state = goal_state.unsqueeze(0).expand(batch_size, -1)

        start_state = start_state.to(device)
        goal_state = goal_state.to(device)

        transition_dim = self.state_dim + self.action_dim
        shape = (batch_size, transition_dim, horizon)

        # Generate trajectory
        result = self.p_sample_loop(
            shape,
            start_state=start_state,
            goal_state=goal_state,
            device=device,
            inpaint_boundary_steps=inpaint_boundary_steps,
        )

        # Transpose to (batch, horizon, transition_dim)
        return result.permute(0, 2, 1)
