import math
import torch


class CosineNoiseSchedule:
    """
    Cosine noise schedule (instead of linear which causes problems).

    Computes:
        alpha_bar(t) = f(t) / f(0)
        where f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
        s = 0.008 (small offset to prevent beta_t from being too small)

    This schedule provides a more gradual noise addition compared to linear,
    which helps preserve structure longer during the forward process.
    """

    def __init__(self, num_timesteps=100, s=0.008):
        """
        - In:
            num_timesteps: number of diffusion steps (N)
            s: small offset for numerical stability
        """
        self.num_timesteps = num_timesteps
        self.s = s

        # Compute schedule values
        self._compute_schedule()

    def _compute_schedule(self):
        """Precompute all schedule values."""
        T = self.num_timesteps

        # Compute alpha_bar (cumulative product of alphas) using cosine schedule
        steps = torch.linspace(0, T, T + 1, dtype=torch.float64)
        alpha_bar = self._alpha_bar_fn(steps / T)

        # Derive betas as beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        # Compute alphas
        alphas = 1.0 - betas

        # Cumulative products
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])

        # Precompute values for forward process q(x_t | x_0)
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Precompute values for reverse process p(x_{t-1} | x_t)
        # Posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        # For computing mean of reverse process
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        #posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        #posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        # Store as float32 tensors
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.float()
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.float()
        self.posterior_variance = posterior_variance.float()
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20)).float()
        self.sqrt_recip_alphas = sqrt_recip_alphas.float()
        #self.posterior_mean_coef1 = posterior_mean_coef1.float()
        #self.posterior_mean_coef2 = posterior_mean_coef2.float()

    def _alpha_bar_fn(self, t):
        """Compute alpha_bar using cosine schedule."""
        return torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2

    def get_schedule_values(self, t, values):
        """
        Extract schedule values for given timesteps.
        - In:
            - t: (batch_size,) timesteps
            - values: (num_timesteps,) schedule values
        - Out:
            - (batch_size, 1, 1) values for broadcasting with (batch, channels, horizon)
        """
        device = t.device
        values = values.to(device)
        out = values.gather(0, t)
        return out.view(-1, 1, 1)  # For broadcasting with trajectory tensor
