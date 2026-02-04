# Temporal U-Net for trajectory denoising.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    # x * tanh(softplus(x))
    # preferred over ReLU in diffusion models because it is smoother and non-monotonic.

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# There are two different notions of "time":
# 1) Planning horizon     t   in {0, 1, ..., T}: the position within the trajectory
# 2) Diffusion timestep   i   in {0, 1, ..., N}: which step of the denoising process we are at

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for diffusion timesteps (i) encoding.

    The diffusion timestep i is an integer from 0 to N (e.g., 0 to 99). 
    We need to convert this scalar into something the network can extract information from

    -> a vector of sine and cosine pairs at different frequencies
        - Low-frequency pairs encode coarse position
        - High-frequency pairs encode fine position
    """

    def __init__(self, dim):
        """ dim: embedding dimension (should be even) """
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        - In:
            - timesteps: (batch_size,) tensor of integer timesteps
        - Out:
            - (batch_size, dim) embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2    # half sin, half cos

        emb_scale = math.log(10000) / (half_dim - 1)
        # ~0.613 for half_dim=16

        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        # emb = [1.0, 0.54, 0.29, 0.16, ..., 0.0001] — geometric sequence from 1 to 1/10000

        emb = timesteps[:, None].float() * emb[None, :]
        # Shape: (batch, 1) * (1, half_dim) → (batch, half_dim)
        # Each row is [i*1.0, i*0.54, i*0.29, ...] — the timestep scaled by each frequency

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # Shape: (batch, dim) -> concatenate sin and cos

        return emb


class TimestepMLP(nn.Module):
    """MLP to project timestep embeddings to match hidden dimensions."""

    def __init__(self, embed_dim, hidden_dim):
        """
        - In:
            - embed_dim: input embedding dimension
            - hidden_dim: output dimension (to match model channels)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish"""

    def __init__(self, in_channels, out_channels, kernel_size=5, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = Mish()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualTemporalBlock(nn.Module):
    """
    Residual block with two 1D temporal convolutions.

    Conv1d -> GroupNorm -> Mish -> Conv1d -> GroupNorm -> Mish
        + Timestep embedding added after first conv
        + Skip connection (with optional projection)
    """

    def __init__(self, in_channels, out_channels, embed_dim, kernel_size=5, groups=8):
        """
        - In:
            - in_channels: input channels
            - out_channels: output channels
            - embed_dim: timestep embedding dimension
            - kernel_size: temporal conv kernel size
            - groups: GroupNorm groups
        """
        super().__init__()

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, groups)

        # Project timestep embedding to match channels
        self.time_mlp = nn.Sequential(
            Mish(),
            nn.Linear(embed_dim, out_channels),
        )

        # Skip connection projection if dimensions don't match
        self.skip_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        """
        - In:
            - x: (batch, channels, horizon) input
            - t_emb: (batch, embed_dim) timestep embedding
        - Out:
            - (batch, out_channels, horizon) output
        """
        h = self.conv1(x)

        # Add timestep embedding (broadcast over horizon dimension)
        h = h + self.time_mlp(t_emb)[:, :, None]

        h = self.conv2(h)

        # Skip connection
        return h + self.skip_conv(x)


class Downsample1d(nn.Module):
    """Downsampling with strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsampling with transposed convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class TemporalUNet(nn.Module):
    """
    U-Net architecture for trajectory denoising.

    Architecture (from paper Appendix C):
        - Repeated residual blocks (3 down, 3 up)
        - Downsampling via strided conv (stride=2)
        - Upsampling via transposed conv
        - Skip connections between encoder/decoder
        - Fully convolutional in horizon dimension

    The model predicts noise epsilon for the diffusion process.
    """

    def __init__(self, transition_dim=4, dim=32, dim_mults=[1, 2, 4], kernel_size=5, groups=8):
        """
        - In:
            - transition_dim: state_dim + action_dim (trajectory width)
            - dim: base channel dimension
            - dim_mults: channel multipliers per level
            - kernel_size: temporal conv kernel size
            - groups: GroupNorm groups
        """
        super().__init__()

        self.transition_dim = transition_dim
        self.dim = dim
        self.dim_mults = dim_mults

        # Timestep embedding
        time_dim = dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            TimestepMLP(dim, time_dim),
        )

        # Initial projection from transition_dim to model dimension
        self.init_conv = nn.Conv1d(transition_dim, dim, kernel_size, padding=kernel_size // 2)  # 4 -> 32

        # Compute channel dimensions at each level
        dims = [dim] + [dim * m for m in dim_mults]     # [32, 64, 128]
        in_out = list(zip(dims[:-1], dims[1:]))         # [(32, 64), (64, 128)]

        # 2 encoder levels
        # Encoder
        self.downs = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(
                nn.ModuleList([
                    ResidualTemporalBlock(in_ch, out_ch, time_dim, kernel_size, groups),
                    ResidualTemporalBlock(out_ch, out_ch, time_dim, kernel_size, groups),
                    Downsample1d(out_ch) if not is_last else nn.Identity(),
                ])
            )

        # 2 bottleneck levels
        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim, kernel_size, groups)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim, kernel_size, groups)

        # 2 decoder levels
        # Decoder
        self.ups = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(reversed(in_out)):    # [(64, 128), (32, 64)]
            is_last = i == len(in_out) - 1
            self.ups.append(
                nn.ModuleList([
                    # Input channels doubled due to skip connection
                    ResidualTemporalBlock(out_ch * 2, in_ch, time_dim, kernel_size, groups),
                    ResidualTemporalBlock(in_ch, in_ch, time_dim, kernel_size, groups),
                    Upsample1d(in_ch) if not is_last else nn.Identity(),
                ])
            )

        # Final projection back to transition_dim
        self.final_conv = nn.Sequential(
            # 32 -> 32, to refine features
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, dim),
            nn.Mish(),

            # 32 -> 4, 1x1 projection
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, timestep):
        """
        Predict noise for denoising.
        - In:
            - x: (batch, transition_dim, horizon) noisy trajectory i
            - timestep: (batch,) diffusion step i
        - Out:
            - (batch, transition_dim, horizon) predicted noise
        """
        # Timestep embedding
        t_emb = self.time_embed(timestep)

        # Initial projection
        x = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for res1, res2, downsample in self.downs:
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            skips.append(x)     # save before downasampling
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        # Decoder with skip connections
        for res1, res2, upsample in self.ups:
            skip = skips.pop()
            # potential size mismatch from downsampling from padding
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = upsample(x)

        # Final projection
        return self.final_conv(x)
