# Trajectory dataset for Diffuser training

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DatasetNormalizer():
    """
    Normalize trajectories to have zero mean and unit variance.

    Computes statistics over entire dataset for each dimension.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, trajectories):
        """Compute normalization statistics from data."""
        # trajectories: (N, transition_dim, horizon) array
        # Compute mean and std over batch and horizon dimensions
        self.mean = trajectories.mean(axis=(0, 2))
        self.std = trajectories.std(axis=(0, 2))
        # shape: (transition_dim,)

        # prevent division by zero
        self.std = np.clip(self.std, a_min=1e-6, a_max=None)

        self._fitted = True
        return self

    def normalize(self, x):
        """
        Transform data to normalized space.
        - In: 
            - x: (..., transition_dim, horizon) array
        - Out:
            - Normalized array
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        # Broadcast mean and std to match input shape
        # mean/std shape: (transition_dim,) -> (transition_dim, 1)
        return (x - self.mean[:, None]) / self.std[:, None]

    def unnormalize(self, x):
        """
        Transform data back to original space.
        - In:
            - x: (..., transition_dim, horizon) array
        - Out:
            - Unnormalized array
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        return x * self.std[:, None] + self.mean[:, None]

    def normalize_torch(self, x):
        """Normalize using torch tensors."""
        device = x.device
        mean = torch.from_numpy(self.mean).float().to(device)
        std = torch.from_numpy(self.std).float().to(device)
        return (x - mean[:, None]) / std[:, None]

    def unnormalize_torch(self, x):
        """Unnormalize using torch tensors."""
        device = x.device
        mean = torch.from_numpy(self.mean).float().to(device)
        std = torch.from_numpy(self.std).float().to(device)
        return x * std[:, None] + mean[:, None]

    def save(self, path):
        """Save normalizer to file."""
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        """Load normalizer from file."""
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        self._fitted = True
        return self


class TrajectoryDataset(Dataset):
    """
    Dataset of trajectories for diffusion model training.

    Data format (similar to equation 2 from paper):
        tau = [s0, s1, ..., sT; a0, a1, ..., aT]
        Shape: (transition_dim, horizon) = (state_dim + action_dim, T)

    For 2D maze: (4, 128) arrays where:
        - tau[0:2, :] = (x, y) positions over time
        - tau[2:4, :] = (dx, dy) velocities over time
    """

    def __init__(self, trajectories, normalizer=None):
        """
        - In:
            - trajectories: (N, transition_dim, horizon) array
            - normalizer: optional normalizer (will fit if not already fitted)
        """
        self.trajectories = trajectories.astype(np.float32)

        # Set up normalizer and fit
        if normalizer is None:
            normalizer = DatasetNormalizer()

        if not normalizer._fitted:
            normalizer.fit(self.trajectories)

        self.normalizer = normalizer

        # Normalize data
        self.normalized_trajectories = normalizer.normalize(self.trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Return normalized trajectory as tensor
        return torch.from_numpy(self.normalized_trajectories[idx])

    def get_raw(self, idx):
        """Return unnormalized trajectory."""
        return self.trajectories[idx]

    @classmethod
    def load(cls, path, normalizer=None):
        """
        Load dataset from .npz file.
        -In :
            - path: path to .npz file containing "trajectories" key
            - normalizer: optional pre-fitted normalizer
        - Out:
            - TrajectoryDataset instance
        """
        data = np.load(path)
        trajectories = data["trajectories"]
        return cls(trajectories, normalizer)

    @staticmethod
    def save(trajectories, path):
        """
        Save trajectories to .npz file.
        - In:
            - trajectories: (N, transition_dim, horizon) array
            - path: output path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, trajectories=trajectories)
