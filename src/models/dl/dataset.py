import numpy as np
import random
import torch
from torch.utils.data import Dataset


def temporal_jitter(x, max_shift=10):
    """Apply temporal jitter by rolling the time axis."""
    return np.roll(x, random.randint(-max_shift, max_shift), axis=-1)


def channel_dropout(x, p_drop=0.15):
    """Apply channel dropout by setting random channels to zero."""
    mask = np.random.binomial(1, 1 - p_drop, size=(x.shape[0], 1)).astype(np.float32)
    return x * mask


def gaussian_noise(x, sigma=0.04):
    """Add Gaussian noise to the signal."""
    return x + np.random.randn(*x.shape).astype(np.float32) * sigma


def amplitude_scale(x, lo=0.85, hi=1.15):
    """Scale the amplitude by a random factor."""
    return x * random.uniform(lo, hi)


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data with optional augmentation."""

    def __init__(self, x, y, augment=False):
        """
        Initialize the EEGDataset.

        Args:
            x: numpy array of shape (N, channels, time).
            y: numpy array of shape (N,) containing class labels.
            augment: bool, whether to apply augmentations.
        """
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: index of the sample.

        Returns:
            A tuple of (x_tensor, y_tensor) where x_tensor is a FloatTensor of shape (channels, time)
            and y_tensor is an int64 scalar tensor.
        """
        w = self.x[idx].copy()
        if self.augment:
            w = temporal_jitter(w)
            w = channel_dropout(w)
            w = gaussian_noise(w)
            w = amplitude_scale(w)
        return torch.from_numpy(w), torch.tensor(self.y[idx])
