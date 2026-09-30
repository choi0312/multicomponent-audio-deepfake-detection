"""Interpretable Fourier fakeprint features for AI-generated music.

Method reference: Afchar et al., "A Fourier Explanation of AI-music
Artifacts", ISMIR 2025. The selected project compared two published lower
envelope implementations instead of treating them as equivalent.

The learned 3,585-weight logistic checkpoint is not included here. See the
repository README and THIRD_PARTY_NOTICES.md for provenance and licensing.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import minimum_filter1d


def normalize_residual(residual: np.ndarray) -> np.ndarray:
    """Clip spectral peaks and normalize within one file only."""

    residual = np.clip(residual, 0.0, 5.0)
    return (residual / (1e-6 + np.max(residual))).astype(np.float32)


def publisher_residual(profile_1_to_8khz: np.ndarray) -> np.ndarray:
    """Minimum-filter variant used by the independent lofcz implementation."""

    profile = np.asarray(profile_1_to_8khz, dtype=np.float64)
    if profile.shape != (3585,):
        raise ValueError("Expected 3,585 frequency bins from 1 to 8 kHz")
    lower = minimum_filter1d(profile, size=10, mode="nearest")
    return normalize_residual(profile - np.clip(lower, -45.0, None))


def quadratic_residual(full_spectrum_profile: np.ndarray) -> np.ndarray:
    """Quadratic lower-envelope variant used by ArtifactBench.

    The input is the full one-sided 8,192-point FFT profile. Bins 512:4097 are
    the inclusive 1–8 kHz band at a 16 kHz sampling rate.
    """

    values = np.asarray(full_spectrum_profile, dtype=np.float64)[512:4097]
    if values.shape != (3585,):
        raise ValueError("Unexpected spectrum length")

    patches = np.lib.stride_tricks.sliding_window_view(values, 10)
    indices = np.unique(np.argmin(patches, axis=1) + np.arange(len(patches)))
    indices = np.unique(np.r_[0, indices, len(values) - 1])
    frequencies = np.linspace(1000.0, 8000.0, len(values))
    lower = interp1d(
        frequencies[indices], values[indices], kind="quadratic"
    )(frequencies)
    return normalize_residual(values - np.clip(lower, -45.0, None))


def logistic_probability(features: np.ndarray, weight: np.ndarray, bias: float) -> float:
    """Apply the tiny fakeprint classifier after checking its public shape."""

    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if x.shape != (3585,) or w.shape != (3585,):
        raise ValueError("Fourier detector expects 3,585 features")
    margin = float(x @ w + bias)
    return float(1.0 / (1.0 + np.exp(-np.clip(margin, -60.0, 60.0))))

