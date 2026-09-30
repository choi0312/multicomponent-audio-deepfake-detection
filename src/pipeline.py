"""Readable inference orchestration for the documented architecture.

This file intentionally depends on small protocols instead of bundling model
weights. It documents how raw/stem evidence, context, fusion, and the CPS freeze
fit together in the production system.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .fusion import finalize_prediction


class PreviousModel(Protocol):
    def predict_with_details(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Return five v2 outputs and the frozen expert summary."""


class Separator(Protocol):
    def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return vocal and accompaniment-residual waveforms."""


class Expert(Protocol):
    def summarize(self, audio: np.ndarray, prefix: str) -> Mapping[str, float]:
        """Return deterministic per-file feature summaries."""


class FusionModel(Protocol):
    def predict(
        self, scores: Mapping[str, float], previous: np.ndarray
    ) -> np.ndarray:
        """Return five values; the last two are overwritten by the CPS freeze."""


@dataclass
class ComponentPipeline:
    """Weight-agnostic skeleton of final single-file inference."""

    previous_model: PreviousModel
    separator: Separator
    fusion: FusionModel
    voice_experts: tuple[Expert, ...]
    music_experts: tuple[Expert, ...]
    should_separate_threshold: float = 0.5
    voice_v2_blend: float = 0.5

    def predict(self, audio: np.ndarray) -> np.ndarray:
        """Predict one file without reading any other evaluation sample."""

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform):
            raise ValueError("Expected non-empty mono audio")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio contains NaN or infinity")

        previous, frozen_scores = self.previous_model.predict_with_details(waveform)
        scores = dict(frozen_scores)

        # Separation is an observation branch. Presence values are never changed.
        should_separate = max(previous[3], previous[4]) >= self.should_separate_threshold
        if should_separate:
            vocal, music = self.separator(waveform)
        else:
            vocal = music = waveform

        for index, expert in enumerate(self.voice_experts):
            scores.update(expert.summarize(waveform, f"raw_voice_{index}"))
            scores.update(expert.summarize(vocal, f"voice_{index}"))

        for index, expert in enumerate(self.music_experts):
            scores.update(expert.summarize(waveform, f"raw_music_{index}"))
            scores.update(expert.summarize(music, f"music_{index}"))

        ads_prediction = self.fusion.predict(scores, previous)
        final = finalize_prediction(
            ads_prediction,
            baseline_v2=previous,
            voice_v2_blend=self.voice_v2_blend,
        )
        if not np.isfinite(final).all():
            raise FloatingPointError("Final prediction contains NaN or infinity")
        return final

