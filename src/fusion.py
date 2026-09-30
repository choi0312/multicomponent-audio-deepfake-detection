"""Component-aware score fusion used by the final ADS-only revision.

The module is deliberately independent of the heavyweight encoders.  It shows
the exact ideas that mattered at fusion time:

* separate voice and music evidence;
* split each score into ordinary and context-specific contributions;
* use only train-set normalization stored in the model configuration;
* preserve the two CPS outputs from the previous model exactly.

`scores` is a dictionary of per-file expert summaries. `previous` contains the
five outputs of the frozen v2 system in this order:
file_fake, voice_fake, music_fake, voice_presence, music_presence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit, logit


Array = np.ndarray


def bounded_logit(value: float) -> float:
    """Map a probability to a finite margin.

    Clipping prevents one overconfident expert from dominating a linear head on
    distributions that were not represented in training.
    """

    probability = np.clip(value, 1e-7, 1.0 - 1e-7)
    return float(np.clip(logit(probability), -12.0, 12.0))


def semantic_context(scores: Mapping[str, float]) -> tuple[float, float]:
    """Return singing and speech context ratios from frozen PANNs logits."""

    speech = float(expit(scores["pann_speech_logit"]))
    singing = float(expit(scores["pann_singing_logit"]))
    singing_ratio = singing / (singing + speech + 0.02)
    speech_ratio = speech / (speech + singing + 0.05)
    return singing_ratio, speech_ratio


def pooled(scores: Mapping[str, float], prefix: str) -> float:
    """Stable window pooling selected before the candidate sweep."""

    return 0.8 * scores[f"{prefix}_mean"] + 0.2 * scores[f"{prefix}_max"]


def component_features(
    scores: Mapping[str, float],
    previous: Sequence[float],
    task: str,
    config: Mapping[str, Any],
) -> tuple[Array, list[str]]:
    """Build ordered features for one component head.

    In `gated` mode every evidence value x becomes
    [x * (1-context), x * context]. For voice, `context` is the singing ratio;
    for music it is the speech ratio. This lets the same detector receive a
    different coefficient in ordinary speech/music and cross-component cases.
    """

    mode = config.get("mode", "pure")
    if task not in {"voice", "music"}:
        raise ValueError(f"Unknown task: {task}")
    if mode not in {"pure", "gated", "stem"}:
        raise ValueError(f"Unknown fusion mode: {mode}")

    signals: dict[str, float] = {}
    if not config.get("exclude_previous", False):
        previous_index = 1 if task == "voice" else 2
        signals["previous"] = bounded_logit(previous[previous_index])

    # `stem` is the ablation that discards raw-audio evidence. The selected
    # `gated` model keeps raw and separated observations.
    branches = [task] if mode == "stem" else ["raw", task]

    if task == "voice":
        models = ["arena", "anti_diff"]
        if config.get("sonar", False):
            models.append("sonar")
        for branch in branches:
            for model in models:
                prefix = f"{branch}_{model}"
                signals[prefix] = pooled(scores, prefix)
                if config.get("temporal", False):
                    signals[f"{prefix}_max"] = scores[f"{prefix}_max"]
    else:
        for branch in branches:
            for model in ["sonics", "adapted", "v2"]:
                prefix = f"{branch}_{model}"
                signals[prefix] = pooled(scores, prefix)
                if config.get("temporal", False):
                    signals[f"{prefix}_max"] = scores[f"{prefix}_max"]

            signals[f"{branch}_mert"] = scores[f"{branch}_mert"]

            if config.get("artifact", False):
                prefix = f"{branch}_artifact"
                signals[prefix] = bounded_logit(scores[f"{prefix}_median"])
                if config.get("temporal", False):
                    signals[f"{prefix}_max"] = bounded_logit(
                        scores[f"{prefix}_max"]
                    )

            fourier = config.get("fourier")
            if fourier in {"publisher", "both"}:
                signals[f"{branch}_fourier"] = scores[
                    f"{branch}_fourier_mean"
                ]
            if fourier in {"hull", "both"}:
                signals[f"{branch}_fourier_hull"] = scores[
                    f"{branch}_fourier_hull_mean"
                ]

    singing_ratio, speech_ratio = semantic_context(scores)
    context = singing_ratio if task == "voice" else speech_ratio
    names: list[str] = []
    values: list[float] = []

    for name, raw_value in signals.items():
        value = float(np.clip(raw_value, -12.0, 12.0))
        if mode == "gated":
            names.extend([f"{name}_ordinary", f"{name}_context"])
            values.extend([value * (1.0 - context), value * context])
        else:
            names.append(name)
            values.append(value)

    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Component evidence contains NaN or infinity")
    return result, names


def file_features(
    previous: Sequence[float], voice_fake: float, music_fake: float
) -> tuple[Array, list[str]]:
    """Construct the four inputs of the learned file-level head."""

    voice_presence, music_presence = np.clip(previous[3:5], 0.0, 1.0)
    union = 1.0 - (1.0 - voice_presence * voice_fake) * (
        1.0 - music_presence * music_fake
    )
    values = [
        bounded_logit(union),
        bounded_logit(voice_fake) * voice_presence,
        bounded_logit(music_fake) * music_presence,
        bounded_logit(previous[0]),
    ]
    return np.asarray(values), ["union", "voice", "music", "previous"]


@dataclass(frozen=True)
class LinearHead:
    """A train-normalized logistic head exported as plain numeric arrays."""

    mean: Array
    scale: Array
    coef: Array
    bias: float
    features: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LinearHead":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            coef=np.asarray(data["coef"], dtype=np.float64),
            bias=float(data["bias"]),
            features=tuple(data["features"]),
        )

    def predict(self, values: Array, temperature: float = 1.0) -> float:
        standardized = (np.asarray(values) - self.mean) / self.scale
        margin = float(standardized @ self.coef + self.bias)
        return float(expit(margin / temperature))


def finalize_prediction(
    ads_prediction: Sequence[float],
    baseline_v2: Sequence[float],
    voice_v2_blend: float = 0.5,
) -> Array:
    """Apply voice shrinkage and enforce the CPS-preservation contract.

    The final two assignments are intentional. They provide a direct, testable
    guarantee that the ADS revision cannot change the submitted CPS outputs.
    """

    if not 0.0 <= voice_v2_blend <= 1.0:
        raise ValueError("voice_v2_blend must be between 0 and 1")

    result = np.asarray(ads_prediction, dtype=np.float64).copy()
    baseline = np.asarray(baseline_v2, dtype=np.float64)
    if result.shape != (5,) or baseline.shape != (5,):
        raise ValueError("Both predictions must contain five outputs")

    result[:3] = np.clip(result[:3], 1e-9, 1.0 - 1e-9)
    result[1] = (
        (1.0 - voice_v2_blend) * result[1]
        + voice_v2_blend * baseline[1]
    )
    result[3:] = baseline[3:]  # Exact CPS pass-through.
    return result

