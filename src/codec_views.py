"""Deterministic codec views used for robustness training.

The functions create a transformed waveform in memory. They do not write audio
files, which keeps the external data budget and provenance easier to audit.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np


CODEC_OUTPUT_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3"],
    "aac": ["-c:a", "aac", "-b:a", "32k", "-f", "adts"],
    "telephone": ["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"],
}


def codec_view(audio: np.ndarray, codec: str, ffmpeg: str | None = None) -> np.ndarray:
    """Encode and decode one 16 kHz mono float waveform with a fixed codec."""

    if codec not in CODEC_OUTPUT_ARGS:
        raise ValueError(f"Unknown codec: {codec}")
    executable = ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise FileNotFoundError("ffmpeg is required for codec augmentation")

    waveform = np.asarray(audio, dtype="<f4")
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Expected a finite, non-empty mono waveform")

    encode = [
        executable,
        "-v", "error",
        "-f", "f32le",
        "-ar", "16000",
        "-ac", "1",
        "-i", "pipe:0",
        *CODEC_OUTPUT_ARGS[codec],
        "pipe:1",
    ]
    payload = subprocess.run(
        encode,
        input=waveform.tobytes(),
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout

    decode = [
        executable,
        "-v", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-ar", "16000",
        "-ac", "1",
        "pipe:1",
    ]
    decoded = subprocess.run(
        decode,
        input=payload,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    result = np.frombuffer(decoded, dtype="<f4").copy()
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"Codec transformation failed: {codec}")
    return result


def parent_shared_weights(parent_ids: list[str]) -> np.ndarray:
    """Give all views of one parent a total weight of one.

    For a clean sample plus three codec views this returns 0.25 for every view,
    preventing that selected parent from counting four times.
    """

    counts = {parent: parent_ids.count(parent) for parent in set(parent_ids)}
    return np.asarray([1.0 / counts[parent] for parent in parent_ids])

