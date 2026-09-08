"""Explicit finite numerical vocabulary; no text tokenizer is involved."""

import json
from pathlib import Path

import numpy as np

DIGITS = 16
X, Y, PAD, BOS = range(16, 20)
VOCAB = {
    **{f"{i:X}": i for i in range(16)},
    "[X]": X,
    "[Y]": Y,
    "[PAD]": PAD,
    "[BOS]": BOS,
}
DELTA = 10.0 / 255
CENTERS = -5.0 + np.arange(256, dtype=np.float64) * DELTA
MIDPOINTS = -5.0 + (np.arange(255, dtype=np.float64) + 0.5) * DELTA
PROMPT_LENGTH = 203
SEQUENCE_LENGTH = 205


def quantize(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Numerical codec rejects nonfinite inputs")
    # Equivalent to clipped floor((z+5)/Delta+0.5). Searching the actual bin
    # boundaries avoids arithmetic cancellation at midpoint ties. side=right
    # puts a represented midpoint in the larger bin, including zero -> 128.
    return np.searchsorted(MIDPOINTS, values, side="right").astype(np.int64)


def encode(values):
    indices = quantize(values)
    return np.stack((indices // DIGITS, indices % DIGITS), axis=-1)


def decode(tokens):
    tokens = np.asarray(tokens)
    if (
        tokens.ndim == 0
        or tokens.shape[-1] != 2
        or tokens.dtype.kind not in "iu"
        or np.any((tokens < 0) | (tokens >= DIGITS))
    ):
        raise ValueError("Expected pairs of integer digit IDs in [0, 15]")
    return CENTERS[tokens[..., 0] * DIGITS + tokens[..., 1]]


def build_sequences(context_x, context_y, query_x, query_y, capacity=512):
    count = len(context_x)
    if (
        context_x.shape != (count, 16, 4)
        or context_y.shape != (count, 16)
        or query_x.shape != (count, 4)
        or query_y.shape != (count,)
    ):
        raise ValueError("Expected (B,16,4), (B,16), (B,4), (B,) continuous arrays")
    if SEQUENCE_LENGTH > capacity:
        raise ValueError(f"Sequence length {SEQUENCE_LENGTH} exceeds capacity {capacity}; truncation is forbidden")
    sequence = np.empty((count, SEQUENCE_LENGTH), dtype=np.uint8)
    sequence[:, 0] = BOS
    observations = sequence[:, 1:193].reshape(count, 16, 12)
    observations[:, :, 0] = X
    observations[:, :, 1:9] = encode(context_x).reshape(count, 16, 8)
    observations[:, :, 9] = Y
    observations[:, :, 10:12] = encode(context_y)
    sequence[:, 193] = X
    sequence[:, 194:202] = encode(query_x).reshape(count, 8)
    sequence[:, 202] = Y
    sequence[:, 203:205] = encode(query_y)
    return sequence


def save_codec(directory):
    path = Path(directory) / "codec.json"
    path.write_text(
        json.dumps(
            {
                "vocab": VOCAB,
                "range": [-5, 5],
                "levels": 256,
                "delta": DELTA,
                "midpoint_ties": "larger_index",
                "nonfinite": "reject",
                "scalar_tokens": 2,
                "eos_token_id": None,
                "prompt_length": PROMPT_LENGTH,
                "sequence_length": SEQUENCE_LENGTH,
            },
            indent=2,
        )
        + "\n"
    )
