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
RANGE_MIN, RANGE_MAX = -3.0, 3.0
DELTA = (RANGE_MAX - RANGE_MIN) / 255
CENTERS = RANGE_MIN + np.arange(256, dtype=np.float64) * DELTA
MIDPOINTS = RANGE_MIN + (np.arange(255, dtype=np.float64) + 0.5) * DELTA
DIMENSION = 2
OBSERVATIONS = 64
INPUT_TOKENS = 2 * DIMENSION
OBSERVATION_TOKENS = INPUT_TOKENS + 4
QUERY_OFFSET = 1 + OBSERVATIONS * OBSERVATION_TOKENS
CONTEXT_SLICE = slice(1, QUERY_OFFSET)
PROMPT_LENGTH = QUERY_OFFSET + INPUT_TOKENS + 2
SEQUENCE_LENGTH = PROMPT_LENGTH + 2


def quantize(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Numerical codec rejects nonfinite inputs")
    # Equivalent to clipped floor((z-RANGE_MIN)/Delta+0.5). Searching the actual bin
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


def build_sequences(context_x, context_y, query_x, query_y, capacity=1024):
    count = len(context_x)
    if (
        context_x.shape != (count, OBSERVATIONS, DIMENSION)
        or context_y.shape != (count, OBSERVATIONS)
        or query_x.shape != (count, DIMENSION)
        or query_y.shape != (count,)
    ):
        raise ValueError(
            f"Expected (B,{OBSERVATIONS},{DIMENSION}), (B,{OBSERVATIONS}), (B,{DIMENSION}), (B,) continuous arrays"
        )
    if SEQUENCE_LENGTH > capacity:
        raise ValueError(f"Sequence length {SEQUENCE_LENGTH} exceeds capacity {capacity}; truncation is forbidden")
    sequence = np.empty((count, SEQUENCE_LENGTH), dtype=np.uint8)
    sequence[:, 0] = BOS
    observations = sequence[:, CONTEXT_SLICE].reshape(count, OBSERVATIONS, OBSERVATION_TOKENS)
    observations[:, :, 0] = X
    observations[:, :, 1 : 1 + INPUT_TOKENS] = encode(context_x).reshape(count, OBSERVATIONS, INPUT_TOKENS)
    observations[:, :, 1 + INPUT_TOKENS] = Y
    observations[:, :, -2:] = encode(context_y)
    sequence[:, QUERY_OFFSET] = X
    sequence[:, QUERY_OFFSET + 1 : PROMPT_LENGTH - 1] = encode(query_x).reshape(count, INPUT_TOKENS)
    sequence[:, PROMPT_LENGTH - 1] = Y
    sequence[:, PROMPT_LENGTH:] = encode(query_y)
    return sequence


def codec_config():
    return {
        "vocab": VOCAB,
        "range": [RANGE_MIN, RANGE_MAX],
        "levels": 256,
        "delta": DELTA,
        "midpoint_ties": "larger_index",
        "nonfinite": "reject",
        "scalar_tokens": 2,
        "eos_token_id": None,
        "dimension": DIMENSION,
        "observations": OBSERVATIONS,
        "prompt_length": PROMPT_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
    }


def save_codec(directory):
    path = Path(directory) / "codec.json"
    path.write_text(json.dumps(codec_config(), indent=2) + "\n")
