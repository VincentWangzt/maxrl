"""Explicit finite numerical vocabulary; no text tokenizer is involved."""

import json
from pathlib import Path

import numpy as np

DIGITS = 16
X, SEP, Y, EOO, QUERY, PAD, BOS, EOS = range(16, 24)
VOCAB = {
    **{f"{i:X}": i for i in range(16)},
    "[X]": X,
    "[SEP]": SEP,
    "[Y]": Y,
    "[EOO]": EOO,
    "[QUERY]": QUERY,
    "[PAD]": PAD,
    "[BOS]": BOS,
    "[EOS]": EOS,
}
RANGE_MIN, RANGE_MAX = -3.0, 3.0
DELTA = (RANGE_MAX - RANGE_MIN) / 255
CENTERS = RANGE_MIN + np.arange(256, dtype=np.float64) * DELTA
MIDPOINTS = RANGE_MIN + (np.arange(255, dtype=np.float64) + 0.5) * DELTA
DIMENSION = 2
OBSERVATIONS = 64
SCALAR_TOKENS = 2
INPUT_TOKENS = SCALAR_TOKENS * DIMENSION
OBSERVATION_TOKENS = 1 + SCALAR_TOKENS + 1 + SCALAR_TOKENS + 1 + SCALAR_TOKENS + 1
QUERY_OFFSET = 1 + OBSERVATIONS * OBSERVATION_TOKENS
CONTEXT_SLICE = slice(1, QUERY_OFFSET)
PROMPT_LENGTH = QUERY_OFFSET + 1 + 1 + SCALAR_TOKENS + 1 + SCALAR_TOKENS + 1
SEQUENCE_LENGTH = PROMPT_LENGTH + SCALAR_TOKENS + 1


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
    observations[:, :, 1:3] = encode(context_x[:, :, 0])
    observations[:, :, 3] = SEP
    observations[:, :, 4:6] = encode(context_x[:, :, 1])
    observations[:, :, 6] = Y
    observations[:, :, 7:9] = encode(context_y)
    observations[:, :, 9] = EOO
    sequence[:, QUERY_OFFSET] = QUERY
    sequence[:, QUERY_OFFSET + 1] = X
    sequence[:, QUERY_OFFSET + 2 : QUERY_OFFSET + 4] = encode(query_x[:, 0])
    sequence[:, QUERY_OFFSET + 4] = SEP
    sequence[:, QUERY_OFFSET + 5 : QUERY_OFFSET + 7] = encode(query_x[:, 1])
    sequence[:, PROMPT_LENGTH - 1] = Y
    sequence[:, PROMPT_LENGTH : PROMPT_LENGTH + SCALAR_TOKENS] = encode(query_y)
    sequence[:, -1] = EOS
    return sequence


def codec_config():
    return {
        "vocab": VOCAB,
        "range": [RANGE_MIN, RANGE_MAX],
        "levels": 256,
        "delta": DELTA,
        "midpoint_ties": "larger_index",
        "nonfinite": "reject",
        "scalar_tokens": SCALAR_TOKENS,
        "eos_token_id": EOS,
        "dimension": DIMENSION,
        "observations": OBSERVATIONS,
        "prompt_length": PROMPT_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
    }


def save_codec(directory):
    path = Path(directory) / "codec.json"
    path.write_text(json.dumps(codec_config(), indent=2) + "\n")
