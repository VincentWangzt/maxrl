"""Explicit finite numerical vocabulary; no text tokenizer is involved."""

import json
from dataclasses import dataclass
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
RANGE_MIN, RANGE_MAX = -4.0, 4.0
DELTA = (RANGE_MAX - RANGE_MIN) / 255
CENTERS = RANGE_MIN + np.arange(256, dtype=np.float64) * DELTA
MIDPOINTS = RANGE_MIN + (np.arange(255, dtype=np.float64) + 0.5) * DELTA
DIMENSION = 2
OBSERVATIONS = 64
SCALAR_TOKENS = 2


@dataclass(frozen=True)
class SequenceLayout:
    dimension: int
    observations: int

    def __post_init__(self):
        if not isinstance(self.dimension, int) or not isinstance(self.observations, int):
            raise ValueError("Dimension and observation count must be integers")
        if self.dimension < 1 or self.observations < 1:
            raise ValueError("Dimension and observation count must be positive")

    @property
    def input_tokens(self):
        return SCALAR_TOKENS * self.dimension

    @property
    def observation_tokens(self):
        # [X], d digit pairs separated by [SEP], [Y], one digit pair, [EOO].
        return 1 + self.input_tokens + (self.dimension - 1) + 1 + SCALAR_TOKENS + 1

    @property
    def query_offset(self):
        return 1 + self.observations * self.observation_tokens

    @property
    def context_slice(self):
        return slice(1, self.query_offset)

    @property
    def prompt_length(self):
        # [QUERY], [X], d digit pairs and separators, then [Y].
        return self.query_offset + 1 + 1 + self.input_tokens + (self.dimension - 1) + 1

    @property
    def sequence_length(self):
        return self.prompt_length + SCALAR_TOKENS + 1


def sequence_layout(dimension=DIMENSION, observations=OBSERVATIONS):
    return SequenceLayout(dimension, observations)


DEFAULT_LAYOUT = sequence_layout()
INPUT_TOKENS = DEFAULT_LAYOUT.input_tokens
OBSERVATION_TOKENS = DEFAULT_LAYOUT.observation_tokens
QUERY_OFFSET = DEFAULT_LAYOUT.query_offset
CONTEXT_SLICE = DEFAULT_LAYOUT.context_slice
PROMPT_LENGTH = DEFAULT_LAYOUT.prompt_length
SEQUENCE_LENGTH = DEFAULT_LAYOUT.sequence_length


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
    if context_x.ndim != 3:
        raise ValueError("Expected context_x with shape (batch, observations, dimension)")
    layout = sequence_layout(context_x.shape[2], context_x.shape[1])
    if (
        context_y.shape != (count, layout.observations)
        or query_x.shape != (count, layout.dimension)
        or query_y.shape != (count,)
    ):
        raise ValueError(
            f"Expected (B,{layout.observations},{layout.dimension}), (B,{layout.observations}), "
            f"(B,{layout.dimension}), (B,) continuous arrays"
        )
    if layout.sequence_length > capacity:
        raise ValueError(
            f"Sequence length {layout.sequence_length} exceeds capacity {capacity}; truncation is forbidden"
        )
    sequence = np.empty((count, layout.sequence_length), dtype=np.uint8)
    sequence[:, 0] = BOS
    observations = sequence[:, layout.context_slice].reshape(
        count, layout.observations, layout.observation_tokens
    )
    observations[:, :, 0] = X
    position = 1
    for coordinate in range(layout.dimension):
        observations[:, :, position : position + SCALAR_TOKENS] = encode(context_x[:, :, coordinate])
        position += SCALAR_TOKENS
        if coordinate + 1 < layout.dimension:
            observations[:, :, position] = SEP
            position += 1
    observations[:, :, position] = Y
    position += 1
    observations[:, :, position : position + SCALAR_TOKENS] = encode(context_y)
    observations[:, :, -1] = EOO

    sequence[:, layout.query_offset] = QUERY
    sequence[:, layout.query_offset + 1] = X
    position = layout.query_offset + 2
    for coordinate in range(layout.dimension):
        sequence[:, position : position + SCALAR_TOKENS] = encode(query_x[:, coordinate])
        position += SCALAR_TOKENS
        if coordinate + 1 < layout.dimension:
            sequence[:, position] = SEP
            position += 1
    sequence[:, position] = Y
    sequence[:, layout.prompt_length : layout.prompt_length + SCALAR_TOKENS] = encode(query_y)
    sequence[:, -1] = EOS
    return sequence


def codec_config(dimension=DIMENSION, observations=OBSERVATIONS):
    layout = sequence_layout(dimension, observations)
    return {
        "vocab": VOCAB,
        "range": [RANGE_MIN, RANGE_MAX],
        "levels": 256,
        "delta": DELTA,
        "midpoint_ties": "larger_index",
        "nonfinite": "reject",
        "scalar_tokens": SCALAR_TOKENS,
        "eos_token_id": EOS,
        "dimension": layout.dimension,
        "observations": layout.observations,
        "prompt_length": layout.prompt_length,
        "sequence_length": layout.sequence_length,
    }


def save_codec(directory, dimension=DIMENSION, observations=OBSERVATIONS):
    path = Path(directory) / "codec.json"
    path.write_text(json.dumps(codec_config(dimension, observations), indent=2) + "\n")
