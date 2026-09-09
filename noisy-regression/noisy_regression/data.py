"""Generate once, hash, and load immutable complete regression problems."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from noisy_regression.codec import (
    DIMENSION,
    OBSERVATIONS,
    PROMPT_LENGTH,
    RANGE_MAX,
    RANGE_MIN,
    build_sequences,
    codec_config,
    save_codec,
)


@dataclass(frozen=True)
class DatasetConfig:
    train_count: int = 10_000_000
    eval_count: int = 1_024
    dimension: int = DIMENSION
    observations: int = OBSERVATIONS
    sigma: float = 0.001  # Shared standard deviation; context/query draws are independent.
    capacity: int = 1024

    def validate(self):
        if (self.dimension, self.observations, self.capacity) != (DIMENSION, OBSERVATIONS, 1024):
            raise ValueError(f"This experiment requires d={DIMENSION}, n={OBSERVATIONS}, capacity=1024")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Require finite sigma > 0 for both context and query noise")
        if min(self.train_count, self.eval_count) < 1:
            raise ValueError("Require positive pool sizes")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        for start in range(0, len(value), 65_536):
            digest.update(np.ascontiguousarray(value[start : start + 65_536]).tobytes())
    return digest.hexdigest()


def generate_split(config, split):
    config.validate()
    if split not in ("train", "eval"):
        raise ValueError("Only train and held-out eval splits exist")
    count = config.train_count if split == "train" else config.eval_count
    rng = np.random.default_rng()
    w = rng.normal(size=(count, config.dimension)) / np.sqrt(config.dimension)
    context_x = rng.normal(size=(count, config.observations, config.dimension))
    context_noise = rng.normal(scale=config.sigma, size=(count, config.observations))
    context_y = np.einsum("bnd,bd->bn", context_x, w) + context_noise
    query_x = rng.normal(size=(count, config.dimension))
    query_noise = rng.normal(scale=config.sigma, size=count)
    query_signal = np.einsum("bd,bd->b", query_x, w)
    query_y = query_signal + query_noise
    arrays = dict(
        w=w,
        context_x=context_x,
        context_y=context_y,
        context_noise=context_noise,
        query_x=query_x,
        query_y=query_y,
        query_noise=query_noise,
        query_signal=query_signal,
    )
    arrays["tokens"] = build_sequences(context_x, context_y, query_x, query_y, config.capacity)
    arrays["ids"] = np.array([f"{split}-{i:08d}" for i in range(count)])
    arrays["prompt_hashes"] = np.array(
        [hashlib.sha256(row[:PROMPT_LENGTH].tobytes()).hexdigest() for row in arrays["tokens"]], dtype="S64"
    )
    return arrays


def clipping_summary(arrays):
    result = {}
    for name in ("context_x", "context_y", "query_x", "query_y"):
        values = arrays[name]
        low, high = int((values < RANGE_MIN).sum()), int((values > RANGE_MAX).sum())
        result[name] = {
            "below": low,
            "above": high,
            "total_scalars": values.size,
            "fraction": (low + high) / values.size,
        }
    return result


def prepare(directory, config):
    config.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split in ("train", "eval"):
        print(f"Generating {split} pool with fresh random draws", flush=True)
        splits[split] = generate_split(config, split)
        print(f"Generated {len(splits[split]['tokens']):,} {split} examples", flush=True)
    overlap = set(splits["train"]["prompt_hashes"]) & set(splits["eval"]["prompt_hashes"])
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping tokenized prompts across splits")
    metadata = {
        "schema_version": 4,
        "config": asdict(config),
        "codec": codec_config(),
        "rng": "numpy.PCG64; independent OS entropy for each split; no fixed seeds",
        "storage": "uncompressed npz; avoid compression overhead for the 10M pool",
        "prompt_format": f"[BOS] ([X] x [Y] y) * {config.observations} [X] query [Y] answer",
        "numpy_version": np.__version__,
        "split_prompt_overlap": 0,
        "split_role": "held-out evaluation reused for checkpoint selection; no test split",
        "splits": {},
    }
    for split, arrays in splits.items():
        path = directory / f"{split}.npz"
        print(f"Writing and fingerprinting {split} archive", flush=True)
        np.savez(path, **arrays)
        metadata["splits"][split] = {
            "count": len(arrays["tokens"]),
            "content_sha256": array_hash(arrays),
            "file_sha256": file_hash(path),
            "unique_prompts": len(set(arrays["prompt_hashes"])),
            "clipping": clipping_summary(arrays),
        }
        print(f"Finished {split}: {path.stat().st_size:,} bytes", flush=True)
    save_codec(directory)
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_pool(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["schema_version"] != 4:
        raise ValueError("Dataset schema mismatch: prepare a new d=2, n=64 pool with the [-3,3] codec")
    if metadata["codec"] != codec_config() or json.loads((directory / "codec.json").read_text()) != codec_config():
        raise ValueError("Dataset codec mismatch: scalar range and prompt layout must match the running code")
    DatasetConfig(**metadata["config"]).validate()
    splits = {}
    for split in ("train", "eval"):
        path = directory / f"{split}.npz"
        if file_hash(path) != metadata["splits"][split]["file_sha256"]:
            raise ValueError(f"Corrupt or modified {split} dataset")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if array_hash(arrays) != metadata["splits"][split]["content_sha256"]:
            raise ValueError(f"{split} content fingerprint mismatch")
        for value in arrays.values():
            value.flags.writeable = False
        splits[split] = arrays
    if set(splits["train"]["prompt_hashes"]) & set(splits["eval"]["prompt_hashes"]):
        raise ValueError("Train/eval prompt overlap")
    return splits, metadata


def subset(arrays, indices):
    return {name: value[indices] for name, value in arrays.items()}


class FrozenOrder:
    """A resumable shuffled index stream; never mutates or regenerates examples."""

    def __init__(self, count):
        if count < 1:
            raise ValueError("Empty training pool")
        self.count = count
        self.rng = np.random.default_rng()
        self.order = self.rng.permutation(count)
        self.cursor = 0
        self.epochs = 0
        self.presentations = 0

    def take(self, count):
        if count < 1:
            raise ValueError("Batch size must be positive")
        parts = []
        remaining = count
        while remaining:
            if self.cursor == self.count:
                self.epochs += 1
                self.order = self.rng.permutation(self.count)
                self.cursor = 0
            size = min(remaining, self.count - self.cursor)
            parts.append(self.order[self.cursor : self.cursor + size])
            self.cursor += size
            remaining -= size
        self.presentations += count
        return np.concatenate(parts)

    def state_dict(self):
        return {
            "count": self.count,
            "order": self.order.copy(),
            "cursor": self.cursor,
            "epochs": self.epochs,
            "presentations": self.presentations,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state["count"] != self.count:
            raise ValueError("Cannot resume a different training pool")
        self.order = state["order"].copy()
        self.cursor, self.epochs, self.presentations = state["cursor"], state["epochs"], state["presentations"]
        self.rng.bit_generator.state = state["rng"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name, field in DatasetConfig.__dataclass_fields__.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(field.default), default=field.default)
    args = vars(parser.parse_args())
    output = args.pop("output")
    print(json.dumps(prepare(output, DatasetConfig(**args)), indent=2))


if __name__ == "__main__":
    main()
