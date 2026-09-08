"""Generate once, hash, and load immutable complete regression problems."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from noisy_regression.codec import PROMPT_LENGTH, build_sequences, save_codec


@dataclass(frozen=True)
class DatasetConfig:
    train_count: int = 100_000
    eval_count: int = 1_024
    train_seed: int = 1729
    eval_seed: int = 2718
    dimension: int = 4
    observations: int = 16
    sigma: float = 0.5  # Shared standard deviation; context/query draws are independent.
    capacity: int = 512

    def validate(self):
        if (self.dimension, self.observations, self.capacity) != (4, 16, 512):
            raise ValueError("This experiment requires d=4, n=16, capacity=512")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Require finite sigma > 0 for both context and query noise")
        if (
            min(self.train_count, self.eval_count) < 1
            or self.train_seed == self.eval_seed
            or min(self.train_seed, self.eval_seed) < 0
        ):
            raise ValueError("Require positive pool sizes and distinct nonnegative split seeds")


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
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def generate_split(config, split):
    config.validate()
    if split not in ("train", "eval"):
        raise ValueError("Only train and held-out eval splits exist")
    count, seed = (config.train_count, config.train_seed) if split == "train" else (config.eval_count, config.eval_seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    w = rng.normal(size=(count, 4)) / np.sqrt(4)
    context_x = rng.normal(size=(count, 16, 4))
    context_noise = rng.normal(scale=config.sigma, size=(count, 16))
    context_y = np.einsum("bnd,bd->bn", context_x, w) + context_noise
    query_x = rng.normal(size=(count, 4))
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
    arrays["ids"] = np.array([f"{split}-{seed}-{i:08d}" for i in range(count)])
    arrays["prompt_hashes"] = np.array(
        [hashlib.sha256(row[:PROMPT_LENGTH].tobytes()).hexdigest() for row in arrays["tokens"]]
    )
    return arrays


def clipping_summary(arrays):
    result = {}
    for name in ("context_x", "context_y", "query_x", "query_y"):
        values = arrays[name]
        low, high = int((values < -5).sum()), int((values > 5).sum())
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
    splits = {split: generate_split(config, split) for split in ("train", "eval")}
    overlap = set(splits["train"]["prompt_hashes"]) & set(splits["eval"]["prompt_hashes"])
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping tokenized prompts across splits")
    metadata = {
        "schema_version": 1,
        "config": asdict(config),
        "rng": "numpy.PCG64; independent train/eval seeds",
        "numpy_version": np.__version__,
        "split_prompt_overlap": 0,
        "split_role": "held-out evaluation reused for checkpoint selection; no test split",
        "splits": {},
    }
    for split, arrays in splits.items():
        path = directory / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        metadata["splits"][split] = {
            "count": len(arrays["tokens"]),
            "content_sha256": array_hash(arrays),
            "file_sha256": file_hash(path),
            "unique_prompts": len(set(arrays["prompt_hashes"])),
            "clipping": clipping_summary(arrays),
        }
    save_codec(directory)
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_pool(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
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


def fixed_subset_indices(train_count, eval_count, train_eval_size, generation_size, seed):
    rng = np.random.default_rng(seed)
    train_indices = rng.permutation(train_count)[:train_eval_size]
    eval_indices = rng.permutation(eval_count)[:generation_size]
    return train_indices, eval_indices


class FrozenOrder:
    """A resumable shuffled index stream; never mutates or regenerates examples."""

    def __init__(self, count, seed):
        if count < 1:
            raise ValueError("Empty training pool")
        self.count = count
        self.rng = np.random.Generator(np.random.PCG64(seed))
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
