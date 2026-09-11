"""Freeze a uniform training subset and retain the source evaluation archive."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from noisy_regression.codec import save_codec
from noisy_regression.data import file_hash, load_pool, save_split, subset, write_json


def curate(source, output, train_count):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    print(f"Loading and verifying source pool: {source}", flush=True)
    splits, metadata = load_pool(source)
    source_count = len(splits["train"]["tokens"])
    if not isinstance(train_count, int) or not 0 < train_count < source_count:
        raise ValueError("Require integer 0 < train_count < source training count")
    print(f"Selecting {train_count:,} of {source_count:,} complete training examples", flush=True)
    indices = np.sort(np.random.default_rng().choice(source_count, size=train_count, replace=False))
    selected = subset(splits["train"], indices)
    output.mkdir(parents=True, exist_ok=False)
    indices_path = output / "train_source_indices.npy"
    np.save(indices_path, indices, allow_pickle=False)
    metadata["provenance"] = {
        "source_directory": str(source),
        "source_metadata_sha256": file_hash(source / "metadata.json"),
        "source_train_file_sha256": metadata["splits"]["train"]["file_sha256"],
        "source_train_count": source_count,
        "selection": "uniform without replacement, sorted by source index; fresh OS entropy",
        "indices_file": indices_path.name,
        "indices_file_sha256": file_hash(indices_path),
        "evaluation": "byte-for-byte copy of the source eval.npz",
    }
    metadata["config"]["train_count"] = train_count
    metadata["storage"] = "uncompressed npz"
    metadata["rng"] = "Frozen source examples; numpy.PCG64 with fresh OS entropy for subset selection"
    metadata["splits"]["train"] = save_split(output, "train", selected)
    shutil.copyfile(source / "eval.npz", output / "eval.npz")
    if file_hash(output / "eval.npz") != metadata["splits"]["eval"]["file_sha256"]:
        raise ValueError("Copied evaluation archive does not match the verified source")
    config = metadata["config"]
    save_codec(output, config["dimension"], config["observations"])
    write_json(output / "metadata.json", metadata)
    print(f"Finished curated pool: {output}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1_000_000)
    args = parser.parse_args()
    print(json.dumps(curate(args.source, args.output, args.train_count), indent=2))


if __name__ == "__main__":
    main()
