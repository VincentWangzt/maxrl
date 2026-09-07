"""Select a reproducible noisy-maze RL subset without changing observations or evaluation."""

import argparse
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from noisy_maze.prepare import file_sha256, question_fingerprint, write_json


def curate_subset(source_dir: Path, train_count: int = 1024, seed: int = 1024) -> dict:
    source_metadata_path = source_dir / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    title = source_metadata["title"]
    output_title = f"{title}_rl_{train_count}"
    output_dir = source_dir.parent / output_title
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")

    tables, fingerprints, source_files = {}, {}, {}
    for split in ("train", "test"):
        path = source_dir / f"{split}.parquet"
        checksum = file_sha256(path)
        if checksum != source_metadata["files"][f"{source_dir.name}/{path.name}"]:
            raise ValueError(f"Source checksum does not match metadata: {path}")
        table = pq.read_table(path)
        metadata_key = "rl_train" if split == "train" else "rl_eval"
        if table.num_rows != source_metadata["splits"][metadata_key]["rows"]:
            raise ValueError(f"Source row count does not match metadata: {path}")
        keys = {question_fingerprint(row["ground_truth"]) for row in table.column("reward_model").to_pylist()}
        if len(keys) != table.num_rows:
            raise ValueError(f"Duplicate underlying mazes in {path}")
        tables[split], fingerprints[split] = table, keys
        source_files[path.name] = {"sha256": checksum, "rows": table.num_rows}
    if fingerprints["train"] & fingerprints["test"]:
        raise ValueError("Source training and evaluation mazes overlap")
    if not 0 < train_count < tables["train"].num_rows:
        raise ValueError("train_count must be positive and smaller than the source training split")

    indices = random.Random(seed).sample(range(tables["train"].num_rows), train_count)
    subset = tables["train"].take(indices)
    output_dir.mkdir(parents=True)
    pq.write_table(subset, output_dir / "train.parquet", compression="snappy")
    shutil.copyfile(source_dir / "test.parquet", output_dir / "test.parquet")
    if not pq.read_table(output_dir / "train.parquet").equals(subset):
        raise RuntimeError("Written subset differs from selected source rows")
    if file_sha256(output_dir / "test.parquet") != source_files["test.parquet"]["sha256"]:
        raise RuntimeError("Evaluation file changed while copying")

    train_rows = subset.to_pylist()
    metadata = {
        **source_metadata,
        "rl_title": output_title,
        "selection_method": "Seeded sampling without replacement; preserve full rows and evaluation file exactly",
        "selection_seed": seed,
        "selected_source_row_indices": indices,
        "source_metadata": {"path": str(source_metadata_path), "sha256": file_sha256(source_metadata_path)},
        "source_files": source_files,
        "splits": {
            **source_metadata["splits"],
            "rl_train": {
                "rows": train_count,
                "max_sequence_tokens": max(len(row["extra_info"]["answer"].split()) for row in train_rows),
                "max_optimal_path_length": max(row["extra_info"]["optimal_path_length"] for row in train_rows),
            },
        },
        "files": {f"{output_title}/{name}.parquet": file_sha256(output_dir / f"{name}.parquet") for name in ("train", "test")},
    }
    write_json(output_dir / "metadata.json", metadata)
    return {
        "output_dir": str(output_dir),
        "train_rows": train_count,
        "eval_rows": tables["test"].num_rows,
        "noise_fraction": metadata["noise_fraction"],
        "selection_seed": seed,
        "train_eval_overlap": 0,
        "evaluation_preserved_exactly": True,
        "files": metadata["files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(curate_subset(args.source_dir, args.train_count, args.seed), indent=2), flush=True)


if __name__ == "__main__":
    main()
