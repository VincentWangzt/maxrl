"""Select reproducible 2,048/128 subsets of the decontaminated 17x17 maze splits.

Run on the server from the repository root with
``.venv/bin/python -m maze.curate_maze_2048``. Original row indices and all
columns are preserved so the subsets can be compared with the larger runs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from maze.curate_maze_16384 import file_sha256, load_question_fingerprints

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "maze/data/maze_17_16384"
OUTPUT_DIR = REPO_ROOT / "maze/data/maze_17_2048"
TRAIN_COUNT = 2_048
EVAL_COUNT = 128
TRAIN_SEED = 2_048
EVAL_SEED = 128


def select_subset(source: pa.Table, count: int, seed: int) -> tuple[pa.Table, list[int]]:
    if count <= 0 or count > source.num_rows:
        raise ValueError(f"Cannot select {count} rows from a {source.num_rows}-row split")
    indices = random.Random(seed).sample(range(source.num_rows), count)
    return source.take(indices), indices


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT_DIR}")

    source_metadata_path = SOURCE_DIR / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    subsets: dict[str, pa.Table] = {}
    source_fingerprints: dict[str, set[bytes]] = {}
    source_files: dict[str, dict] = {}
    split_specs = (
        ("train.parquet", "train_rows", TRAIN_COUNT, TRAIN_SEED),
        ("test.parquet", "eval_rows", EVAL_COUNT, EVAL_SEED),
    )

    for filename, row_count_key, count, seed in split_specs:
        source_path = SOURCE_DIR / filename
        source_sha256 = file_sha256(source_path)
        if source_sha256 != source_metadata["files"][filename]:
            raise ValueError(f"Source checksum does not match its metadata: {source_path}")
        fingerprints, source_rows = load_question_fingerprints(source_path)
        if source_rows != source_metadata[row_count_key] or len(fingerprints) != source_rows:
            raise ValueError(f"Source row count or question uniqueness validation failed: {source_path}")
        source_fingerprints[filename] = fingerprints
        source_table = pq.read_table(source_path)
        subsets[filename], indices = select_subset(source_table, count, seed)
        source_files[filename] = {
            "path": str(source_path.relative_to(REPO_ROOT)),
            "sha256": source_sha256,
            "rows": source_rows,
            "selection_seed": seed,
            "selected_row_indices": indices,
        }

    if source_fingerprints["train.parquet"] & source_fingerprints["test.parquet"]:
        raise ValueError("Source training and evaluation splits overlap")

    OUTPUT_DIR.mkdir(parents=True)
    written_fingerprints: dict[str, set[bytes]] = {}
    output_files: dict[str, str] = {}
    for filename, _, count, _ in split_specs:
        output_path = OUTPUT_DIR / filename
        pq.write_table(subsets[filename], output_path, compression="snappy")
        if not pq.read_table(output_path).equals(subsets[filename]):
            raise RuntimeError(f"Written subset does not match selected source rows: {output_path}")
        fingerprints, rows = load_question_fingerprints(output_path)
        if rows != count or len(fingerprints) != count:
            raise RuntimeError(f"Written subset row count or uniqueness validation failed: {output_path}")
        if not fingerprints <= source_fingerprints[filename]:
            raise RuntimeError(f"Written subset contains questions outside its source split: {output_path}")
        written_fingerprints[filename] = fingerprints
        output_files[filename] = file_sha256(output_path)

    if written_fingerprints["train.parquet"] & written_fingerprints["test.parquet"]:
        raise RuntimeError("Written training and evaluation splits overlap")

    metadata = {
        "maze_size": source_metadata["maze_size"],
        "algorithm": source_metadata["algorithm"],
        "selection_method": "Seeded sampling without replacement within each original split",
        "train_rows": TRAIN_COUNT,
        "eval_rows": EVAL_COUNT,
        "source_metadata": {
            "path": str(source_metadata_path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(source_metadata_path),
        },
        "source_files": source_files,
        "decontamination": {
            "method": "Inherited from checksum-verified parent splits; original train/test membership is preserved",
            "parent": source_metadata["decontamination"],
        },
        "files": output_files,
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "train_rows": TRAIN_COUNT,
                "eval_rows": EVAL_COUNT,
                "train_eval_overlap": 0,
                "files": output_files,
                "metadata": str(metadata_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
