"""Generate disjoint 23x23 SFT and RL maze datasets.

Run this module on the training server from the repository root:

``python -m maze.prepare_maze_23``

The SFT split contains 100,000 training mazes and 128 evaluation mazes. The
separate 1,024/128 RL split mirrors the current 17x17 comparison without
reusing any maze seen during SFT.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from maze.generate_maze import MazeGenerator

REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_OUTPUT_DIR = REPO_ROOT / "maze/data/maze_23_sft_100k"
RL_OUTPUT_DIR = REPO_ROOT / "maze/data/maze_23_1024"

MAZE_SIZE = 23
GENERATOR_SEED = 23_202_609
SFT_TRAIN_COUNT = 100_000
SFT_EVAL_COUNT = 128
RL_TRAIN_COUNT = 1_024
RL_EVAL_COUNT = 128


def question_from_sequence(sequence: str) -> str:
    prefix, separator, _ = sequence.partition("PATH_START")
    if not separator:
        raise ValueError("Generated maze sequence does not contain PATH_START")
    return f"{prefix.strip()} PATH_START"


def question_fingerprint(sequence: str) -> bytes:
    question = question_from_sequence(sequence)
    return hashlib.sha256(question.encode("utf-8")).digest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, records: list[dict]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(records, file_handle, ensure_ascii=False)
            file_handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def make_rl_record(item: dict, split: str, index: int) -> dict:
    answer = item["sequence"]
    question = question_from_sequence(answer)
    return {
        "data_source": "maze_23",
        "prompt": [{"content": question, "role": "user"}],
        "ability": "maze",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "answer": answer,
            "index": index,
            "optimal_path_length": item["optimal_path_length"],
            "question": question,
            "split": split,
        },
    }


def write_parquet(path: Path, items: list[dict], split: str) -> None:
    records = [make_rl_record(item, split, index) for index, item in enumerate(items)]
    table = pa.Table.from_pylist(records)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pq.write_table(table, temporary_path, compression="snappy")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def path_length_summary(items: list[dict]) -> dict:
    lengths = [item["optimal_path_length"] for item in items]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def main() -> None:
    if MAZE_SIZE % 2 != 1:
        raise ValueError("The current maze generator requires an odd maze size")

    output_paths = {
        "sft_train": SFT_OUTPUT_DIR / "train.json",
        "sft_eval": SFT_OUTPUT_DIR / "test.json",
        "rl_train": RL_OUTPUT_DIR / "train.parquet",
        "rl_eval": RL_OUTPUT_DIR / "test.parquet",
        "metadata": SFT_OUTPUT_DIR / "metadata.json",
    }
    existing_paths = [path for path in output_paths.values() if path.exists()]
    if existing_paths:
        joined_paths = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{joined_paths}")

    split_counts = (
        ("sft_train", SFT_TRAIN_COUNT),
        ("sft_eval", SFT_EVAL_COUNT),
        ("rl_train", RL_TRAIN_COUNT),
        ("rl_eval", RL_EVAL_COUNT),
    )
    generator = MazeGenerator(size=MAZE_SIZE, seed=GENERATOR_SEED, algorithm="prim")
    seen_questions: set[bytes] = set()
    splits: dict[str, list[dict]] = {name: [] for name, _ in split_counts}
    candidates_examined = 0
    duplicate_candidates = 0

    for split_name, target_count in split_counts:
        split_items = splits[split_name]
        while len(split_items) < target_count:
            generator.generate()
            item = generator.to_text_sequence()
            if item is None:
                continue

            candidates_examined += 1
            fingerprint = question_fingerprint(item["sequence"])
            if fingerprint in seen_questions:
                duplicate_candidates += 1
                continue

            seen_questions.add(fingerprint)
            split_items.append(item)
            if split_name == "sft_train" and len(split_items) % 10_000 == 0:
                print(f"Generated {len(split_items):,}/{target_count:,} SFT training mazes", flush=True)

    SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(output_paths["sft_train"], splits["sft_train"])
    write_json(output_paths["sft_eval"], splits["sft_eval"])
    write_parquet(output_paths["rl_train"], splits["rl_train"], "train")
    write_parquet(output_paths["rl_eval"], splits["rl_eval"], "test")

    metadata = {
        "maze_size": MAZE_SIZE,
        "algorithm": "prim",
        "generator_seed": GENERATOR_SEED,
        "candidate_rows_examined": candidates_examined,
        "duplicate_candidates_rejected": duplicate_candidates,
        "disjointness_key": "SHA-256 of the complete question through PATH_START",
        "splits": {
            split_name: {
                "rows": len(splits[split_name]),
                "optimal_path_length": path_length_summary(splits[split_name]),
            }
            for split_name, _ in split_counts
        },
        "files": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for name, path in output_paths.items()
            if name != "metadata"
        },
    }
    temporary_metadata_path = output_paths["metadata"].with_suffix(".json.tmp")
    try:
        temporary_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_metadata_path, output_paths["metadata"])
    finally:
        temporary_metadata_path.unlink(missing_ok=True)

    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
