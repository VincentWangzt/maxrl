"""Extend SFT training with distinct mazes, preserving evaluation and excluding RL."""

import argparse
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from noisy_maze.generate_maze import MazeGenerator, obscure_observation
from noisy_maze.prepare import file_sha256, question_fingerprint, write_json


def extend_sft(source_dir: Path, exclude_rl_dir: Path, train_count: int, generator_seed: int, noise_seed: int) -> dict:
    source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    rl_metadata = json.loads((exclude_rl_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in ("title", "maze_size", "noise_fraction", "algorithm"):
        if source_metadata[key] != rl_metadata[key]:
            raise ValueError(f"SFT and excluded RL metadata disagree on {key}")
    if source_metadata["sft_title"] != source_dir.name or rl_metadata["rl_title"] != exclude_rl_dir.name:
        raise ValueError("Dataset directory names do not match metadata")
    output_title = f"{source_metadata['title']}_sft_{train_count}"
    output_dir = source_dir.parent / output_title
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")

    source_files, split_keys, sft_splits = {}, {}, {}
    seen: set[bytes] = set()
    sources = (
        ("sft_train", source_dir / "train.json", source_metadata),
        ("sft_eval", source_dir / "test.json", source_metadata),
        ("rl_train", exclude_rl_dir / "train.parquet", rl_metadata),
        ("rl_eval", exclude_rl_dir / "test.parquet", rl_metadata),
    )
    for split, path, metadata in sources:
        checksum = file_sha256(path)
        if checksum != metadata["files"][f"{path.parent.name}/{path.name}"]:
            raise ValueError(f"Source checksum does not match metadata: {path}")
        if split.startswith("sft"):
            rows = json.loads(path.read_text(encoding="utf-8"))
            sft_splits[split] = rows
        else:
            rows = pq.read_table(path, columns=["reward_model"]).column("reward_model").to_pylist()
        keys = {question_fingerprint(row["ground_truth"]) for row in rows}
        if len(rows) != metadata["splits"][split]["rows"]:
            raise ValueError(f"Source row count does not match metadata: {path}")
        if len(keys) != len(rows) or keys & seen:
            raise ValueError(f"Duplicate or overlapping underlying mazes in {path}")
        seen.update(keys)
        split_keys[split] = keys
        source_files[split] = {"path": str(path), "sha256": checksum, "rows": len(rows)}

    train_rows = sft_splits["sft_train"]
    original_count = len(train_rows)
    if train_count <= original_count:
        raise ValueError("train_count must exceed the source training split size")
    generator = MazeGenerator(size=source_metadata["maze_size"], seed=generator_seed, algorithm=source_metadata["algorithm"])
    noise_rng = random.Random(noise_seed)
    rejected = 0
    while len(train_rows) < train_count:
        if len(train_rows) - original_count + rejected >= 100 * train_count:
            raise ValueError("Cannot generate enough distinct mazes; increase maze size or reduce train_count")
        generator.generate()
        item = generator.to_text_sequence()
        key = question_fingerprint(item["sequence"])
        if key in seen:
            rejected += 1
            continue
        seen.add(key)
        train_rows.append(obscure_observation(item, source_metadata["maze_size"], str(source_metadata["noise_fraction"]), noise_rng))
        if len(train_rows) % 10_000 == 0 or len(train_rows) == train_count:
            print(f"Prepared {len(train_rows):,}/{train_count:,} distinct SFT training mazes", flush=True)

    output_dir.mkdir(parents=True)
    write_json(output_dir / "train.json", train_rows)
    shutil.copyfile(source_dir / "test.json", output_dir / "test.json")
    if file_sha256(output_dir / "test.json") != source_files["sft_eval"]["sha256"]:
        raise RuntimeError("Evaluation file changed while copying")
    metadata = {
        **source_metadata,
        "sft_title": output_title,
        "rl_title": rl_metadata["rl_title"],
        "extension": {
            "method": "Retain source training rows, append distinct real mazes excluding SFT evaluation and both RL splits",
            "generator_seed": generator_seed,
            "noise_seed": noise_seed,
            "retained_train_rows": original_count,
            "added_train_rows": train_count - original_count,
            "rejected_candidates": rejected,
            "excluded_rows": {split: len(keys) for split, keys in split_keys.items() if split != "sft_train"},
            "train_eval_rl_overlap": 0,
            "evaluation_preserved_exactly": True,
        },
        "source_files": source_files,
        "source_metadata": [
            {"path": str(directory / "metadata.json"), "sha256": file_sha256(directory / "metadata.json")}
            for directory in (source_dir, exclude_rl_dir)
        ],
        "splits": {
            **source_metadata["splits"],
            "rl_train": rl_metadata["splits"]["rl_train"],
            "rl_eval": rl_metadata["splits"]["rl_eval"],
            "sft_train": {
                "rows": train_count,
                "max_sequence_tokens": max(len(row["sequence"].split()) for row in train_rows),
                "max_optimal_path_length": max(row["optimal_path_length"] for row in train_rows),
            },
        },
        "files": {f"{output_title}/{split}.json": file_sha256(output_dir / f"{split}.json") for split in ("train", "test")},
    }
    write_json(output_dir / "metadata.json", metadata)
    return {"output_dir": str(output_dir), "train_rows": train_count, "eval_rows": len(sft_splits["sft_eval"]), **metadata["extension"], "files": metadata["files"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--exclude-rl-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--generator-seed", type=int, required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    print(json.dumps(extend_sft(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
