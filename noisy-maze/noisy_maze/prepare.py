"""Generate disjoint noisy-maze SFT and RL splits without reading maze/ artifacts."""

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from noisy_maze.generate_maze import MazeGenerator, connecting_positions, normalize_noise_fraction, obscure_observation

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetConfig:
    size: int = 17
    noise_fraction: str = "0.1"
    generator_seed: int = 17_202_609
    noise_seed: int = 71_202_609
    sft_train_count: int = 192_000
    sft_eval_count: int = 128
    rl_train_count: int = 1_024
    rl_eval_count: int = 128

    @property
    def title(self) -> str:
        return f"noisy_maze_{self.size}_noise_{normalize_noise_fraction(self.noise_fraction)}"


def question_from_sequence(sequence: str) -> str:
    prefix, separator, _ = sequence.partition("PATH_START")
    if not separator:
        raise ValueError("Maze sequence does not contain PATH_START")
    return f"{prefix.strip()} PATH_START"


def question_fingerprint(sequence: str) -> bytes:
    return hashlib.sha256(question_from_sequence(sequence).encode("utf-8")).digest()


def make_rl_record(item: dict, split: str, index: int, title: str) -> dict:
    question = question_from_sequence(item["sequence"])
    return {
        "data_source": title,
        "prompt": [{"content": question, "role": "user"}],
        "ability": "noisy_maze",
        "reward_model": {"ground_truth": item["ground_truth"], "style": "rule"},
        "extra_info": {
            "answer": item["sequence"],
            "question": question,
            "index": index,
            "split": split,
            "optimal_path_length": item["optimal_path_length"],
            "noise_fraction": item["noise_fraction"],
            "masked_positions": item["masked_positions"],
        },
    }


def generate_splits(config: DatasetConfig) -> tuple[dict[str, list[dict]], int]:
    # Validate even if the requested counts would otherwise skip a loop.
    normalize_noise_fraction(config.noise_fraction)
    generator = MazeGenerator(size=config.size, seed=config.generator_seed, algorithm="prim")
    noise_rng = random.Random(config.noise_seed)
    split_counts = {
        "sft_train": config.sft_train_count,
        "sft_eval": config.sft_eval_count,
        "rl_train": config.rl_train_count,
        "rl_eval": config.rl_eval_count,
    }
    if any(count <= 0 for count in split_counts.values()):
        raise ValueError("All split sizes must be positive")
    splits = {name: [] for name in split_counts}
    seen: set[bytes] = set()
    duplicates = 0
    candidates = 0
    max_candidates = 100 * sum(split_counts.values())
    for name, count in split_counts.items():
        while len(splits[name]) < count:
            candidates += 1
            if candidates > max_candidates:
                raise ValueError("Cannot generate enough distinct mazes; increase maze size or reduce split sizes")
            generator.generate()
            item = generator.to_text_sequence()
            fingerprint = question_fingerprint(item["sequence"])
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            splits[name].append(obscure_observation(item, config.size, config.noise_fraction, noise_rng))
            if len(splits[name]) % 10_000 == 0:
                print(f"Generated {len(splits[name]):,}/{count:,} {name} mazes", flush=True)
    return splits, duplicates


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_datasets(config: DatasetConfig, output_root: Path) -> dict:
    sft_title = f"{config.title}_sft_{config.sft_train_count}"
    rl_title = f"{config.title}_rl_{config.rl_train_count}"
    sft_dir, rl_dir = output_root / sft_title, output_root / rl_title
    for directory in (sft_dir, rl_dir):
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite existing dataset: {directory}")
    splits, duplicates = generate_splits(config)
    sft_dir.mkdir(parents=True)
    rl_dir.mkdir(parents=True)
    files = {"sft_train": sft_dir / "train.json", "sft_eval": sft_dir / "test.json", "rl_train": rl_dir / "train.parquet", "rl_eval": rl_dir / "test.parquet"}
    for name, path in files.items():
        if name.startswith("sft"):
            write_json(path, splits[name])
        else:
            split = "train" if name == "rl_train" else "test"
            rows = [make_rl_record(item, split, i, config.title) for i, item in enumerate(splits[name])]
            temporary = path.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(rows), temporary, compression="snappy")
            os.replace(temporary, path)
    edge_count = len(connecting_positions(config.size))
    masked_count = len(splits["sft_train"][0]["masked_positions"])
    metadata = {
        "title": config.title,
        "sft_title": sft_title,
        "rl_title": rl_title,
        "maze_size": config.size,
        "algorithm": "prim",
        "generator_seed": config.generator_seed,
        "noise_seed": config.noise_seed,
        "noise_fraction": float(normalize_noise_fraction(config.noise_fraction)),
        "connecting_positions": edge_count,
        "masked_positions_per_maze": masked_count,
        "actual_noise_fraction": masked_count / edge_count,
        "mask_policy": "floor(fraction * interior connecting positions), uniform without replacement, fixed per row",
        "disjointness_key": "SHA-256 of the unclouded question through PATH_START, across all four splits",
        "duplicate_candidates_rejected": duplicates,
        "splits": {
            name: {
                "rows": len(items),
                "max_sequence_tokens": max(len(item["sequence"].split()) for item in items),
                "max_optimal_path_length": max(item["optimal_path_length"] for item in items),
            }
            for name, items in splits.items()
        },
        "files": {str(path.relative_to(output_root)): file_sha256(path) for path in files.values()},
    }
    write_json(sft_dir / "metadata.json", metadata)
    write_json(rl_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=17)
    parser.add_argument("--noise-fraction", default="0.1")
    parser.add_argument("--generator-seed", type=int, default=17_202_609)
    parser.add_argument("--noise-seed", type=int, default=71_202_609)
    parser.add_argument("--sft-train-count", type=int, default=192_000)
    parser.add_argument("--sft-eval-count", type=int, default=128)
    parser.add_argument("--rl-train-count", type=int, default=1_024)
    parser.add_argument("--rl-eval-count", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "data")
    args = vars(parser.parse_args())
    output_root = args.pop("output_root")
    print(json.dumps(prepare_datasets(DatasetConfig(**args), output_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
