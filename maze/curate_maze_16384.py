"""Build a fresh 17x17 maze split disjoint from the original 1M corpus."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from maze.generate_maze import MazeGenerator


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    REPO_ROOT / "maze/data/train.parquet",
    REPO_ROOT / "maze/data/test.parquet",
)
OUTPUT_DIR = REPO_ROOT / "maze/data/maze_17_16384"
TRAIN_COUNT = 16_384
EVAL_COUNT = 256
MAZE_SIZE = 17
CANDIDATE_SEED = 20_260_903
SPLIT_SEED = 16_384
HASH_BATCH_SIZE = 8_192


def question_fingerprint(question: str) -> bytes:
    return hashlib.sha256(question.encode("utf-8")).digest()


def load_question_fingerprints(path: Path) -> tuple[set[bytes], int]:
    fingerprints: set[bytes] = set()
    row_count = 0
    parquet_file = pq.ParquetFile(path)

    for batch in parquet_file.iter_batches(
        batch_size=HASH_BATCH_SIZE,
        columns=["extra_info"],
    ):
        for extra_info in batch.column(0).to_pylist():
            question = extra_info.get("question")
            if not isinstance(question, str) or not question:
                raise ValueError(f"Missing question in {path} at row {row_count}")
            fingerprints.add(question_fingerprint(question))
            row_count += 1

    return fingerprints, row_count


def question_from_answer(answer: str) -> str:
    prefix, separator, _ = answer.partition("PATH_START")
    if not separator:
        raise ValueError("Generated maze answer does not contain PATH_START")
    return f"{prefix.strip()} PATH_START"


def make_record(answer: str, split: str, index: int) -> dict:
    question = question_from_answer(answer)
    return {
        "data_source": "maze_17",
        "prompt": [{"content": question, "role": "user"}],
        "ability": "maze",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "answer": answer,
            "index": index,
            "question": question,
            "split": split,
        },
    }


def write_parquet(path: Path, answers: list[str], split: str) -> None:
    records = [make_record(answer, split, index) for index, answer in enumerate(answers)]
    table = pa.Table.from_pylist(records)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pq.write_table(table, temporary_path, compression="snappy")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for source_file in SOURCE_FILES:
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing contamination reference: {source_file}")

    train_path = OUTPUT_DIR / "train.parquet"
    eval_path = OUTPUT_DIR / "test.parquet"
    metadata_path = OUTPUT_DIR / "metadata.json"
    for output_path in (train_path, eval_path, metadata_path):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    contamination_fingerprints: set[bytes] = set()
    reference_rows = 0
    for source_file in SOURCE_FILES:
        print(f"Scanning contamination reference {source_file}...", flush=True)
        source_fingerprints, source_rows = load_question_fingerprints(source_file)
        contamination_fingerprints.update(source_fingerprints)
        reference_rows += source_rows

    target_count = TRAIN_COUNT + EVAL_COUNT
    generator = MazeGenerator(size=MAZE_SIZE, seed=CANDIDATE_SEED, algorithm="prim")
    curated_answers: list[str] = []
    curated_fingerprints: set[bytes] = set()
    candidate_count = 0
    rejected_contamination = 0
    rejected_duplicate = 0

    while len(curated_answers) < target_count:
        generator.generate()
        generated_item = generator.to_text_sequence()
        if generated_item is None:
            continue

        candidate_count += 1
        answer = generated_item["sequence"]
        fingerprint = question_fingerprint(question_from_answer(answer))
        if fingerprint in contamination_fingerprints:
            rejected_contamination += 1
            continue
        if fingerprint in curated_fingerprints:
            rejected_duplicate += 1
            continue

        curated_answers.append(answer)
        curated_fingerprints.add(fingerprint)

    random.Random(SPLIT_SEED).shuffle(curated_answers)
    eval_answers = curated_answers[:EVAL_COUNT]
    train_answers = curated_answers[EVAL_COUNT:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(train_path, train_answers, "train")
    write_parquet(eval_path, eval_answers, "test")

    written_train_fingerprints, written_train_rows = load_question_fingerprints(train_path)
    written_eval_fingerprints, written_eval_rows = load_question_fingerprints(eval_path)
    if written_train_rows != TRAIN_COUNT or len(written_train_fingerprints) != TRAIN_COUNT:
        raise RuntimeError("Training split row count or uniqueness validation failed")
    if written_eval_rows != EVAL_COUNT or len(written_eval_fingerprints) != EVAL_COUNT:
        raise RuntimeError("Evaluation split row count or uniqueness validation failed")
    if written_train_fingerprints & written_eval_fingerprints:
        raise RuntimeError("Training and evaluation splits overlap")
    if (written_train_fingerprints | written_eval_fingerprints) & contamination_fingerprints:
        raise RuntimeError("Curated data overlaps the original 1M corpus")

    metadata = {
        "maze_size": MAZE_SIZE,
        "algorithm": "prim",
        "candidate_seed": CANDIDATE_SEED,
        "split_seed": SPLIT_SEED,
        "train_rows": written_train_rows,
        "eval_rows": written_eval_rows,
        "candidate_rows_examined": candidate_count,
        "rejected_as_original_1m_contamination": rejected_contamination,
        "rejected_as_curated_duplicate": rejected_duplicate,
        "decontamination": {
            "key": "SHA-256 of the full question through PATH_START",
            "reference_files": [str(path.relative_to(REPO_ROOT)) for path in SOURCE_FILES],
            "reference_rows": reference_rows,
            "reference_unique_questions": len(contamination_fingerprints),
            "scope": "Union of the original 1M train and test corpora; this is stronger than excluding an unrecoverable shuffled SFT minibatch subset.",
        },
        "files": {
            "train.parquet": file_sha256(train_path),
            "test.parquet": file_sha256(eval_path),
        },
    }
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    try:
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata_path, metadata_path)
    finally:
        temporary_metadata_path.unlink(missing_ok=True)

    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
