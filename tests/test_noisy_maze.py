"""Focused CPU validation; run on cmu-L40-live with PYTHONPATH=$PWD/noisy-maze:$PWD."""

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from noisy_maze.curate import curate_subset
from noisy_maze.generate_maze import MazeGenerator, connecting_positions, normalize_noise_fraction, obscure_observation
from noisy_maze.prepare import DatasetConfig, generate_splits, make_rl_record, prepare_datasets, question_fingerprint
from noisy_maze.reward import compute_optimal_length, compute_score, compute_scores, validate_solution
from noisy_maze.sft import MAZE_VOCAB, MazeSFTDataset, MazeSFTTrainer, collate_fn, create_model_from_scratch, estimate_pass_at_k
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture
def hidden_wall_item():
    # RIGHT crosses a hidden WALL; DOWN follows the real path.
    grid = "WALL WALL WALL WALL WALL NEWLINE WALL START WALL PATH WALL NEWLINE "
    grid += "WALL PATH WALL PATH WALL NEWLINE WALL PATH PATH GOAL WALL NEWLINE WALL WALL WALL WALL WALL NEWLINE"
    sequence = f"<bos> GRID_START {grid} GRID_END PATH_START DOWN DOWN RIGHT RIGHT DONE <eos>"
    return obscure_observation({"sequence": sequence, "optimal_path_length": 4}, 5, "1", random.Random(42))


@pytest.mark.parametrize("size,masked_count", [(17, 11), (23, 22)])
def test_exact_edge_mask_preserves_truth(size, masked_count):
    generator = MazeGenerator(size=size, seed=42)
    generator.generate()
    real_grid = generator.grid.copy()
    clean = generator.to_text_sequence()
    item = obscure_observation(clean, size, "0.1", random.Random(17))
    np.testing.assert_array_equal(generator.grid, real_grid)
    assert item["ground_truth"] == clean["sequence"]
    assert item["sequence"].split().count("UNKNOWN") == masked_count
    assert len(item["masked_positions"]) == masked_count
    observed, truth = item["sequence"].split(), item["ground_truth"].split()
    offset = truth.index("GRID_START") + 1
    changed = {i for i, (a, b) in enumerate(zip(observed, truth, strict=True)) if a != b}
    assert changed == {offset + r * (size + 1) + c for r, c in item["masked_positions"]}
    assert all(truth[i] in {"WALL", "PATH"} and observed[i] == "UNKNOWN" for i in changed)
    assert set(map(tuple, item["masked_positions"])) <= set(connecting_positions(size))
    fully_hidden = obscure_observation(clean, size, "1", random.Random(17))
    assert fully_hidden["sequence"].split().count("UNKNOWN") == len(connecting_positions(size))
    assert obscure_observation(clean, size, "0", random.Random(17))["sequence"] == clean["sequence"]
    solution = item["sequence"].partition("PATH_START")[2]
    assert compute_score("noisy_maze_test", solution, item["ground_truth"]) == 1
    assert compute_optimal_length(item["ground_truth"]) == item["optimal_path_length"]


def test_noise_validation_and_normalized_title():
    assert normalize_noise_fraction("0.10") == "0.1"
    assert "noise_0.1" in DatasetConfig(noise_fraction="0.10").title
    for value in ("-0.1", "1.1", "NaN", "Infinity"):
        with pytest.raises(ValueError):
            normalize_noise_fraction(value)


def test_disjoint_reproducible_splits_and_noise_stream():
    config = DatasetConfig(size=9, sft_train_count=8, sft_eval_count=2, rl_train_count=4, rl_eval_count=2)
    splits, _ = generate_splits(config)
    assert generate_splits(config)[0] == splits
    fingerprints = [question_fingerprint(item["ground_truth"]) for items in splits.values() for item in items]
    assert len(fingerprints) == len(set(fingerprints)) == 16
    noiseless, _ = generate_splits(replace(config, noise_fraction="0"))
    for name, items in splits.items():
        assert [item["ground_truth"] for item in items] == [item["ground_truth"] for item in noiseless[name]]
        for i, item in enumerate(items):
            row = make_rl_record(item, "train", i, config.title)
            assert "UNKNOWN" in row["prompt"][0]["content"]
            assert "UNKNOWN" not in row["reward_model"]["ground_truth"]


def test_dataset_files_and_metadata(tmp_path):
    config = DatasetConfig(size=9, sft_train_count=4, sft_eval_count=2, rl_train_count=4, rl_eval_count=2)
    metadata = prepare_datasets(config, tmp_path)
    assert metadata["noise_fraction"] == 0.1
    assert metadata["masked_positions_per_maze"] == 2
    assert metadata["actual_noise_fraction"] == 2 / 24
    assert pq.read_table(tmp_path / metadata["rl_title"] / "train.parquet").num_rows == 4
    assert len(json.loads((tmp_path / metadata["sft_title"] / "train.json").read_text())) == 4
    assert json.loads((tmp_path / metadata["rl_title"] / "metadata.json").read_text()) == metadata
    with pytest.raises(FileExistsError):
        prepare_datasets(config, tmp_path)


def test_curated_subset_preserves_observations_truth_and_evaluation(tmp_path):
    config = DatasetConfig(size=9, sft_train_count=4, sft_eval_count=2, rl_train_count=8, rl_eval_count=2)
    parent_metadata = prepare_datasets(config, tmp_path)
    source_dir = tmp_path / parent_metadata["rl_title"]
    result = curate_subset(source_dir, train_count=4, seed=1024)
    output_dir = Path(result["output_dir"])
    metadata = json.loads((output_dir / "metadata.json").read_text())
    expected_indices = random.Random(1024).sample(range(8), 4)
    assert metadata["selected_source_row_indices"] == expected_indices
    assert pq.read_table(output_dir / "train.parquet").equals(pq.read_table(source_dir / "train.parquet").take(expected_indices))
    assert (output_dir / "test.parquet").read_bytes() == (source_dir / "test.parquet").read_bytes()
    assert result["train_rows"] == metadata["splits"]["rl_train"]["rows"] == 4
    assert metadata["noise_fraction"] == 0.1
    assert metadata["rl_title"].endswith("_rl_4")
    with pytest.raises(FileExistsError):
        curate_subset(source_dir, train_count=4, seed=1024)


def test_true_maze_reward_rejects_hidden_wall_and_invalid_responses(hidden_wall_item):
    truth = hidden_wall_item["ground_truth"]
    good = "DOWN DOWN RIGHT RIGHT DONE <eos>"
    bad = "RIGHT RIGHT DOWN DOWN DONE"
    assert validate_solution(good, truth) == (True, "success")
    assert validate_solution(bad, truth) == (False, "hit_wall")
    assert validate_solution("DOWN DOWN RIGHT RIGHT", truth) == (False, "missing_done")
    assert validate_solution("UNKNOWN DOWN DOWN RIGHT RIGHT DONE", truth) == (False, "invalid_action")
    assert validate_solution(hidden_wall_item["sequence"], truth) == (False, "invalid_action")
    assert compute_scores(["noisy_maze_test"] * 2, [good, bad], [truth] * 2, [None] * 2) == [1.0, 0.0]
    with pytest.raises(ValueError, match="unclouded"):
        validate_solution(good, hidden_wall_item["sequence"])


@pytest.fixture
def tiny_model_path(tmp_path):
    assert not torch.cuda.is_available(), "Run these tests with CUDA_VISIBLE_DEVICES=''"
    args = SimpleNamespace(output_dir=str(tmp_path / "model"), hidden_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, max_position_embeddings=128)
    return create_model_from_scratch(args)


def test_tokenizer_labels_and_cpu_optimizer_step(tmp_path, tiny_model_path, hidden_wall_item):
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_path)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_path).float()
    assert tokenizer.encode("UNKNOWN", add_special_tokens=False) == [18]
    assert tokenizer.unk_token_id != MAZE_VOCAB["UNKNOWN"]
    assert model.config.vocab_size == len(tokenizer) == 32
    assert model.get_input_embeddings().weight.shape[0] == 32
    json_path, parquet_path = tmp_path / "train.json", tmp_path / "train.parquet"
    json_path.write_text(json.dumps([hidden_wall_item]))
    pq.write_table(pa.Table.from_pylist([make_rl_record(hidden_wall_item, "train", 0, "noisy_maze_test")]), parquet_path)
    dataset = MazeSFTDataset(str(json_path), tokenizer, max_length=128)
    parquet_dataset = MazeSFTDataset(str(parquet_path), tokenizer, max_length=128)
    torch.testing.assert_close(dataset[0]["input_ids"], parquet_dataset[0]["input_ids"])
    assert dataset.get_ground_truth(0) == hidden_wall_item["ground_truth"]
    assert "UNKNOWN" in dataset.get_prompt(0)
    sample = dataset[0]
    path_index = sample["input_ids"].tolist().index(MAZE_VOCAB["PATH_START"])
    assert torch.all(sample["labels"][: path_index + 1] == -100)
    assert MAZE_VOCAB["UNKNOWN"] in sample["input_ids"]
    assert MAZE_VOCAB["UNKNOWN"] not in sample["labels"]
    batch = collate_fn([sample, sample, sample], tokenizer.pad_token_id)
    trainer = MazeSFTTrainer.__new__(MazeSFTTrainer)
    trainer.model, trainer.device = model, torch.device("cpu")
    trainer.args = SimpleNamespace(micro_batch_size=2)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    before = model.get_input_embeddings().weight.detach().clone()
    loss = trainer.train_step(batch)
    assert math.isfinite(loss) and loss > 0
    assert not torch.equal(before, model.get_input_embeddings().weight)
    with pytest.raises(ValueError, match="exceeding"):
        MazeSFTDataset(str(json_path), tokenizer, max_length=10)


def test_generative_evaluation_uses_truth_and_logs_through_256(tmp_path, tiny_model_path, hidden_wall_item):
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_path)
    data_path = tmp_path / "eval.json"
    data_path.write_text(json.dumps([hidden_wall_item]))
    dataset = MazeSFTDataset(str(data_path), tokenizer, 128)

    class FakeGenerator:
        config = SimpleNamespace(max_position_embeddings=128)

        def __init__(self):
            self.count = 0
            self.chunk_sizes = []

        def eval(self):
            return self

        def generate(self, input_ids, num_return_sequences, **kwargs):
            assert MAZE_VOCAB["UNKNOWN"] in input_ids
            self.chunk_sizes.append(num_return_sequences)
            outputs = []
            for _ in range(num_return_sequences):
                solution = "DOWN DOWN RIGHT RIGHT DONE" if self.count % 2 == 0 else "RIGHT RIGHT DOWN DOWN DONE"
                response = tokenizer.encode(solution, add_special_tokens=False, return_tensors="pt")
                outputs.append(torch.cat([input_ids[0], response[0]]))
                self.count += 1
            return torch.stack(outputs)

    trainer = MazeSFTTrainer.__new__(MazeSFTTrainer)
    trainer.model, trainer.tokenizer, trainer.val_dataset = FakeGenerator(), tokenizer, dataset
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(seed=42, eval_generation_batch_size=31)
    metrics = trainer.generative_evaluate(num_samples=1, n_samples_per_prompt=256, max_new_tokens=16)
    assert trainer.model.chunk_sizes == [31] * 8 + [8]
    assert metrics["eval/pass@1"] == pytest.approx(0.5)
    assert metrics["eval/hit_wall_rate"] == 0.5
    assert metrics["eval/pass@256"] == metrics["eval/optimal_pass@256"] == 1
    assert metrics["eval/generations"] == 256
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        assert metrics[f"eval/pass@{k}"] == pytest.approx(1 - math.comb(128, k) / math.comb(256, k))
    assert estimate_pass_at_k(256, 0, 256) == 0
    with pytest.raises(ValueError):
        estimate_pass_at_k(128, 1, 256)


def test_rl_custom_reward_loader(hidden_wall_item, tiny_model_path):
    from verl import DataProto
    from verl.trainer.ppo.reward import load_reward_manager

    reward_path = Path(__file__).resolve().parents[1] / "noisy-maze/noisy_maze/reward.py"
    config = OmegaConf.create(
        {
            "custom_reward_function": {"path": str(reward_path), "name": "compute_scores"},
            "reward_model": {"reward_manager": "batch"},
            "data": {"reward_fn_key": "data_source"},
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_path)
    prompt = hidden_wall_item["sequence"].partition("PATH_START")[0] + "PATH_START"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").repeat(2, 1)
    response_ids = tokenizer(["DOWN DOWN RIGHT RIGHT DONE", "RIGHT RIGHT DOWN DOWN DONE"], return_tensors="pt")["input_ids"]
    data = DataProto.from_dict(
        tensors={
            "prompts": prompt_ids,
            "responses": response_ids,
            "attention_mask": torch.ones((2, prompt_ids.shape[1] + response_ids.shape[1]), dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["noisy_maze_test"] * 2, dtype=object),
            "reward_model": np.array([{"ground_truth": hidden_wall_item["ground_truth"]}] * 2, dtype=object),
            "extra_info": np.array([{}] * 2, dtype=object),
        },
    )
    manager = load_reward_manager(config, tokenizer, num_examine=0)
    result = manager(data, return_dict=True)
    assert result["reward_tensor"].sum(dim=1).tolist() == [1.0, 0.0]
    assert data.batch["acc"].tolist() == [1.0, 0.0]


def test_evaluation_metrics_share_training_wandb_step(tmp_path, monkeypatch):
    logged, saved = [], []
    monkeypatch.setattr("noisy_maze.sft.wandb.log", lambda metrics, step: logged.append((step, metrics)))
    monkeypatch.setattr("noisy_maze.sft.wandb.finish", lambda: None)
    trainer = MazeSFTTrainer.__new__(MazeSFTTrainer)
    trainer.args = SimpleNamespace(num_epochs=1, max_steps=2, save_steps=1, eval_steps=1, use_generative_eval=True, eval_samples=128, n_samples_per_prompt=256, eval_temperature=1.0, eval_max_new_tokens=180, output_dir=str(tmp_path))
    trainer.train_loader = [None] * 4
    trainer.train_step = lambda batch: 0.5
    trainer.validate_loss = lambda: 0.4
    trainer.generative_evaluate = lambda **kwargs: {"eval/pass@256": 0.75}
    trainer.save_checkpoint = saved.append
    trainer.scheduler = SimpleNamespace(get_last_lr=lambda: [5e-4])
    trainer.global_step, trainer.use_wandb = 0, True
    trainer.train()
    assert [step for step, _ in logged] == saved == [1, 2]
    assert all(metrics["eval/pass@256"] == 0.75 and metrics["train/loss"] == 0.5 for _, metrics in logged)
    local_metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in local_metrics] == [1, 2]
    assert all(record["eval/pass@256"] == 0.75 for record in local_metrics)
