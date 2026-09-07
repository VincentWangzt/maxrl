import argparse
import json
import logging
import os
import random
import time
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch
import wandb
from tokenizers import AddedToken, Tokenizer, pre_tokenizers
from tokenizers import models as tok_models
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, get_cosine_schedule_with_warmup

from noisy_maze.prepare import question_from_sequence
from noisy_maze.reward import compute_optimal_length, parse_actions, parse_ground_truth, validate_solution

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MazeSFTDataset(Dataset):
    """Train only on clouded sequences; retain the real maze separately for evaluation."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        start_time = time.time()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # 加载数据
        if data_path.endswith(".parquet"):
            logger.info(f"Reading parquet file: {data_path}")
            for row in pq.read_table(data_path).to_pylist():
                self.examples.append(self._process_sequence(row["extra_info"]["answer"], row["reward_model"]["ground_truth"]))
        else:
            # JSON格式
            logger.info(f"Reading JSON file: {data_path}")
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Processing {len(data)} items...")
            for item in tqdm(data, desc=f"Loading {os.path.basename(data_path)}"):
                self.examples.append(self._process_sequence(item["sequence"], item["ground_truth"]))
        if not self.examples:
            raise ValueError(f"Empty dataset: {data_path}")

        elapsed_time = time.time() - start_time
        logger.info(f"Loaded {len(self.examples)} examples from {data_path} in {elapsed_time:.2f}s")

    def _process_sequence(self, sequence: str, ground_truth: str) -> Dict:
        """处理序列，构建input_ids和labels"""
        # 编码完整序列
        input_ids = self.tokenizer.encode(sequence, add_special_tokens=False)

        if len(input_ids) > self.max_length:
            raise ValueError(f"Sequence has {len(input_ids)} tokens, exceeding max_length={self.max_length}")
        if self.tokenizer.unk_token_id in input_ids:
            raise ValueError("Dataset sequence contains tokens outside the model vocabulary")
        parse_ground_truth(ground_truth)
        if sequence.partition("PATH_START")[2] != ground_truth.partition("PATH_START")[2]:
            raise ValueError("Clouded training target differs from the ground-truth solution")

        # 找到PATH_START位置，只预测路径部分
        path_start_token = "PATH_START"
        path_start_id = self.tokenizer.encode(path_start_token, add_special_tokens=False)

        if len(path_start_id) != 1:
            raise ValueError("PATH_START must encode as one token")

        path_start_id = path_start_id[0]

        path_start_idx = input_ids.index(path_start_id)

        # 构建labels：PATH_START之前（包括）设为-100
        labels = [-100] * (path_start_idx + 1) + input_ids[path_start_idx + 1 :]

        # 保存prompt用于生成式评估
        prompt = question_from_sequence(sequence)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt": prompt,
            "ground_truth": ground_truth,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "input_ids": example["input_ids"],
            "labels": example["labels"],
            "attention_mask": torch.ones_like(example["input_ids"]),
        }

    def get_prompt(self, idx) -> str:
        return self.examples[idx]["prompt"]

    def get_ground_truth(self, idx) -> str:
        return self.examples[idx]["ground_truth"]


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


MAZE_VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "GRID_START": 4,
    "GRID_END": 5,
    "PATH_START": 6,
    "DONE": 7,
    "PATH": 8,
    "WALL": 9,
    "GOAL": 10,
    "START": 11,
    "NEWLINE": 12,
    "UP": 13,
    "DOWN": 14,
    "LEFT": 15,
    "RIGHT": 16,
    "\n": 17,
    "UNKNOWN": 18,
}


def create_model_from_scratch(args):
    num_reserved = 13
    vocab_size = len(MAZE_VOCAB) + num_reserved

    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        dtype="bfloat16",
        use_cache=True,
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    reserved = {f"RESERVED_{i}": len(MAZE_VOCAB) + i for i in range(num_reserved)}
    full_vocab = {**MAZE_VOCAB, **reserved}
    tok = Tokenizer(tok_models.WordLevel(vocab=full_vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    for t in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        tok.add_special_tokens([AddedToken(t, special=True)])

    save_dir = os.path.join(args.output_dir, "init_model")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save(os.path.join(save_dir, "tokenizer.json"))
    for fname, data in [
        (
            "tokenizer_config.json",
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": args.max_position_embeddings,
            },
        ),
        (
            "special_tokens_map.json",
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            },
        ),
    ]:
        with open(os.path.join(save_dir, fname), "w") as f:
            json.dump(data, f, indent=2)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized from scratch: {num_params:,} params, vocab_size={vocab_size}, saved to {save_dir}")
    return save_dir


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate: 1 - C(n-c, k) / C(n, k)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Expected 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


class MazeSFTTrainer:
    """Maze SFT Trainer with Generative Evaluation"""

    def __init__(self, args):
        self.args = args
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = args.model_path
        if model_path is None:
            logger.info("No model_path provided, creating model from scratch...")
            model_path = create_model_from_scratch(args)

        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        if self.tokenizer.encode("UNKNOWN", add_special_tokens=False) != [MAZE_VOCAB["UNKNOWN"]]:
            raise ValueError("Model must use the noisy-maze vocabulary with UNKNOWN at token ID 18")
        if self.model.get_input_embeddings().num_embeddings < len(self.tokenizer):
            raise ValueError("Model embeddings do not cover the tokenizer vocabulary")

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载数据集
        logger.info("Loading datasets...")
        dataset_load_start = time.time()
        self.train_dataset = MazeSFTDataset(args.train_data, self.tokenizer, args.max_length)
        self.val_dataset = MazeSFTDataset(args.val_data, self.tokenizer, args.max_length)
        dataset_load_time = time.time() - dataset_load_start
        logger.info(f"All datasets loaded in {dataset_load_time:.2f}s")

        # 创建DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_token_id),
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.micro_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_token_id),
            num_workers=4,
            pin_memory=True,
        )

        # 优化器和调度器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        available_steps = len(self.train_loader) * args.num_epochs
        if args.max_steps is not None and args.max_steps > available_steps:
            raise ValueError(f"max_steps={args.max_steps} exceeds the {available_steps} steps available across num_epochs={args.num_epochs}")
        total_steps = args.max_steps if args.max_steps is not None else available_steps
        warmup_steps = int(total_steps * args.warmup_ratio)

        # 选择学习率调度器
        if args.lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif args.lr_scheduler == "constant":
            # 常数学习率（带warmup）
            from transformers import get_constant_schedule_with_warmup

            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")

        # 验证器

        # 日志
        self.global_step = 0

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 初始化wandb
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            wandb.init(
                project=args.project_name,
                name=args.experiment_name,
                config={
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "micro_batch_size": args.micro_batch_size,
                    "num_epochs": args.num_epochs,
                    "max_steps": args.max_steps,
                    "max_length": args.max_length,
                    "model_path": args.model_path,
                    "train_data": args.train_data,
                    "val_data": args.val_data,
                    "optimizer": "AdamW",
                    "eval_steps": args.eval_steps,
                    "n_samples_per_prompt": args.n_samples_per_prompt,
                    "eval_generation_batch_size": args.eval_generation_batch_size,
                    "seed": args.seed,
                },
            )
            logger.info("Wandb initialized successfully")

        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练"""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 梯度累积
        micro_batch_size = self.args.micro_batch_size
        batch_size = input_ids.size(0)
        target_counts = (labels[:, 1:] != -100).sum()

        self.optimizer.zero_grad()
        total_loss = 0.0

        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)

            outputs = self.model(
                input_ids=input_ids[i:end_idx],
                attention_mask=attention_mask[i:end_idx],
                labels=labels[i:end_idx],
            )

            # Weight by predicted tokens so accumulation matches a full-batch causal LM loss.
            weight = (labels[i:end_idx, 1:] != -100).sum() / target_counts
            loss = outputs.loss * weight
            loss.backward()
            total_loss += loss.item()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        return total_loss

    @torch.no_grad()
    def validate_loss(self) -> float:
        """计算验证集loss"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def generative_evaluate(
        self,
        num_samples: int = 128,
        n_samples_per_prompt: int = 256,
        temperature: float = 1.0,
        max_new_tokens: int = 180,
    ) -> Dict[str, float]:
        """Sample paths from clouded prompts, then score only against the real maze."""
        self.model.eval()
        success_counts, optimal_counts = [], []
        error_counts = {}
        total_generations = 0
        # Use the same subset at every evaluation, independent of training RNG state.
        indices = np.random.default_rng(self.args.seed).choice(len(self.val_dataset), min(num_samples, len(self.val_dataset)), replace=False)
        done_ids = self.tokenizer.encode("DONE", add_special_tokens=False)
        if len(done_ids) != 1:
            raise ValueError("DONE must encode as one token")
        for idx in tqdm(indices, desc="Generative Eval"):
            prompt = self.val_dataset.get_prompt(idx)
            ground_truth = self.val_dataset.get_ground_truth(idx)
            optimal_len = compute_optimal_length(ground_truth)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
            if input_ids.shape[1] + max_new_tokens > self.model.config.max_position_embeddings:
                raise ValueError("Evaluation prompt plus generation budget exceeds model context length")
            success_count, optimal_count = 0, 0
            for start in range(0, n_samples_per_prompt, self.args.eval_generation_batch_size):
                chunk_size = min(self.args.eval_generation_batch_size, n_samples_per_prompt - start)
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=0,
                    top_p=1.0,
                    num_return_sequences=chunk_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=done_ids[0],
                )
                for output in output_ids:
                    # Never parse the echoed clouded grid (or a model-supplied grid) as truth.
                    solution = self.tokenizer.decode(output[input_ids.shape[1] :], skip_special_tokens=False)
                    success, reason = validate_solution(solution, ground_truth)
                    total_generations += 1
                    if success:
                        success_count += 1
                        actions, _ = parse_actions(solution)
                        optimal_count += int(len(actions) == optimal_len)
                    else:
                        error_counts[reason] = error_counts.get(reason, 0) + 1
            success_counts.append(success_count)
            optimal_counts.append(optimal_count)
        n = n_samples_per_prompt
        k_values = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, n})
        metrics = {}
        for k in k_values:
            if k <= n:
                metrics[f"eval/pass@{k}"] = float(np.mean([estimate_pass_at_k(n, c, k) for c in success_counts]))
                metrics[f"eval/optimal_pass@{k}"] = float(np.mean([estimate_pass_at_k(n, c, k) for c in optimal_counts]))
        for reason in ("missing_done", "no_actions", "invalid_action", "out_of_bounds", "hit_wall", "not_at_goal"):
            metrics[f"eval/{reason}_rate"] = error_counts.get(reason, 0) / total_generations
        metrics["eval/avg_success_rate"] = sum(success_counts) / total_generations
        metrics["eval/avg_optimal_rate"] = sum(optimal_counts) / total_generations
        metrics["eval/prompts"] = len(indices)
        metrics["eval/generations"] = total_generations
        return metrics

    def save_checkpoint(self, step: int):
        """保存checkpoint"""
        path = os.path.join(self.args.output_dir, f"ckpt-{step}")
        os.makedirs(path, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        logger.info(f"Saved checkpoint to {path}")

    def train(self):
        """训练主循环"""
        logger.info("Starting training...")

        reached_max_steps = False
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.num_epochs}")

            for batch in pbar:
                self.global_step += 1

                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1

                # 更新进度条
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                step_metrics = {
                    "train/loss": loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1,
                }

                # 先保存checkpoint，避免生成式评估失败时丢失训练状态
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(self.global_step)

                # 评估
                if self.global_step % self.args.eval_steps == 0:
                    val_loss = self.validate_loss()
                    logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}")

                    eval_metrics = {"eval/val_loss": val_loss}

                    if self.args.use_generative_eval:
                        metrics = self.generative_evaluate(
                            num_samples=self.args.eval_samples,
                            n_samples_per_prompt=self.args.n_samples_per_prompt,
                            temperature=self.args.eval_temperature,
                            max_new_tokens=self.args.eval_max_new_tokens,
                        )
                        eval_metrics.update(metrics)
                        logger.info(f"Step {self.global_step} - Pass@1={metrics.get('eval/pass@1', 0):.4f}, Pass@2={metrics.get('eval/pass@2', 0):.4f}, Pass@4={metrics.get('eval/pass@4', 0):.4f}, Pass@8={metrics.get('eval/pass@8', 0):.4f}")
                        logger.info(f"Step {self.global_step} - Optimal Pass@1={metrics.get('eval/optimal_pass@1', 0):.4f}, Avg Success={metrics.get('eval/avg_success_rate', 0):.4f}")
                        logger.info(f"Step {self.global_step} - Pass@{self.args.n_samples_per_prompt}={metrics.get(f'eval/pass@{self.args.n_samples_per_prompt}', 0):.4f}, Optimal Pass@{self.args.n_samples_per_prompt}={metrics.get(f'eval/optimal_pass@{self.args.n_samples_per_prompt}', 0):.4f}")

                    step_metrics.update(eval_metrics)
                    logger.info("Step %s evaluation: %s", self.global_step, json.dumps(eval_metrics, sort_keys=True))
                    with open(os.path.join(self.args.output_dir, "metrics.jsonl"), "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": self.global_step, **eval_metrics}) + "\n")

                # Commit training and evaluation together so W&B does not drop a second
                # log call at an already committed optimizer step.
                if self.use_wandb:
                    wandb.log(step_metrics, step=self.global_step)

                if self.args.max_steps is not None and self.global_step >= self.args.max_steps:
                    reached_max_steps = True
                    break

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

            if reached_max_steps:
                break

        # 保存最终模型
        if self.global_step % self.args.save_steps != 0:
            self.save_checkpoint(self.global_step)

        # 关闭wandb
        if self.use_wandb:
            wandb.finish()

        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Independent noisy-maze SFT trainer")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model (if omitted, trains from scratch)")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--max_position_embeddings", type=int, default=512)

    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Micro batch size for gradient accumulation")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=3000, help="Stop after this many optimizer steps")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["cosine", "constant"], help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio for learning rate scheduler")

    # 评估和保存参数
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--use_generative_eval", action="store_true", help="Use generative evaluation (RL-style)")
    parser.add_argument("--eval_samples", type=int, default=128, help="Number of prompts for generative evaluation")
    parser.add_argument("--n_samples_per_prompt", type=int, default=256, help="Number of samples per prompt for Pass@k evaluation")
    parser.add_argument("--eval_generation_batch_size", type=int, default=32, help="Maximum generations in one forward batch")
    parser.add_argument("--eval_temperature", type=float, default=1.0, help="Temperature for sampling during evaluation")
    parser.add_argument("--eval_max_new_tokens", type=int, default=180, help="Maximum generated tokens per evaluation sample")

    # 日志参数
    parser.add_argument("--project_name", type=str, default="noisy_maze_maxrl_17x17", help="Project name for logging")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if args.eval_max_new_tokens <= 0:
        parser.error("--eval_max_new_tokens must be positive")
    if args.n_samples_per_prompt <= 0:
        parser.error("--n_samples_per_prompt must be positive")
    for name in ("batch_size", "micro_batch_size", "eval_generation_batch_size", "eval_samples", "eval_steps", "save_steps", "num_epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    # 创建训练器并开始训练
    trainer = MazeSFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
