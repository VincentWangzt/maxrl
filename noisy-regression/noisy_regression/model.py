"""Scratch Qwen2 and the single digit-restricted autoregressive distribution."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen2Config
from transformers.cache_utils import DynamicCache

from noisy_regression.codec import (
    BOS,
    DIGITS,
    EOS,
    PAD,
    PROMPT_LENGTH,
    QUERY,
    QUERY_OFFSET,
    SEQUENCE_LENGTH,
    VOCAB,
    X,
    Y,
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = len(VOCAB)
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 512
    max_position_embeddings: int = 1024
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: None = None
    bos_token_id: int = BOS
    pad_token_id: int = PAD
    eos_token_id: int = EOS


def create_model(config):
    hf_config = Qwen2Config(**asdict(config), use_cache=True)
    # FP32 master weights/Adam states; CUDA forward uses BF16 autocast.
    return AutoModelForCausalLM.from_config(hf_config, attn_implementation="sdpa")


def answer_labels(tokens):
    if (
        tokens.ndim != 2
        or tokens.shape[1] != SEQUENCE_LENGTH
        or not torch.all(tokens[:, QUERY_OFFSET] == QUERY)
        or not torch.all(tokens[:, QUERY_OFFSET + 1] == X)
        or not torch.all(tokens[:, PROMPT_LENGTH - 1] == Y)
        or not torch.all(tokens[:, -1] == EOS)
    ):
        raise ValueError(
            f"Expected complete {SEQUENCE_LENGTH}-token examples ending in "
            "[QUERY] [X] a b [SEP] c d [Y] e f [EOS]"
        )
    if torch.any((tokens[:, PROMPT_LENGTH : PROMPT_LENGTH + 2] < 0) | (tokens[:, PROMPT_LENGTH : PROMPT_LENGTH + 2] >= DIGITS)):
        raise ValueError("Targets must be digit pairs")
    labels = torch.full_like(tokens, -100)
    labels[:, PROMPT_LENGTH:] = tokens[:, PROMPT_LENGTH:]
    return labels


def answer_nll_from_logits(logits, tokens):
    labels = answer_labels(tokens)
    # Do not pass labels into HF, which would shift internally and normalize
    # digit predictions over all vocabulary entries. The two digit shifts use
    # a constrained 16-way CE; termination uses the full vocabulary.
    selected = logits[:, PROMPT_LENGTH - 1 : PROMPT_LENGTH + 1, :DIGITS].float()
    digit_nll = F.cross_entropy(
        selected.reshape(-1, DIGITS),
        labels[:, PROMPT_LENGTH : PROMPT_LENGTH + 2].reshape(-1),
        reduction="none",
    ).reshape(-1, 2)
    eos_nll = F.cross_entropy(logits[:, PROMPT_LENGTH + 1].float(), labels[:, -1], reduction="none")
    return torch.cat((digit_nll, eos_nll[:, None]), dim=1)


def teacher_forced_nll(model, tokens):
    if tokens.shape[1] > model.config.max_position_embeddings:
        raise ValueError("Overlength input; truncation is forbidden")
    return answer_nll_from_logits(model(input_ids=tokens, use_cache=False).logits, tokens)


@torch.no_grad()
def conditional_log_probs(model, prompts):
    """Prefill once per prompt, then extend its KV cache with all 16 first digits.

    The same conditional table supports real sequential sampling and exact
    enumeration. No full prompt is repeated for the 256 completions.
    """
    if (
        prompts.ndim != 2
        or prompts.shape[1] != PROMPT_LENGTH
        or not torch.all(prompts[:, QUERY_OFFSET] == QUERY)
        or not torch.all(prompts[:, QUERY_OFFSET + 1] == X)
        or not torch.all(prompts[:, -1] == Y)
    ):
        raise ValueError(f"Expected {PROMPT_LENGTH}-token prompts ending in [QUERY] [X] a b [SEP] c d [Y]")
    if prompts.shape[1] + 2 > model.config.max_position_embeddings:
        raise ValueError("Overlength generation; truncation is forbidden")
    batch = len(prompts)
    prefill = model.model(input_ids=prompts, use_cache=True, past_key_values=DynamicCache())
    first = F.log_softmax(model.lm_head(prefill.last_hidden_state[:, -1])[:, :DIGITS].float(), dim=-1)
    cache = prefill.past_key_values
    cache.batch_repeat_interleave(DIGITS)
    digits = torch.arange(DIGITS, device=prompts.device).repeat(batch).view(-1, 1)
    continuation = model.model(input_ids=digits, past_key_values=cache, use_cache=True)
    second = F.log_softmax(model.lm_head(continuation.last_hidden_state[:, -1])[:, :DIGITS].float(), dim=-1)
    return first.double(), second.double().reshape(batch, DIGITS, DIGITS)


def joint_log_probs(first, second):
    joint = (first[:, :, None] + second).flatten(1)
    if not torch.isfinite(joint).all() or not torch.allclose(
        joint.exp().sum(-1), torch.ones(len(joint), dtype=joint.dtype, device=joint.device), atol=1e-6, rtol=0
    ):
        raise ValueError("Constrained answer distribution does not normalize")
    return joint


def sample_answers(first, second, count, generator):
    if count < 1:
        raise ValueError("Require a positive completion count")
    # Replacement gives independent temperature-1 samples with no truncation.
    a = torch.multinomial(first.exp(), count, replacement=True, generator=generator)
    b_probs = second.exp()[torch.arange(len(first), device=first.device)[:, None], a]
    b = torch.multinomial(b_probs.flatten(0, 1), 1, replacement=True, generator=generator).reshape_as(a)
    return torch.stack((a, b), dim=-1)


def make_optimizer(model, learning_rate, beta1, beta2, weight_decay, epsilon):
    decay, no_decay, decay_names, no_decay_names = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Matrices, including the tied embedding/output matrix, decay. Biases
        # and RMSNorm scales do not. named_parameters deduplicates tied weights.
        if parameter.ndim >= 2:
            decay.append(parameter)
            decay_names.append(name)
        else:
            no_decay.append(parameter)
            no_decay_names.append(name)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(
        groups, lr=learning_rate, betas=(beta1, beta2), eps=epsilon, foreach=False, fused=False
    )
    return optimizer, {
        "decay": decay_names,
        "no_decay": no_decay_names,
        "rule": "ndim >= 2 decays, including tied embeddings; biases and norm scales excluded",
    }


def make_scheduler(optimizer, warmup_steps, max_steps, min_learning_rate, schedule):
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("Require 0 <= warmup_steps < max_steps")
    base_learning_rates = {group["lr"] for group in optimizer.param_groups}
    if len(base_learning_rates) != 1:
        raise ValueError("Scheduler requires one shared base learning rate")
    (base_learning_rate,) = base_learning_rates
    if base_learning_rate <= 0 or not 0 <= min_learning_rate <= base_learning_rate:
        raise ValueError("Require base learning rate > 0 and 0 <= minimum learning rate <= base learning rate")
    if schedule not in ("linear_warmup_cosine_decay", "linear_warmup_constant"):
        raise ValueError(f"Unknown learning-rate schedule: {schedule}")
    minimum_ratio = min_learning_rate / base_learning_rate

    def multiplier(completed):
        update = completed + 1
        if warmup_steps and update <= warmup_steps:
            if schedule == "linear_warmup_constant":
                return minimum_ratio + (1.0 - minimum_ratio) * update / warmup_steps
            if warmup_steps == 1:
                return 1.0
            return minimum_ratio + (1.0 - minimum_ratio) * (update - 1) / (warmup_steps - 1)
        if schedule == "linear_warmup_constant":
            return 1.0
        if warmup_steps:
            decay_progress = (update - warmup_steps) / (max_steps - warmup_steps)
        else:
            decay_progress = (update - 1) / max(max_steps - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(decay_progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    # LambdaLR evaluates completed=0 at construction, so the optimizer's LR
    # already matches update 1 before the first optimizer.step().
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
