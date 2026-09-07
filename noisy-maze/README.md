# Noisy maze

Independent variant of the maze experiment. Maze construction and the SFT model
start from copies of the original implementation. This folder owns its generator,
tokenizer, trainer, reward function, data, and checkpoints; it does not import
`maze` or read its datasets or models. RL uses the repository's `verl` trainer.

## Observation and targets

The default maze size is **17×17** and the noise fraction is **0.1**. A connecting
position is an interior edge between two cell centers: exactly one coordinate is
even and the other is odd. Borders, odd/odd cell centers, even/even wall pillars,
START, and GOAL are never masked.

After constructing and solving the real maze, sample
`floor(noise_fraction × number_of_connecting_positions)` edges uniformly without
replacement, regardless of whether they contain WALL or PATH. Replace these tokens
with **UNKNOWN** in the observation. The mask stays fixed for that dataset row.
At 17×17 this masks 11 of 112 edges (actual fraction ≈0.098214); at 23×23 it masks
22 of 220. Metadata records both the requested and actual fraction and each row
records its masked coordinates. Topology and masking use separate seeded random
streams, so changing the noise fraction preserves the underlying maze corpus.

SFT JSON records contain `sequence` (clouded grid plus the real solution) and
`ground_truth` (unclouded grid plus the same solution). Only the clouded sequence
is encoded for training, with labels masked through PATH_START. Evaluation feeds
the clouded prompt to the model and validates only its generated action suffix
against the separate ground truth. RL Parquet stores the clouded prompt and the
real maze in `reward_model.ground_truth`.

The SFT and RL validator is shared within this variant. A solution must contain
valid direction tokens followed by DONE. Entering a hidden wall fails; UNKNOWN
is never treated as an open cell by the judge. Reaching GOAL terminates execution,
as in the original maze environment. Malformed or clouded ground truth raises an
error instead of silently granting a reward.

UNKNOWN is vocabulary ID 18, distinct from the tokenizer's `<unk>` token (ID 3).
It replaces one reserved vocabulary slot, keeping the total model vocabulary at
32 entries. The model is initialized from scratch; loading a checkpoint without
the noisy-maze token mapping fails explicitly.

## Datasets and training

The generator creates four disjoint splits, deduplicated by the **real maze**:

| Split | Rows |
| --- | ---: |
| SFT train | 100,000 |
| SFT evaluation | 128 |
| RL train | 2,048 |
| RL evaluation | 128 |

At the defaults, dataset directories are
`data/noisy_maze_17_noise_0.1_sft_100000` and
`data/noisy_maze_17_noise_0.1_rl_2048`. Both contain metadata and file hashes.
These are fresh splits within this experiment; no exclusion check is made against
the original experiment's corpus. Distinct real mazes can produce identical
clouded observations, especially at high noise, so perfect pass@1 is not generally
an attainable target.

SFT uses batch size **32**, microbatch size 8, **3,000 optimizer steps**, and
**AdamW at constant LR 5e-4**, betas `(0.9, 0.95)`, weight decay 0.01, and no
warmup. The copied Qwen2 architecture has hidden size 256, four layers, four
attention heads, two KV heads, and intermediate size 1,024.

Every **500 steps**, save a checkpoint, compute validation loss, and sample
**256 solutions per evaluation maze** at temperature 1.0. Generation batches
contain at most 32 solutions. Log unbiased pass@k and optimal_pass@k for
**k = 1, 2, 4, 8, 16, 32, 64, 128, 256**, plus failure rates, to W&B, the console,
and `metrics.jsonl`. The generation budget is 180 tokens for 17×17 and 256 for
23×23; each covers every simple solution in that maze size. Overlength SFT
sequences raise an error instead of truncating the solution.

RL starts from this variant's `ckpt-3000`, with 32 prompts per training step,
128 rollouts per prompt, LR 5e-5, 200 epochs (12,800 steps), and no KL penalty.
MaxRL, GRPO, and RLOO share the existing script style and optimizer settings.
Evaluation uses 128 held-out mazes with 256 samples each, before training and
every 64 steps; checkpoints are saved every 64 steps. `LR=1e-4` selects the
alternate learning rate and gives it a separate run name.

All SFT and RL launchers log to the W&B project `noisy_maze_maxrl_17x17`.
Run names include the maze size and noise fraction; RL names also include the
training set size, advantage estimator, rollout count, and learning rate. For example:
`noisy_maze_17_noise_0.1_sft_100000-constant-lr-5e-4-3000steps` and
`noisy_maze_17_noise_0.1_rl_2048-maxrl_128rollouts-lr_5e-5`.

The custom scorer uses `verl`'s batch reward manager. The existing prime manager
passes its callback through a spawn process pool, but the framework's custom
reward loader returns a local closure that cannot be pickled. Batch scoring avoids
that exception/fallback and shares exactly the SFT validator without modifying
the original experiment or shared framework.

## Server commands

Run these from `~/maxrl` on `cmu-L40-live`, after committing locally, pushing to
GitHub, and pulling on the server:

```bash
bash noisy-maze/prepare.sh
bash noisy-maze/sft.sh GPU_ID
bash noisy-maze/rl_maxrl.sh GPU_ID
bash noisy-maze/rl_grpo.sh GPU_ID
bash noisy-maze/rl_rloo.sh GPU_ID
```

Replace `GPU_ID` with an explicitly selected free physical GPU. Launchers check
that the GPU exists and is idle. SFT must finish before RL starts. They use the
repository `.venv` and load W&B credentials from `.env`, like the existing scripts.
Generated artifacts remain under `noisy-maze/`; dataset creation and SFT refuse
to overwrite existing output directories.

To change size or noise, pass the same settings to each stage:

```bash
MAZE_SIZE=23 NOISE_FRACTION=0.2 bash noisy-maze/prepare.sh
MAZE_SIZE=23 NOISE_FRACTION=0.2 bash noisy-maze/sft.sh GPU_ID
MAZE_SIZE=23 NOISE_FRACTION=0.2 bash noisy-maze/rl_maxrl.sh GPU_ID
```

Shell launchers accept sizes 17 and 23, and canonical noise fractions without
trailing zeros. The Python preparation module also supports other odd sizes and
custom split counts for focused validation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/noisy-maze:$PWD" \
  .venv/bin/python -m noisy_maze.prepare --help

CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/noisy-maze:$PWD" \
  .venv/bin/python -m pytest -q tests/test_noisy_maze.py
```

The focused CPU tests cover masking and reproducibility, split separation,
ground-truth rewards, JSON/Parquet inputs, vocabulary and labels, an optimizer
step, 256-sample evaluation with a known fake generator, and the custom RL reward
loader. GPU training is not part of this validation command.
