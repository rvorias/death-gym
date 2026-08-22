# Death Gym

![Death Mountain](https://i.glifusercontent.com/unsafe/w3840/plain/s3://glif-assets-production/img/u45fdfm07387oaj35dm59.jpg)

This project more or less reimplements the [Death Mountain](https://github.com/Provable-Games/death-mountain)
adventure RPG as a vectorized RL environment. It also includes a PPO agent that
learns to play the game.

The game normally runs as a Cairo contract on StarkNet. I did not know how to get the original Cairo contracts running so I reimplemented everything from scratch in python. Then I had agents translate this to C (this repo). `engine/` contains the same game in ~9k lines of C.

It has three properties: the step loop does no allocation; one contiguous batch
holds all state; Python reads every buffer as a zero-copy numpy view. The engine
runs at approximately **0.8M environment steps/second** on a desktop with few
cores. This speed is sufficient for billions of PPO steps.

```
adventurer explores → meets a beast → fights, flees, or dies
                   → loots, levels up, buys gear at the market
                   → repeat until dead. score = XP at death.
```

## Quick start

Requires a C compiler, `make`, [`just`](https://github.com/casey/just), and
Python 3.10+. The engine is C11, needs nothing beyond libm, and builds without
`-Werror` or `-march=native` so a new compiler warning cannot fail your build.
If it does not compile, that is a bug worth reporting.

**If a `.venv/` is already present, do not create one over it.** Both `uv venv`
and `python -m venv` rewrite `pyvenv.cfg` while leaving the old `site-packages`,
and the interpreter then dies with `No module named 'encodings'` before it can
print a traceback. Install into it instead: `uv pip install -e '.[dev]'`.

From scratch:

```bash
uv venv && uv pip install -e '.[dev]'

just build     # compile the native engine
just test      # 114 tests
just bench     # env-steps/s on your machine
```

`torch>=2.4` resolves to the newest CUDA build, which may be newer than your
driver. The install still succeeds and then trains on CPU, ~40x slower, so check:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that prints `False` on a machine with an NVIDIA GPU, install a build matching
your driver. Take the CUDA version from `nvidia-smi` and use it as the tag —
12.6 is `cu126`, 12.8 is `cu128`:

```bash
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

The trainer says the same thing at startup if it lands on CPU.

Then take a short run end to end:

```bash
just train --total-steps 20000000 --run-name smoke   # ~7 min on one GPU
just eval local/checkpoints/smoke/final.safetensors
just tui                                             # live dashboard, second terminal
```

**`just train` with no arguments runs for 10 billion steps** — days. Always pass
`--total-steps` until you mean it. One iteration is `2048 envs x 32 steps` =
**65,536 env steps**, and `--total-steps` floors to whole iterations: 100000
buys one gradient update, and anything under 65,536 buys none. `final.safetensors`
is written whatever happens, so a short run still produces a checkpoint.

```
  seed 3930         avg    180.3   max  1253.0   median   138.0   truncated 0
  seed 7717         avg    181.6   max  1067.0   median   139.0   truncated 0
  seed 20477        avg    181.2   max  1218.0   median   139.0   truncated 0
eval_avg_xp:       181.0      the score: mean over the three banks
eval_banks:        3 x 16384 worlds
eval_seeds:        3930,7717,20477
eval_batch:        2048
eval_truncated:    0          must be 0
eval_gear_lvl15:   23.5%      reached one item at greatness 15+
eval_gear_lvl20:   10.2%      ...at greatness 20
eval_full_cloth:   0.0%       all five armour slots, one material
eval_full_hide:    0.1%
eval_full_metal:   4.5%
eval_final_stats:  str 2.9  dex 8.0  vit 5.5  int 2.7  wis 2.9  cha 6.4  luck 0.1
```

**Know the floor before you read a score.** Uniform random legal play scores
**~27**, and so does an untrained network — measured, not estimated. Anything in
the 25-30 band means the policy has learned nothing yet, however plausible the
number looks. A 20M-step run is still down there; the ~180 above took 305
iterations. `just eval` takes about 40s on a trained checkpoint, and only accepts
`.safetensors` — convert a legacy `.pt` with `python checkpoint.py <file>`.

The gear and stat lines are averages over the bank, read off each adventurer's
final observation at death. They say *how* a policy died, not just how far it
got: a run that scores well on XP but never reaches greatness 15 is winning by
volume, not by building a character.

Two evals of one checkpoint agree to the decimal **on the same GPU**. Across
GPUs they drift: one model scored 167.9 / 167.6 / 167.3 on an RTX 4000 Ada, an
A5000 and an A10G, because cuBLAS picks reduction orders per device. Every entry
is therefore scored on one pinned card, so expect your local number to sit a few
tenths off the board's. The trainer's own 1000-world bank at seed 7 is a
different measurement again, not comparable to either.

**Seeds 3930, 7717 and 20477 are the public group, not the scored ones.**
`reset()` is a pure function of its seed, so published scoring seeds would be a
test set anyone could train on. The final score is the mean over **three private
banks**, released with the results. Train for worlds you have not seen.

The private seeds are committed to before entries open, so they cannot be
changed once scoring has started:

```
02951b50635236518f1c9b61a6af9138a3fc0b4f92d3c6dcba9ac4c9090e3b5a
```

When the competition closes, the seeds and the salt are published with the final
board. Recompute `sha256("<seeds>|<salt>")`, compare it with the hash above, then
set `DM_COMPETITION_SEEDS` to those seeds and `just score-submission` reproduces
any published row, within the cross-GPU margin. If the digest does not match, the
ranking is void. You do not have to trust the scoring — you can check it.

[The leaderboard](https://gist.github.com/rvorias/545d0b413e31b315a017157339adca9e) updates as entries arrive, scored on the
**public** group, so `just score-submission` reproduces your own row. It names
the seeds and the evaluator commit and carries every entry's sha256 — check it
against your own `sha256sum submission.zip`. Only entries that passed appear: a
missing row means yours did not, and `just check-submission` says why.

Run `just` with no arguments to see every recipe.

## Layout

```
engine/src/     the game, in C — combat, loot, market, beasts, progression
dmfast/         Python bindings: BatchEnv (core) and VecEnv (SB3 adapter)
train.py        PPO trainer: masked categorical policy, LSTM memory, GAE
checkpoint.py   safetensors checkpoint I/O with the architecture recorded
rewards.py      reward presets and the env wrapper the trainer uses
tui.py          live multi-run dashboard (stdlib only)
tools/          pack, validate, score, and publish the leaderboard
tests/          engine, reward-model, checkpoint, and submission tests
docs/           environment reference
local/          NOT COMMITTED: checkpoints, logs, secrets
```

## The environment

```python
from dmfast import BatchEnv

env = BatchEnv(num_envs=4096, seed=42, max_steps=2048)
env.reset(seed=42)

env.step(actions)          # int32 (num_envs,) — updates buffers in place
env.obs                    # (num_envs, 463) float32
env.action_mask            # (num_envs, 57)  uint8 — illegal actions are 0
env.reward                 # (num_envs,)     float32
env.terminated             # (num_envs,)     uint8
```

Envs auto-reset on death, so the batch never stalls. The 463-dim observation
contains the game phase, adventurer stats, equipment, bag, market, current beast,
and a block of simulated-combat statistics. The 57 actions cover explore, attack,
flee, buy potion, buy item, equip, drop, and stat upgrades. **Always** filter with
`action_mask` — most actions are illegal in most phases.

[`docs/environment.md`](docs/environment.md) documents the phase machine, every
observation index range, and the per-phase masks.

There is also a Stable-Baselines3 adapter, if you would rather bring your own
trainer:

```python
from dmfast import VecEnv          # in '.[dev]'; standalone: uv pip install -e '.[sb3]'
from sb3_contrib import MaskablePPO

model = MaskablePPO("MultiInputPolicy", VecEnv(num_envs=64, seed=42))
model.learn(total_timesteps=10_000_000)
```

## Training

`train.py` is a single-file PPO implementation: a masked categorical policy over
a flat-item encoder, a shared MLP trunk, a gated-residual LSTM for in-episode
memory, GAE returns, a linear entropy schedule, and periodic checkpointing.
Defaults are 2048 parallel envs and 32-step rollouts.

```bash
just train --total-steps 20000000       # a real short run
just train --run-name my-exp            # → local/checkpoints/my-exp/
just train --resume local/checkpoints/my-exp/final.safetensors
```

Three architectures, all eligible for the competition: the default LSTM, `--mlp`
(the exact ablation, LSTM skipped), and `--transformer` (a stateless entity-token
transformer). `--embed-dim`, `--hidden-dim` and `--trunk-blocks` set the shape;
`--hidden-dim` alone still moves both widths together, `--embed-dim` decouples
them.

New checkpoints are **safetensors, not pickles**, and record the architecture that
produced them, so `--resume` rebuilds the policy at any shape without flags.
Legacy `.pt` files still load, with the architecture inferred from tensor shapes.

`just train` passes everything through to `train.py`; see `python train.py --help`
for the reward-shaping and curriculum flags. The trainer uses the first visible
CUDA device, or the CPU if there is none — select one with `CUDA_VISIBLE_DEVICES`.
`train.py` pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` first, so ids match `nvidia-smi`.

## Competition

A submission is **data, never code**: safetensors weights plus one JSON file,
at most 20 MiB of tensors, against a whitelisted architecture the evaluator
owns. Nothing a contestant sends is imported, unpickled, or executed.

```bash
just submit local/checkpoints/my-run/final.safetensors  # → submission.zip
just check-submission submission.zip                            # the evaluator's checks
just score-submission submission.zip                            # the real number
```

You write none of the zip by hand. The checkpoint records the architecture that
produced it, so `just submit` derives the config from the weights — the two
cannot disagree — and validates what it wrote. `just score-submission` scores
the zip itself, not the checkpoint it came from.

```text
submission.zip                    config.json: the architecture and shape,
├── model.safetensors             derived from the weights, plus the reward
└── config.json                   and hyperparameter record
```

Your identity comes from the submission platform, not the zip.

### What you choose

Pick a family; the name *is* the choice. Shapes come from an enumerated set,
never a range, so the largest model anyone can describe is known before the
evaluator allocates anything.

| `architecture` | trained with | shape keys, allowed values |
|---|---|---|
| `dm_mlp_v1` | `just train --mlp` | `embed_dim`, `hidden_dim` ∈ 128/256/512; `num_trunk_blocks` ∈ 1/2/3 |
| `dm_lstm_v1` | `just train` | same |
| `dm_transformer_v1` | `just train --transformer` | `d_model` ∈ 128/256; `n_layers` ∈ 2/4; `n_heads` ∈ 4/8; `ff_dim` ∈ 512/1024 |

`obs_dim` and `act_dim` are the environment's, fixed at 463 and 57.

**Do whatever you want with the trainer.** "Data, never code" describes the
submission, not how you produce it: fork `train.py`, throw it away, change
`NUM_ENVS`, rewrite the reward, skip PPO entirely, train in another framework.
Nobody looks at how the weights were made.

The only requirement is that they **run under the evaluator** — load into one
of the architectures above and play. `just check-submission` is that exact
test; run it before you send. The evaluator scores at its own batch width of
2048, so your training setup cannot affect your score. Reward weights and
hyperparameters ride along as a declared record it never acts on.

Bigger is not better: 512/512/3 is 4.8x the baseline's parameters and scores
167.6 against its 181.0.

`--equivariant-head`, `--quantile-value` and a non-`lstm` `MEMORY_TYPE` change
the tensor set and cannot be entered. The trainer prints `NOT SUBMITTABLE` at
startup rather than letting you find out after the GPU-days.

### What is enforced

The gate is `tools/validate_submission.py`, and `just check-submission` runs
exactly what the evaluator runs.

**The archive** — those two filenames and nothing else; no duplicates,
directories, symlinks, `..`, absolute paths or subdirectories; bounded entry
count, archive size, decompressed size and compression ratio. A `.pt` handed
over by mistake is told what it is, not reported as a member count.

**The config** — `format_version`, an architecture on the whitelist, exactly
that architecture's key set, every shape value from its enumerated list and
type-checked (`true` is not `1`), and a `training` block that is flat, bounded
and finite. JSON size, nesting depth and string length are capped.

**The tensors** — names, shapes and dtypes must match the module the config
names; the tensor region must tile the data segment exactly, so nothing can be
stapled after the last tensor; tied copies must agree; values must be finite;
payload ≤20 MiB.

Every check reads the safetensors **header** and stops there. The tensor bytes
are not touched until the file has proved itself, so a hostile submission cannot
make the evaluator allocate memory it did not agree to.

Safetensors carries a name, shape, dtype and raw bytes — no pickle, so loading
weights cannot reach `__reduce__() -> os.system(...)`, the classic `.pt` danger.
Scoring still belongs in a sandbox: no network, minimal filesystem, CPU/RAM/GPU
limits, a timeout.

## Upstream, license, and IP

The game — rules, content, semantics — is
[Provable-Games/death-mountain](https://github.com/Provable-Games/death-mountain),
© 2025 Provable Games, Inc. This repo is an independent reimplementation as an RL
environment; no Cairo source was copied, but the mechanics are theirs.

Upstream ships `licenses/BUSL_LICENSE` (Business Source License 1.1) and
`licenses/MIT_LICENSE`, and does not document which files fall under which. BUSL
1.1 allows copying and derivative works but restricts **production use** to the
Additional Use Grant at `v1-death-mountain-license-grants.provablegames.eth`,
until the Change Date (the earlier of 2028-01-01 or
`death-mountain-license-date.provablegames.eth`), when it becomes MIT.
