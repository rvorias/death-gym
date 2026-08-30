# Death Gym

![Death Mountain](https://i.glifusercontent.com/unsafe/w3840/plain/s3://glif-assets-production/img/u45fdfm07387oaj35dm59.jpg)

A fast, vectorized RL environment for [Death Mountain](https://github.com/Provable-Games/death-mountain)
— the adventure RPG behind Loot Survivor — plus a PPO baseline that learns to
play it. The question the repo exists to answer: **how well can an agent
autoplay this game?**

The game normally runs as a Cairo contract on StarkNet. I did not know how to
get the original Cairo contracts running, so I reimplemented everything from
scratch in Python, then had agents translate it to C. `engine/` contains the
same game in ~9k lines of C.

It has three properties: the step loop does no allocation; one contiguous batch
holds all state; Python reads every buffer as a zero-copy numpy view. The engine
runs at approximately **0.8M environment steps/second** on a desktop with few
cores — enough for billions of PPO steps.

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
just test      # the test suite
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

## Reading a score

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

**Three banks, not one.** `reset()` is a pure function of its seed, so a single
bank is one draw and a lucky seed is worth real points. The default seeds are
3930, 7717 and 20477; set `DM_EVAL_SEEDS` to score on worlds you have not tuned
against. A policy selected on the default seeds has been selected on a test set
it already saw.

**Numbers are comparable only at equal batch width and on the same GPU.** The
policy samples one action per env per step from a shared stream, so the same
bank scored at a different `num_envs` returns a different number — the full eval
therefore pins 2048. Across GPUs it drifts: one model scored 167.9 / 167.6 /
167.3 on an RTX 4000 Ada, an A5000 and an A10G, because cuBLAS picks reduction
orders per device.

> The numbers above were measured **before** the luck fix in `e836b0b` and have
> not been re-established since. Treat every baseline in this README as
> approximate until re-run.

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

Three architectures: the default LSTM, `--mlp` (the exact ablation, LSTM
skipped), and `--transformer` (a stateless entity-token transformer).
`--embed-dim`, `--hidden-dim` and `--trunk-blocks` set the shape; `--hidden-dim`
alone still moves both widths together, `--embed-dim` decouples them.

Checkpoints are **safetensors, not pickles**, and record the architecture that
produced them, so `--resume` rebuilds the policy at any shape without flags.
Legacy `.pt` files still load, with the architecture inferred from tensor shapes.

`just train` passes everything through to `train.py`; see `python train.py --help`
for the reward-shaping and curriculum flags. The trainer uses the first visible
CUDA device, or the CPU if there is none — select one with `CUDA_VISIBLE_DEVICES`.
`train.py` pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` first, so ids match `nvidia-smi`.

## Where the difficulty is

The game is not hard to play legally — the mask does that for you. It is hard to
play *well*, and the open problems are worth knowing before you start:

- **Long, sparse credit assignment.** The score arrives once, at death, hundreds
  of steps after the decisions that caused it. A stat point spent at level 2
  pays off at level 15, or does not.
- **The market is the real game.** Most of the decision space is buying and
  equipping, not fighting. Policies that learn to fight well and shop badly
  plateau early.
- **Death is cheap and information is expensive.** Fleeing preserves an
  adventurer who has learned nothing; fighting risks one who has.
- **Reward shaping moves the answer a lot.** The defaults in `rewards.py` are one
  set of choices, not the right ones — `--reward-override` exists so you can
  disagree with them.

## Layout

```
engine/src/     the game, in C — combat, loot, market, beasts, progression
dmfast/         Python bindings: BatchEnv (core) and VecEnv (SB3 adapter)
train.py        PPO trainer: masked categorical policy, LSTM memory, GAE
checkpoint.py   safetensors checkpoint I/O with the architecture recorded
rewards.py      reward presets and the env wrapper the trainer uses
tui.py          live multi-run dashboard (stdlib only)
tests/          engine, reward-model, checkpoint, and environment tests
docs/           environment reference, and the 2026 competition record
local/          NOT COMMITTED: checkpoints, logs
```

## History

This repo ran an open competition in August 2026, since concluded. The result,
the revealed scoring seeds, and the disclosure of an engine bug that affected
every entry are recorded in
[`docs/competition-2026.md`](docs/competition-2026.md). The competition
machinery has been removed; that document is the only trace kept.

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
