# The environment

This page describes the `BatchEnv` environment: the game loop, the 463-dim
observation, and the 57-dim action space. All layouts here come from
`engine/src/dmfast_exact.c` (`ex_pack_obs`) and `engine/src/dmfast_masks.c`.

```python
from dmfast import BatchEnv

env = BatchEnv(num_envs=4096, seed=42, max_steps=2048)
env.reset(seed=42)
env.step(actions)      # int32 (num_envs,)
```

`BatchEnv` holds every environment in one C batch. `env.obs`, `env.action_mask`,
`env.reward`, `env.terminated`, and `env.truncated` are zero-copy numpy views
into that batch. `step()` writes the buffers in place. Do not hold a reference
to a view across a step if you need the old values — copy the array first.

## The game loop

One adventurer explores a dungeon until it dies. Each explore action gives one
of three outcomes with equal probability:

1. **Beast** — the environment enters combat. The beast can ambush first.
2. **Obstacle** — the adventurer takes damage, or dodges it, and gains XP.
3. **Discovery** — the adventurer finds gold, health, or an item.

XP raises the adventurer level. A new level gives stat upgrade points and opens
a market. The episode ends when the adventurer dies. The score is the XP at
death.

### Phases

The environment is a state machine with six phases. `env.phase` holds the
current phase index per env. Phase index 0 is also the first element of the
observation.

| Index | Phase | The agent chooses |
|---|---|---|
| 0 | `UPGRADE` | one stat to raise, for each available upgrade point |
| 1 | `MARKET` | explore, buy an item, buy a potion, equip, or drop |
| 2 | `DROP` | which bag slot to discard |
| 3 | `EQUIP` | which bag item to put on |
| 4 | `COMBAT` | attack, flee, or swap gear |
| 5 | `BUY` | which market slot to purchase |

`DROP`, `EQUIP`, and `BUY` are selection phases. The agent enters them from
`MARKET` or `COMBAT`, picks one slot, and returns. This keeps the action space
flat: the agent never needs a compound "buy item *n*" action.

A gear swap during combat costs one free beast attack. The beast attacks once
when the agent next commits to attack or flee, and it attacks once no matter
how many slots the agent swapped. See `ex_pending_switch_retaliation`.

### Episode end

| Condition | Flag set |
|---|---|
| `health <= 0` | `terminated` |
| `episode_length >= max_steps` | `truncated` |
| level reached `max_level`, if the curriculum sets it | `truncated` |

`step()` auto-resets a finished env by default, so the batch never stalls. The
final observation and mask of the episode stay available in
`env.last_terminal_obs` and `env.last_terminal_action_mask`. Pass
`auto_reset=False` to stop this behaviour.

## Observation space

`env.obs` has shape `(num_envs, 463)` and dtype float32. It is one flat vector
of seven blocks:

| Index range | Block | Size |
|---|---|---|
| `0` | game phase | 1 |
| `1:14` | adventurer | 13 |
| `14:86` | equipment — 8 slots × 9 fields | 72 |
| `86:221` | bag — 15 slots × 9 fields | 135 |
| `221:446` | market — 25 slots × 9 fields | 225 |
| `446:453` | current beast | 7 |
| `453:463` | simulated-combat statistics | 10 |

Most fields are normalized to roughly `[0, 1]`. Categorical fields (item id,
item type, slot, specials, phase) are **raw integers cast to float**, because
the policy encoder embeds them rather than reading them as magnitudes.

### Adventurer block (`1:14`)

| Offset | Field | Scale |
|---|---|---|
| 1 | health | `/1023` |
| 2 | max health | `/1023` |
| 3 | XP | `/32767` |
| 4 | gold | `/511` |
| 5 | stat upgrades available | clamped to 15, then `/15` |
| 6 | beast health | `/1023` |
| 7-12 | strength, dexterity, vitality, intelligence, wisdom, charisma | `/31` |
| 13 | luck | `/100` |

Stats are the sum of the base stat and the stat bonus from item specials.

### Item vector (9 fields)

Equipment, bag, and market slots all use the same 9-field vector. An empty slot
is nine zeros.

| Offset | Field | Scale |
|---|---|---|
| 0 | item id | raw, 1-101 |
| 1 | tier | inverted to `(6 - tier) / 5`, so a higher value is a better item |
| 2 | type | raw: 1 magic/cloth, 2 blade/hide, 3 bludgeon/metal, 4 necklace, 5 ring |
| 3 | slot | raw: 1 weapon, 2 chest, 3 head, 4 waist, 5 foot, 6 hand, 7 neck, 8 ring |
| 4-6 | specials 1-3 | raw |
| 7 | item XP | `/400` |
| 8 | greatness | `(greatness - 1) / 20` |

Market items always have XP 0 and greatness 0. Their specials are computed from
the item specials seed at greatness 20, so the agent can see what a market item
*would* roll before it buys.

### Beast block (`446:453`)

All seven fields are zero when no beast is present.

| Offset | Field | Scale |
|---|---|---|
| 446 | starting health | `/1023` |
| 447 | tier | inverted, `(6 - tier) / 5` |
| 448 | level | `log1p(level) / log1p(640)` |
| 449 | type | raw, 1-3 |
| 450-452 | specials 1-3 | raw |

### Simulated-combat block (`453:463`)

The engine runs a full battle simulation against the current beast and writes
ten summary statistics. This is the block that lets the policy judge a fight
without learning the damage formula itself. All ten are zero when no beast is
present.

| Offset | Statistic |
|---|---|
| 453 | survivability of the adventurer |
| 454 | survivability of the beast |
| 455 | risk — the ratio the flee gate uses |
| 456 | average damage to the beast |
| 457 | average damage to the adventurer |
| 458-462 | average damage to chest, head, waist, foot, hand |

## Action space

`env.action_mask` has shape `(num_envs, 57)` and dtype uint8. A `0` marks an
illegal action. **Always filter with the mask.** Most actions are illegal in
most phases, and an unmasked policy spends its whole budget on invalid moves.

| Index range | Group |
|---|---|
| `0:11` | base actions |
| `11:26` | bag slots 0-14 |
| `26:51` | market slots 0-24 |
| `51:57` | stat upgrades |

The engine decodes an index by phase. In `EQUIP` and `DROP`, index 11 means bag
slot 0. In `BUY`, index 26 means market slot 0. In `UPGRADE`, index 51 means
strength. See `ex_decode_action`.

### Base actions (`0:11`)

| Index | Action | Effect |
|---|---|---|
| 0 | explore | draw one beast, obstacle, or discovery |
| 1 | attack | one combat exchange |
| 2 | flee | one flee attempt |
| 3 | equip | enter the `EQUIP` phase |
| 4 | drop | enter the `DROP` phase |
| 5 | buy item | enter the `BUY` phase |
| 6 | buy potion | heal 10 health for the current potion price |
| 7 | macro: fight | attack until the fight ends, up to 64 exchanges |
| 8 | macro: flee | flee until the fight ends, up to 64 attempts |
| 9 | macro: equip best | equip the bag item with the largest tier gain |
| 10 | macro: explore and soften | explore up to 24 times, then attack up to 32 times |

The four macros are single actions that run a loop inside the engine. They give
the agent a short-cut through the most repetitive sequences, so one policy
decision can cover a whole fight. Macro 9 returns the invalid-action penalty if
no bag item improves the current gear.

### Stat upgrades (`51:57`)

Index 51 to 56 raise strength, dexterity, vitality, intelligence, wisdom, and
charisma. A stat is legal only while it is below 31. **Luck has no upgrade
action** — it comes from item specials only.

### Which actions are legal in each phase

| Phase | Legal actions |
|---|---|
| `UPGRADE` | `51:57`, for each stat below 31 |
| `MARKET` | 0 explore, 10 macro-explore, plus 5 buy-item, 6 buy-potion, 3 equip, and 4 drop when their conditions hold |
| `DROP` | `11:26`, and only when the bag is full |
| `EQUIP` | `11:26`, for bag items whose slot is not already in the swap buffer |
| `COMBAT` | 1 attack, 2 flee, 7 macro-fight, 8 macro-flee, plus 3 equip when a bag item fits a free slot |
| `BUY` | `26:51`, for affordable items that improve the gear |

The buy mask is stricter than "can I afford it". `dmfast_set_buy_item_mask`
removes a market slot when any of these is true:

- The adventurer already owns that item id.
- The price is above the current gold.
- The item does not beat the best owned tier for that slot and type.
- The agent already bought an item for that slot this visit.

An optional flee gate can also clear actions 2 and 8. It fires when the
simulation says the fight is safe, the health ratio is high enough, and the
beast level is low enough. Set `mask_block_flee_on_safe` to enable it.

## Rewards

The C step path computes game logic only; it leaves the native reward buffer at
zero. `BatchEnv.step()` then derives `env.reward` from the difference between
the previous and current observation and info arrays, through
`dmfast_reward_compute_batch`. Reward shaping is therefore a configuration
choice, not an engine change — see `rewards.py` and `dmfast/reward.py`.

`env.info` has shape `(num_envs, 9)`: adventurer level, XP, health, gold, beast
level, beasts killed, items bought, potions bought, and a curriculum-start flag.
On the step where an episode ends, the same nine fields are copied into
`env.last_episode_info`.

## A minimal loop

```python
import numpy as np
from dmfast import BatchEnv

env = BatchEnv(num_envs=1024, seed=42, max_steps=2048)
env.reset(seed=42)
rng = np.random.default_rng(0)

for _ in range(1000):
    logits = policy(env.obs)                      # your policy
    logits = np.where(env.action_mask, logits, -np.inf)
    env.step(logits.argmax(axis=1).astype(np.int32))

env.close()
```

`env.sample_masked_actions(rng)` draws a uniform legal action per env. Use it
for benchmarks and smoke tests.
