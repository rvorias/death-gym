# The Death Gym competition, August 2026

A historical record. The competition is over; its machinery has been removed
from this repo. Nothing here is needed to train an agent — it is kept so the
result stays checkable and the disclosure below stays public.

## What it was

An open bounty, posted 2026-08-22 and closed 2026-08-29: train the best policy
to play Death Mountain, scored by mean XP at death. 100 USDC, winner take all.
Submissions were **data, never code** — safetensors weights plus a JSON config,
against a whitelisted architecture the evaluator owned. Nothing an entrant sent
was imported, unpickled, or executed.

**262 submissions, 247 scored, 36 distinct workers.**

## The result

Winner: `0xfc930b2D66d1d0E49e3b6D14df76acdd6aE7b05a`, **350.94 mean XP**.

| | |
|---|---|
| submission | `d8b2a39b-c8aa-4e5a-85d2-c1ae3639fd6d` |
| architecture | `dm_lstm_v1` |
| zip sha256 | `41c714d7cf7652ee2bf0559dd4eac6bc5ea072f95b7b5acd5e15f37065d72ad0` |
| per-seed | 348.1 / 350.0 / 354.8 |

That entrant submitted 100 of the 247 scored entries and held all 20 top
places. The next-best worker peaked at 320.74.

The public and private boards agreed closely: Spearman rank correlation
**0.9985**, median rank shift 1 place, mean private−public **+0.43 XP**
(sd 1.19). Same winner on both. No entry showed meaningful overfitting to the
published seeds.

## The seed commitment, revealed

Scoring used three private seeds, committed to before entries opened so they
could not be chosen after the fact. Published commitment:

```
02951b50635236518f1c9b61a6af9138a3fc0b4f92d3c6dcba9ac4c9090e3b5a
```

Revealed after close:

```
seeds  5000,1020591045,10227105712410311627
salt   8531a9021f7cc0f519cffa7b47d2f3304db9f967ef5b45a965f4ca0da998a262
```

Check it with `sha256("<seeds>|<salt>")` — the string is the comma-joined
seeds, a literal `|`, then the salt.

The full board, with every entry's score and hash, is archived at
[this gist](https://gist.github.com/rvorias/545d0b413e31b315a017157339adca9e).

## Disclosure: the luck bug

**The engine that scored every entry had a bug, and it is fixed in this repo
as of commit `e836b0b`.**

`special_stats[LUCK]` caches a value derived from equipped and bagged jewelry
greatness, but only the inventory kernel ever recomputed it. Two paths change
its inputs without going through that kernel: combat raises equipment XP — and
so greatness — on every hit, and buying an item writes it straight into the
equipment array. So a necklace or ring that levelled during a fight, or one
bought at market, **contributed nothing to critical-hit rate** until the
adventurer next equipped or dropped something.

It was known before the competition closed and deliberately left in place:
changing the rules mid-competition would have been worse than shipping them
imperfect. Every entry played the same engine, so no entrant was advantaged or
disadvantaged relative to another, and the ranking stands.

Measured effect, paired on 4000 worlds with one policy: −0.75 XP (σ_Δ 30.4,
z = −1.57, not significant), with 2.9% of worlds changing at all. But that
policy was *trained* against the buggy engine, so the number measures
adaptation, not the bug's true cost. A policy trained against the fixed engine
has never been compared.

**Consequence for reproduction:** the archived board's rows were produced by
the pre-fix evaluator and **cannot be reproduced against `e836b0b` or later**.
To reproduce a published row, check out `e057878` — the evaluator commit the
board itself names.
