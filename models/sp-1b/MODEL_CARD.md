# Model card: sp-1b

MIT-licensed policy weights and inference source. Retained after the
September 6, 2026 training campaign; no claim of an optimal policy.

## Model and training history

The policy uses semantic88 observations, a shared item-action head, slot-pair
encoding, and a 32-wide recurrent LSTM. Its inherited pretraining completed
999,948,288 transitions. No new candidate weights were substituted in this release.
Full architecture and original metadata remain in `model.safetensors`.

The parent predates the native reward-layout repair. Historical training must
not be described as using the subsequently corrected intended reward. Critic
outputs estimate discounted training rewards, **not terminal XP**.

The follow-up campaign trained two seeds of a zero-initialized item-mechanics
extension and two matched controls, 49,938,432 transitions each: 199,753,728 in
total, plus a separate 1,048,576-transition execution smoke. These were
fine-tuning runs from one pretrained parent, not independent pretraining seeds.
Aggregate trainer-process time was 1.247 hours across an RTX A5000 and RTX 4000
SFF Ada; this includes compilation and embedded diagnostic evaluations.
Historical GPU time and electricity consumption were not measured here.

## Selection and independent confirmation

All final endpoints used the same 4,000-world selection bank, seed 20261201.
Selection ranked median XP, then mean XP. The incumbent scored mean 359.849,
median 206 on this bank.

| Endpoint | Mean XP | Median XP |
|---|---:|---:|
| item-mechanics50-mechanics-s31 | 368.051 | 209 |
| item-mechanics50-control-s31 | 358.923 | 207 |
| item-mechanics50-control-s32 | 368.063 | 209 |
| item-mechanics50-mechanics-s32 | 361.632 | 208 |

`item-mechanics50-control-s32` won selection. Its comparison with the incumbent on
the untouched 16,000-world bank, seed 20261202, was:

| Policy | Mean XP | Median XP |
|---|---:|---:|
| Selected candidate | 362.561 | 207 |
| Retained incumbent | 359.804 | 207 |

Candidate minus incumbent: mean **+2.757 XP**, paired 95% bootstrap interval
**[-3.022, 8.441]**; median **0 XP**, interval **[-3, 4]**. All evaluations had
zero truncations. The candidate failed the predeclared rule: median gain >=8,
mean gain >=0, median interval lower bound >0, mean interval lower bound >=-5,
and zero truncations. The new item inputs also showed opposite mean effects
across the two matched seeds on the selection bank; no repeatable benefit was
established by this experiment.

Evaluation used sampled masked actions, float32 inference on an RTX A5000,
native CPU game dynamics, batch 512, and a 2,048-action episode limit. Macro
actions 7–10 were excluded by the canonical mask. Each world contributed its
first episode. Intervals used 4,000 paired world-bootstrap replicates.

Per-world scores, terminal statistics, checkpoint identities, source hashes,
and evaluation manifests are in [benchmarks/](benchmarks/). The predeclared
protocol and training lineage are in [release_evidence.json](release_evidence.json)
and its linked evidence records. The model package contains no simulator;
reported game scores refer to the recorded simulator and evaluation protocol.

## Inference validation and limitations

`python policy_api.py --self-test` checks the bundled 384-observation fixture,
including recurrent resets, without the research repository or game engine.
For this incumbent, both raw-observation and exact-feature paths reproduced
research CPU logits, values, and recurrent states bit-for-bit in the recorded
environment. See [validation.json](validation.json). This fixture is an
execution-equivalence check, not a separate game-score evaluation.

Different numerical backends can change sampled trajectories. The inputs must
follow the documented observation schema, including combat-simulation fields,
and the caller must provide valid action masks and reset memory after episodes.
The retained model is the project's incumbent under its stated promotion rule;
null results do not prove that further improvement is impossible.
