# Death Mountain PPO policy: sp-1b

The retained project champion is now available as MIT-licensed weights and
standalone CPU inference code. On 16,000 fresh games it scored **359.80 mean XP**
and **207 median XP**, with zero truncations.

A subsequent 199.75M-transition, four-run training comparison found no candidate
that passed the predeclared promotion rule. This release therefore preserves
the incumbent. [Model card and results](MODEL_CARD.md).

[Hugging Face](https://huggingface.co/rvorias/death-mountain-sp-1b) is the preferred
distribution for the weights, inference code, and self-test. Download the
immutable snapshot behind release `v1.0`:

```sh
python -m pip install huggingface_hub
hf download rvorias/death-mountain-sp-1b \
  --revision 1c38b4d60add8852d0a456c29e7f08cab0b2eb55 \
  --local-dir death-mountain-sp-1b
cd death-mountain-sp-1b
python -m pip install -r requirements.txt
python policy_api.py --self-test
```

The commit pin keeps the download reproducible as the model repository changes.
See the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)
for download options.

The self-test uses the included real-observation fixture and expected outputs;
it needs no game installation. To play, supply observations and legal masks
from a compatible environment. [Observation/action contract](../../docs/environment.md).

## Inference API

No game engine, C compiler, dmfast, Triton, CUDA driver, or original source checkout is needed.
Install requirements.txt, then provide observations and legal-action masks from your environment.
When exported with --validate, run the included real-observation self-test without a simulator:

```sh
python policy_api.py --self-test
```

The test checks artifact hashes, exact-feature bitwise equality, and raw-input equality within
atol=1e-4/rtol=1e-5 against saved CPU outputs. Both tests cover logits, value, and recurrent state.
Bitwise reproduction targets the recorded package versions and CPU numerical behavior;
different PyTorch versions or CPU backends may require investigating numerical differences.
The numeric interface is 463 raw float32 observation fields and 57 bool action-mask fields.
Observations must include the existing combat-simulation fields; this model does not compute game dynamics.

```python
import torch
from policy_api import DeathMountainPolicy

policy = DeathMountainPolicy()
state = policy.initial_state(batch_size)
# raw_obs: (batch_size, 463); legal_mask: (batch_size, 57)
logits, value, state = policy.step(raw_obs, legal_mask, state)
actions = torch.distributions.Categorical(logits=logits).sample()
# After your environment advances, reset memory for ended episodes.
state = policy.reset_state(state, done)
```

The default excludes macro actions 7–10 with the canonical nonempty-mask fallback.
Pass primitive_actions=False to use exactly your supplied legal mask.
Use logits.argmax(-1) for greedy actions; PPO score reports ordinarily use sampling.
A NumPy reference derives semantic features from raw observations. Its floating-point rounding
can differ slightly from native C features. An optional semantic_features argument accepts
the exact 88/90 features from an external evaluator for exact checkpoint equivalence.
The raw-input validation report records actual numerical differences; it does not claim
bit-identical sampled trajectories or certify benchmark scores across implementations.

Weights are safetensors without optimizer state; full architecture and original checkpoint
metadata are retained. See manifest.json for hashes and LICENSE for MIT terms.
