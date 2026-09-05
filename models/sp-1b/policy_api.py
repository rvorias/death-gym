"""Engine-independent CPU inference from observations, legal masks and memory."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import policy_core
import policy_loader


class DeathMountainPolicy:
    def __init__(self, checkpoint=ROOT / "model.safetensors"):
        os.environ["DM_SEM_V1"] = "0"
        policy_core.BF16_ENC = False
        policy_core.FUSED_ITF = False
        policy_core.ITEM_TF_COMPILE = False
        self.model, self.arch = policy_loader.load(str(checkpoint), torch.device("cpu"))
        self.extra_dim = self.arch["obs_extra_dim"]
        # Each instance owns its semantic module so loading semantic88 and
        # semantic90 policies in one process cannot change existing instances.
        flags = {"DM_SEM_V1": "0", "DM_SEM_V3": "1", "DM_SEM_V4": "1"}
        previous = {name: os.environ.get(name) for name in flags}
        try:
            os.environ.update(flags)
            spec = importlib.util.spec_from_file_location("_policy_semantic_reference", ROOT / "semantic_obs.py")
            self.semantics = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.semantics)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def initial_state(self, batch_size):
        return self.model.init_state(batch_size, torch.device("cpu"))

    @staticmethod
    def reset_state(state, done):
        keep = (~torch.as_tensor(done, dtype=torch.bool)).to(state[0].dtype).view(1, -1, 1)
        return tuple(part * keep for part in state)

    @torch.inference_mode()
    def step(self, raw_obs, legal_mask, state=None, *, primitive_actions=True, semantic_features=None):
        """Return (masked_logits, value, next_state), with no simulator calls.

        raw_obs: float32 CPU tensor/array (N,463); legal_mask: bool (N,57).
        The caller supplies observations/masks and resets state on episode end.
        Optional semantic_features (N,88/90) reproduces an external evaluator's
        exact feature values; by default they are calculated from raw_obs here.
        """
        raw = torch.as_tensor(raw_obs, dtype=torch.float32, device="cpu")
        mask = torch.as_tensor(legal_mask, dtype=torch.bool, device="cpu").clone()
        if raw.ndim != 2 or raw.shape[1] != 463 or mask.shape != (len(raw), 57):
            raise ValueError("expected raw_obs (N,463) and legal_mask (N,57)")
        if not torch.isfinite(raw).all() or not mask.any(-1).all():
            raise ValueError("observations must be finite and every row needs a legal action")
        if primitive_actions:
            # Same sequential, nonempty-mask fallback as train.mask_macros.
            for action in (7, 8, 9, 10):
                candidate = mask.clone()
                candidate[:, action] = False
                mask[candidate.any(-1), action] = False
        if semantic_features is None:
            extra = self.semantics.semantic_features(raw.numpy())[:, :self.extra_dim]
            extra = torch.from_numpy(extra.copy())
        else:
            extra = torch.as_tensor(semantic_features, dtype=torch.float32, device="cpu")
        if extra.shape != (len(raw), self.extra_dim) or not torch.isfinite(extra).all():
            raise ValueError("invalid semantic feature shape or values")
        obs = torch.cat((raw, extra), dim=-1)
        if state is None:
            state = self.initial_state(len(raw))
        return self.model.step(obs, mask, state)


def run_fixture(policy, path, *, exact_features=False):
    with np.load(path, allow_pickle=False) as f:
        raw, masks, done = f["raw_obs"], f["masks"], f["done"]
        extra = f["semantic_features"] if exact_features else None
    state = policy.initial_state(raw.shape[1])
    outputs = {key: [] for key in ("logits", "values", "h", "c")}
    for index, (obs, mask, ended) in enumerate(zip(raw, masks, done)):
        logits, value, state = policy.step(obs, mask, state,
            semantic_features=extra[index] if extra is not None else None)
        for key, tensor in zip(outputs, (logits, value, *state)):
            outputs[key].append(tensor.numpy().copy())
        state = policy.reset_state(state, ended)
    return {key: np.stack(value) for key, value in outputs.items()}


def self_test(policy):
    """Check packaged numerical observations against saved research CPU outputs."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected_hash in manifest["files_sha256"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"packaged file missing or changed: {name}")
    fixture = ROOT / "validation-fixture.npz"
    with np.load(ROOT / "validation-expected.npz", allow_pickle=False) as archive:
        expected = {key: archive[key] for key in ("logits", "values", "h", "c")}
    report = {"status": "passed", "device": "cpu", "comparisons": {}}
    for exact in (False, True):
        actual = run_fixture(policy, fixture, exact_features=exact)
        errors = {}
        for key, reference in expected.items():
            if exact:
                np.testing.assert_array_equal(actual[key], reference, err_msg=key)
            else:
                np.testing.assert_allclose(actual[key], reference, atol=1e-4, rtol=1e-5, err_msg=key)
            errors[key] = float(np.max(np.abs(actual[key] - reference)))
        errors["greedy_action_disagreements"] = int(np.count_nonzero(
            actual["logits"].argmax(-1) != expected["logits"].argmax(-1)))
        report["comparisons"]["exact_features" if exact else "raw_observations"] = errors
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path,
                        help="NPZ containing raw_obs[T,N,463], masks[T,N,57], done[T,N]")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exact-features", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="verify included observations and expected CPU outputs")
    args = parser.parse_args()
    if args.self_test:
        if args.fixture or args.output or args.exact_features:
            parser.error("--self-test is used without fixture/output/exact-features options")
    elif args.fixture is None or args.output is None:
        parser.error("provide --self-test or both --fixture and --output")
    torch.set_num_threads(1)
    policy = DeathMountainPolicy()
    if args.self_test:
        print(json.dumps(self_test(policy), indent=2))
    else:
        np.savez_compressed(args.output, **run_fixture(policy, args.fixture, exact_features=args.exact_features))


if __name__ == "__main__":
    main()
