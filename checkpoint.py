"""Checkpoint I/O on safetensors, with the architecture recorded in the file.

A `torch.save` checkpoint is a pickle: loading one executes arbitrary code, and
it carries no record of the architecture that produced it. Both facts caused
real problems — a submission could declare the wrong architecture and still
validate, because nothing could contradict it.

This module writes one `.safetensors` file instead:

    model.<name>            every model tensor
    optim.<index>.<key>     every optimizer state tensor
    __metadata__            json: arch, meta, and the optimizer's param_groups

`load()` also reads legacy `.pt` checkpoints so existing runs stay usable. It
infers their architecture from tensor shapes, because they do not record one.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# The architecture record. These fields fully determine the tensor set, so a
# reader can rebuild the exact module that produced the weights.
# The family name carries what used to be a use_lstm flag: dm_mlp_v1,
# dm_lstm_v1, dm_transformer_v1. Shape keys differ per family, so only the
# universal ones are required here.
ARCH_FIELDS = ("architecture", "obs_dim", "act_dim")
# Feature switches that also change the tensor set. They live in train.py as
# module globals, so without recording them a checkpoint is ambiguous.
FEATURE_FIELDS = (
    "memory_type", "decoupled_value", "quantile_value", "equivariant_head",
    "spr_aux",
)


def save(path, model_state: dict, arch: dict, meta: dict | None = None,
         optimizer_state: dict | None = None) -> Path:
    """Write a checkpoint. Returns the path actually written."""
    path = Path(path).with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [f for f in ARCH_FIELDS if f not in arch]
    if missing:
        raise ValueError(f"arch record is missing {missing}")

    # clone(): the encoder ties one embedding across three paths, and
    # safetensors refuses tensors that share storage.
    tensors = {f"model.{k}": v.detach().cpu().clone()
               for k, v in model_state.items() if torch.is_tensor(v)}

    param_groups = None
    if optimizer_state is not None:
        for idx, entries in optimizer_state["state"].items():
            for key, value in entries.items():
                if torch.is_tensor(value):
                    tensors[f"optim.{idx}.{key}"] = value.detach().cpu().clone()
        param_groups = optimizer_state["param_groups"]

    metadata = {
        "arch": json.dumps(arch),
        "meta": json.dumps(meta or {}),
        "param_groups": json.dumps(param_groups, default=list),
    }
    save_file(tensors, str(path), metadata=metadata)
    return path


def load(path, map_location="cpu") -> dict:
    """Read a checkpoint. Returns {model, optimizer, arch, meta, legacy}."""
    path = Path(path)
    if path.suffix == ".pt":
        return _load_legacy(path, map_location)

    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    metadata = header.get("__metadata__", {})

    tensors = load_file(str(path), device=str(map_location))
    model, optim_tensors = {}, {}
    for key, value in tensors.items():
        if key.startswith("model."):
            model[key[len("model."):]] = value
        elif key.startswith("optim."):
            _, idx, field = key.split(".", 2)
            optim_tensors.setdefault(int(idx), {})[field] = value

    optimizer = None
    param_groups = json.loads(metadata.get("param_groups", "null"))
    if param_groups is not None and optim_tensors:
        optimizer = {"state": optim_tensors, "param_groups": param_groups}

    return {
        "model": model,
        "optimizer": optimizer,
        "arch": json.loads(metadata.get("arch", "{}")),
        "meta": json.loads(metadata.get("meta", "{}")),
        "legacy": False,
    }


def _load_legacy(path: Path, map_location) -> dict:
    """Read a pickle checkpoint and reconstruct what it never recorded."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    model = blob.get("model_state_dict", blob)
    meta = {k: v for k, v in blob.items()
            if k not in ("model_state_dict", "optimizer_state_dict")}
    return {
        "model": model,
        "optimizer": blob.get("optimizer_state_dict"),
        "arch": infer_arch(model),
        "meta": meta,
        "legacy": True,
    }


def infer_arch(model_state: dict) -> dict:
    """Recover an architecture record from tensor shapes alone.

    Only for legacy checkpoints. `use_lstm` is NOT inferable — the module builds
    the LSTM cell whether or not the forward path uses it — so this reports the
    trainer's default and flags the field as a guess.
    """
    def shape(name):
        t = model_state.get(name)
        return tuple(t.shape) if t is not None else None

    proj = shape("input_trunk.proj.weight")
    if proj is None:
        raise ValueError(
            "cannot infer the architecture of this checkpoint: it has no "
            "'input_trunk.proj.weight'. Transformer (--transformer) checkpoints "
            "are not supported by this format."
        )
    hh = shape("lstm.weight_hh_l0")
    blocks = {k.split(".")[2] for k in model_state
              if k.startswith("input_trunk.blocks.")}
    head = shape("policy_head.weight")
    return {
        "architecture": "dm_lstm_v1",
        "obs_dim": 463,
        "act_dim": int(head[0]) if head else 57,
        "embed_dim": int(proj[1]),
        "hidden_dim": int(hh[1]) if hh else int(proj[0]),
        "num_trunk_blocks": len(blocks),
        # A pickle cannot say whether the LSTM path ran, so the family is a
        # guess the caller must resolve; see the docstring.
        "use_lstm_inferred": True,
        "memory_type": "lstm",
        "decoupled_value": "value_trunk.proj.weight" in model_state,
        "quantile_value": bool(head) and shape("value_head.weight")[0] > 1,
        "equivariant_head": "item_key.weight" in model_state,
        "spr_aux": any(k.startswith("spr_") for k in model_state),
    }


def peek_arch(path, map_location="cpu") -> dict:
    """The architecture record only, without materialising the optimizer."""
    return load(path, map_location)["arch"]


if __name__ == "__main__":
    # Convert a legacy pickle checkpoint so it can be evaluated:
    #     python checkpoint.py local/checkpoints/run/final.pt
    import sys

    for src in sys.argv[1:]:
        blob = load(src)
        out = save(Path(src).with_suffix(""), blob["model"], arch=blob["arch"],
                   meta=blob["meta"], optimizer_state=blob["optimizer"])
        print(f"{src} -> {out}")
