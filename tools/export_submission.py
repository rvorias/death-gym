#!/usr/bin/env python3
"""Pack a training checkpoint into a competition submission zip.

    python tools/export_submission.py local/checkpoints/my-run/final.safetensors

The checkpoint records the architecture that produced it, so nothing here is
guessed or asked of you: the config is read off the file. The submission is
checked against the competition rules before it is written, so a zip this tool
produces is one the evaluator accepts.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import checkpoint  # noqa: E402

# The config a submission carries is exactly the whitelist's key set for that
# architecture, taken from the checkpoint's own record. Nothing is declared by
# hand, so nothing can be declared wrongly.

# torch.compile and DDP both prefix every parameter name.
PREFIXES = ("_orig_mod.", "module.")


def strip_prefix(name: str) -> str:
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def die(message: str) -> int:
    print(f"cannot pack this checkpoint: {message}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path,
                    help="a .safetensors checkpoint (or a legacy .pt)")
    ap.add_argument("-o", "--out", type=Path, default=Path("submission.zip"),
                    help="output path (default: submission.zip)")
    ap.add_argument("--mlp", action="store_true",
                    help="LEGACY .pt ONLY: the run used train.py --mlp. A "
                         ".safetensors checkpoint records this itself.")
    args = ap.parse_args()

    if args.checkpoint.is_dir():
        found = sorted(args.checkpoint.glob("*.safetensors")) or \
                sorted(args.checkpoint.glob("*.pt"))
        hint = f" Did you mean {found[-1]}?" if found else ""
        return die(f"{args.checkpoint} is a directory, not a checkpoint file.{hint}")
    if not args.checkpoint.exists():
        return die(f"{args.checkpoint} does not exist")

    try:
        ckpt = checkpoint.load(args.checkpoint, map_location="cpu")
    except ValueError as exc:
        return die(str(exc))

    arch = dict(ckpt["arch"])
    if ckpt["legacy"]:
        print(f"note: {args.checkpoint} is a legacy pickle checkpoint. Its "
              f"architecture was inferred from tensor shapes.")
        if arch.get("use_lstm_inferred"):
            arch["architecture"] = "dm_mlp_v1" if args.mlp else "dm_lstm_v1"
            print(f"note: a .pt cannot record whether the LSTM path was used, so "
                  f"the family is taken from "
                  f"{'--mlp' if args.mlp else 'the trainer default'}: "
                  f"{arch['architecture']}. Re-save to record it.")
    elif args.mlp:
        return die("--mlp is only for legacy .pt checkpoints; this file records "
                   "use_lstm itself")

    weights = {strip_prefix(k): v.detach().to("cpu", torch.float32).clone()
               for k, v in ckpt["model"].items() if torch.is_tensor(v)}
    if not weights:
        return die("no tensors found in the checkpoint")

    if arch["architecture"] == "dm_mlp_v1":
        dropped = [k for k in weights if k.startswith("lstm")]
        for k in dropped:
            del weights[k]
        print(f"stateless entry: dropped {len(dropped)} LSTM tensors the "
              f"evaluator will not read")

    import tools.validate_submission as validator
    schema = validator.ARCHITECTURES.get(arch["architecture"])
    if schema is None:
        return die(f"architecture {arch['architecture']!r} is not eligible; "
                   f"the competition accepts {sorted(validator.ARCHITECTURES)}")
    fields = ["architecture", *schema["fixed"], *schema["choices"]]
    missing = [f for f in fields if f not in arch]
    if missing:
        return die(f"the checkpoint's architecture record is missing {missing}")
    # The entrant's identity comes from the submission platform, not the zip:
    # a self-declared name is unverifiable and duplicates what Taskmarket holds.
    config = {k: arch[k] for k in fields}
    config["format_version"] = 3
    config["training"] = ckpt["meta"].get("training", {"reward": {}, "params": {}})

    staging = args.out.parent / f".{args.out.stem}-staging"
    staging.mkdir(parents=True, exist_ok=True)
    st_path = staging / "model.safetensors"
    save_file(weights, str(st_path))
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("model.safetensors", "config.json"):
            z.write(staging / name, arcname=name)
    for name in ("model.safetensors", "config.json"):
        (staging / name).unlink()
    staging.rmdir()

    # Check what we just wrote against the rules the evaluator applies, so a
    # rejection surfaces here rather than after the entry is submitted.
    import tools.validate_submission as validator
    try:
        validator.validate(args.out, verbose=False)
    except validator.Invalid as exc:
        args.out.unlink(missing_ok=True)
        return die(f"the packed submission would be rejected: {exc}")

    n_params = sum(t.numel() for t in weights.values())
    payload = sum(t.numel() * t.element_size() for t in weights.values())
    print(f"{args.out}  {args.out.stat().st_size/1e6:.2f} MB zipped  "
          f"{payload/1024/1024:.2f} MiB payload  {n_params:,} params in "
          f"{len(weights)} tensors")
    shape = "  ".join(f"{k}={config[k]}" for k in schema["choices"])
    print(f"  architecture  {config['architecture']}  {shape}")
    print("  validated OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
