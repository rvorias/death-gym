#!/usr/bin/env python3
"""Validate a competition submission zip, then load it into the reference model.

    python tools/validate_submission.py submission.zip

The evaluator owns the architecture. A submission carries data only: weights in
safetensors, plus two JSON files. Nothing here imports, unpickles, or executes
anything a contestant supplied.

Every check below runs BEFORE the tensor bytes are read, so a hostile file
cannot make this process allocate memory it did not agree to allocate.

Exit code 0 means the submission is valid.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path

MAX_WEIGHT_BYTES = 20 * 1024 * 1024   # 20 MiB of tensor payload
MAX_JSON_BYTES = 64 * 1024
MAX_JSON_DEPTH = 8
MAX_JSON_STRING = 256
MAX_ZIP_ENTRIES = 8
MAX_UNCOMPRESSED = 64 * 1024 * 1024   # zip-bomb guard, checked from the header
MAX_COMPRESSION_RATIO = 200
MAX_ZIP_BYTES = 64 * 1024 * 1024      # the archive itself, before it is opened

# Every size in a zip's central directory is attacker-controlled, so the header
# checks are only a cheap pre-filter. These are the limits actually enforced,
# by reading each member through a bounded stream.
MEMBER_LIMITS = {
    "model.safetensors": MAX_WEIGHT_BYTES + MAX_JSON_BYTES,   # payload + header
    "config.json": MAX_JSON_BYTES,
}

ALLOWED_NAMES = {"model.safetensors", "config.json"}
ALLOWED_DTYPES = {"F32"}              # safetensors dtype tags
MAX_TENSOR_ELEMENTS = 8_000_000
MAX_TENSOR_RANK = 4

# The architecture whitelist. Add an entry here to allow a new architecture;
# there is no way for a submission to introduce one.
#
# "fixed" keys are decided by the environment, not the contestant, and must
# match exactly. "choices" keys are the contestant's to pick, but only from an
# ENUMERATED set — never a range. The set is finite and small, so the largest
# model any submission can describe is known ahead of time and the evaluator
# can instantiate the reference module without risking a resource attack.
#
# Worst case for dm_lstm_v1 (512/512/3) is 4,294,104 params = 16.38 MiB, which
# is inside MAX_WEIGHT_BYTES. Widen these lists and that stops being true:
# recompute before you do.
ARCHITECTURES = {
    # The three families a contestant may enter. The family is the name -- there
    # is no separate flag to interpret, and the name alone decides the tensor
    # set. "fixed" keys belong to the environment; "choices" are the
    # contestant's, from an ENUMERATED set, never a range, so the largest model
    # any submission can describe is known before anything is allocated.
    #
    # Worst cases, all inside MAX_WEIGHT_BYTES:
    #   dm_mlp_v1          512/512/3   2,192,855 params   8.36 MiB
    #   dm_lstm_v1         512/512/3   4,294,104 params  16.38 MiB
    #   dm_transformer_v1  256/4/8/1024 3,718,164 params 14.18 MiB
    # A test asserts this, so widening a list fails the suite rather than
    # quietly admitting a model the size gate would then reject.
    "dm_mlp_v1": {
        "fixed": {"obs_dim": 463, "act_dim": 57},
        "choices": {
            "embed_dim": [128, 256, 512],
            "hidden_dim": [128, 256, 512],
            "num_trunk_blocks": [1, 2, 3],
        },
    },
    "dm_lstm_v1": {
        "fixed": {"obs_dim": 463, "act_dim": 57},
        "choices": {
            "embed_dim": [128, 256, 512],
            "hidden_dim": [128, 256, 512],
            "num_trunk_blocks": [1, 2, 3],
        },
    },
    "dm_transformer_v1": {
        "fixed": {"obs_dim": 463, "act_dim": 57},
        "choices": {
            "d_model": [128, 256],
            "n_layers": [2, 4],
            "n_heads": [4, 8],
            "ff_dim": [512, 1024],
        },
    },
}

# Reward weights and training hyperparameters travel with the entry. The
# evaluator does not act on them -- a submission is scored purely on how the
# policy plays -- but an entry that cannot say how it was produced is not
# reproducible. They are validated for shape and bounds only, so a declaration
# can never become a resource attack.
TRAINING_SECTIONS = ("reward", "params")
MAX_TRAINING_KEYS = 128
TRAINING_KEY_RE = r"[a-z][a-z0-9_]{0,63}"
MAX_TRAINING_MAGNITUDE = 1e12


def check_choice(key, value, allowed):
    """Type-strict membership. `True == 1` in Python, so a bare `in` test would
    accept 1 where a bool is required and True where an int is required."""
    want_bool = isinstance(allowed[0], bool)
    got_bool = isinstance(value, bool)
    if want_bool != got_bool or not isinstance(value, (bool, int)):
        raise Invalid(f"config[{key!r}] has the wrong type: {value!r}")
    if value not in allowed:
        raise Invalid(f"config[{key!r}] is {value!r}, must be one of {allowed}")


def check_training(training) -> None:
    """Bounded, flat, declared-only. Never used to build anything."""
    import math
    import re

    if not isinstance(training, dict):
        raise Invalid("config.training must be an object")
    unknown = set(training) - set(TRAINING_SECTIONS)
    if unknown:
        raise Invalid(f"config.training has unknown sections: {sorted(unknown)}; "
                      f"allowed: {list(TRAINING_SECTIONS)}")
    for section, entries in training.items():
        if not isinstance(entries, dict):
            raise Invalid(f"config.training.{section} must be an object")
        if len(entries) > MAX_TRAINING_KEYS:
            raise Invalid(f"config.training.{section} has {len(entries)} keys, "
                          f"limit is {MAX_TRAINING_KEYS}")
        for key, value in entries.items():
            if not re.fullmatch(TRAINING_KEY_RE, key):
                raise Invalid(f"config.training.{section} key {key!r} is not a "
                              f"lowercase identifier")
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise Invalid(f"config.training.{section}[{key!r}] is not finite")
                if abs(value) > MAX_TRAINING_MAGNITUDE:
                    raise Invalid(f"config.training.{section}[{key!r}] is out of range")
                continue
            if isinstance(value, str) and len(value) <= 64:
                continue
            raise Invalid(f"config.training.{section}[{key!r}] must be a number, "
                          f"bool, or short string")


class Invalid(Exception):
    pass


def json_depth_ok(node, depth=0):
    if depth > MAX_JSON_DEPTH:
        raise Invalid(f"json nested deeper than {MAX_JSON_DEPTH}")
    if isinstance(node, dict):
        for k, v in node.items():
            if len(k) > MAX_JSON_STRING:
                raise Invalid(f"json key longer than {MAX_JSON_STRING} chars")
            json_depth_ok(v, depth + 1)
    elif isinstance(node, list):
        for v in node:
            json_depth_ok(v, depth + 1)
    elif isinstance(node, str) and len(node) > MAX_JSON_STRING:
        raise Invalid(f"json string longer than {MAX_JSON_STRING} chars")


def read_zip(path: Path) -> dict[str, bytes]:
    """Return the zip's members as bytes, rejecting anything unsafe first.

    Checks run against the central directory, so a zip bomb is refused before
    a single byte is decompressed.
    """
    path = Path(path)   # callers pass strings; the size check needs a Path
    size = path.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise Invalid(f"the archive is {size/1024/1024:.1f} MiB, limit is "
                      f"{MAX_ZIP_BYTES/1024/1024:.0f} MiB")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise Invalid(not_a_zip_reason(path)) from None
    with archive as z:
        infos = z.infolist()
        # A .pt IS a zip, so without this the next check reports its internal
        # member count and reads as "your archive has too many files".
        if any(i.filename.endswith((".pkl", "data.pkl")) for i in infos):
            raise Invalid("this is a PyTorch checkpoint, not a submission. Pack it "
                          "first:  python tools/export_submission.py <ckpt>")
        if len(infos) > MAX_ZIP_ENTRIES:
            raise Invalid(f"{len(infos)} entries, limit is {MAX_ZIP_ENTRIES}")

        seen = set()
        total = 0
        for info in infos:
            name = info.filename
            # A dict keyed by filename would silently keep the LAST member and
            # discard the first, so two different configs could ride in
            # one archive and inspection tools would show the shadowed one.
            if name in seen:
                raise Invalid(f"duplicate archive member: {name!r}")
            seen.add(name)
            if info.is_dir():
                raise Invalid(f"directory entry: {name!r}")
            # A unix zip stores st_mode in the high 16 bits of external_attr.
            # Only the file-type field decides: a symlink is S_IFLNK. Entries
            # written by zipfile.writestr leave the type field zero, which is
            # normal and must stay allowed.
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type and file_type != 0o100000:
                raise Invalid(f"not a regular file: {name!r}")
            if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
                raise Invalid(f"absolute path: {name!r}")
            if ".." in Path(name).parts or "/" in name or "\\" in name:
                raise Invalid(f"path traversal or subdirectory: {name!r}")
            if name not in ALLOWED_NAMES:
                raise Invalid(f"unexpected file: {name!r}; a submission contains "
                              f"exactly {sorted(ALLOWED_NAMES)}")
            if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                raise Invalid(f"compression ratio above {MAX_COMPRESSION_RATIO}x: {name!r}")
            total += info.file_size
        if total > MAX_UNCOMPRESSED:
            raise Invalid(f"uncompressed size {total} above {MAX_UNCOMPRESSED}")

        names = {i.filename for i in infos}
        missing = ALLOWED_NAMES - names
        if missing:
            raise Invalid(f"missing: {sorted(missing)}")
        return {i.filename: read_member(z, i) for i in infos}


def read_member(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Decompress one member without trusting what the archive says it weighs.

    The central directory's file_size is attacker-controlled: understate it and
    the ratio and total-size gates above both pass, then a plain z.read()
    happily inflates the real payload. A 399 KiB archive claiming 1 KiB expanded
    to 400 MiB before zipfile noticed the CRC was wrong -- an out-of-memory kill
    with the diagnostics arriving too late to matter.

    So read at most limit+1 bytes and refuse anything that keeps coming.
    """
    limit = MEMBER_LIMITS[info.filename]
    try:
        with z.open(info) as fh:
            data = fh.read(limit + 1)
    except zipfile.BadZipFile as exc:
        raise Invalid(f"{info.filename!r} is corrupt: {exc}") from None
    if len(data) > limit:
        raise Invalid(f"{info.filename!r} decompresses to more than {limit} bytes, "
                      f"whatever its header claims")
    return data


def not_a_zip_reason(path) -> str:
    """Say what the file actually is, rather than only what it is not."""
    path = Path(path)
    head = path.read_bytes()[:16] if path.exists() else b""
    if len(head) >= 9 and head[8:9] == b"{":
        return ("this is a safetensors weights file, not a submission zip. A "
                "submission is a zip of model.safetensors + config.json + "
                "pack it with tools/export_submission.py")
    if head[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return ("this is a Python pickle, not a submission zip. Pack it with "
                "tools/export_submission.py")
    return "not a zip file"


def load_json(raw: bytes, label: str) -> dict:
    if len(raw) > MAX_JSON_BYTES:
        raise Invalid(f"{label} is {len(raw)} bytes, limit is {MAX_JSON_BYTES}")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Invalid(f"{label} is not valid json: {exc}") from exc
    if not isinstance(doc, dict):
        raise Invalid(f"{label} must be a json object")
    json_depth_ok(doc)
    return doc


def parse_safetensors_header(blob: bytes) -> tuple[dict, int]:
    """Read the safetensors header without touching the tensor payload.

    Layout: 8 bytes little-endian header length, then that many bytes of JSON,
    then the raw tensor bytes. The JSON gives every tensor's dtype, shape, and
    byte range, which is everything needed to reject a file before allocating.
    """
    if len(blob) < 8:
        raise Invalid("model.safetensors is truncated")
    if blob[:2] in (b"PK", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        raise Invalid("model.safetensors is not a safetensors file -- it looks like a "
                      "zip or a pickle. Renaming a .pt does not convert it; run "
                      "python tools/export_submission.py <ckpt>")
    (header_len,) = struct.unpack("<Q", blob[:8])
    if header_len > MAX_JSON_BYTES or 8 + header_len > len(blob):
        raise Invalid(f"safetensors header length {header_len} is out of range")
    header = load_json(blob[8:8 + header_len], "safetensors header")
    return header, 8 + header_len


def validate(path: Path, verbose: bool = True) -> dict:
    members = read_zip(path)

    config = load_json(members["config.json"], "config.json")

    arch = config.get("architecture")
    if not isinstance(arch, str) or arch not in ARCHITECTURES:
        raise Invalid(f"architecture {arch!r} is not whitelisted; "
                      f"allowed: {sorted(ARCHITECTURES)}")
    schema = ARCHITECTURES[arch]
    fixed, choices = schema["fixed"], schema["choices"]

    allowed_keys = (set(fixed) | set(choices)
                    | {"architecture", "format_version", "training"})
    extra = set(config) - allowed_keys
    if extra:
        raise Invalid(f"unknown config keys: {sorted(extra)}")
    missing = allowed_keys - set(config)
    if missing:
        raise Invalid(f"missing config keys: {sorted(missing)}")

    # Fixed first: these describe the environment, and nothing downstream is
    # meaningful if they are wrong.
    for key, want in fixed.items():
        if config[key] != want or isinstance(config[key], bool) != isinstance(want, bool):
            raise Invalid(f"config[{key!r}] is {config[key]!r}, must be {want!r}")
    # Then the contestant's picks. Every value is checked against the
    # enumerated set BEFORE the reference module is built, so instantiation
    # can only ever allocate one of the shapes listed above.
    for key, allowed in choices.items():
        check_choice(key, config[key], allowed)

    if config.get("format_version") != 3:
        raise Invalid(f"format_version {config.get('format_version')!r} is not 3")
    # 'training' is in allowed_keys, so a missing one already failed above.
    check_training(config["training"])

    blob = members["model.safetensors"]
    header, data_start = parse_safetensors_header(blob)
    header.pop("__metadata__", None)

    expected = reference_tensor_spec(arch, config)
    got = set(header)
    if got != set(expected):
        missing = sorted(set(expected) - got)
        unknown = sorted(got - set(expected))
        raise Invalid(f"tensor name mismatch; missing={missing} unknown={unknown}")

    payload = 0
    spans = []
    for name, spec in header.items():
        if not isinstance(spec, dict):
            raise Invalid(f"tensor {name!r} has a malformed header entry")
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if dtype not in ALLOWED_DTYPES:
            raise Invalid(f"tensor {name!r} has dtype {dtype!r}, allowed: {sorted(ALLOWED_DTYPES)}")
        if not isinstance(shape, list) or len(shape) > MAX_TENSOR_RANK:
            raise Invalid(f"tensor {name!r} has rank {len(shape) if isinstance(shape, list) else '?'}")
        if any(not isinstance(d, int) or d < 0 for d in shape):
            raise Invalid(f"tensor {name!r} has a non-integer or negative dimension: {shape}")
        elements = 1
        for d in shape:
            elements *= d
            if elements > MAX_TENSOR_ELEMENTS:
                raise Invalid(f"tensor {name!r} claims {elements} elements, "
                              f"limit is {MAX_TENSOR_ELEMENTS}")
        if tuple(shape) != expected[name]:
            raise Invalid(f"tensor {name!r} has shape {tuple(shape)}, must be {expected[name]}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(o, int) and o >= 0 for o in offsets)
                or offsets[1] < offsets[0]
                or data_start + offsets[1] > len(blob)):
            raise Invalid(f"tensor {name!r} has data_offsets outside the file")
        if offsets[1] - offsets[0] != elements * 4:
            raise Invalid(f"tensor {name!r} byte range does not match its shape")
        payload += elements * 4
        spans.append((offsets[0], offsets[1], name))

    # The size gate counts REFERENCED bytes, so on its own it would let a
    # submission staple unlimited junk after the last tensor: the payload total
    # stays small while the file does not. Require the tensor region to tile
    # the data segment exactly -- no gaps, no overlaps, nothing trailing.
    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if start != cursor:
            raise Invalid(f"tensor {name!r} starts at {start}, expected {cursor}: "
                          f"the data segment has a gap or an overlap")
        cursor = end
    if data_start + cursor != len(blob):
        extra = len(blob) - (data_start + cursor)
        raise Invalid(f"model.safetensors has {extra} bytes after the last tensor")

    if payload > MAX_WEIGHT_BYTES:
        raise Invalid(f"weights are {payload/1024/1024:.2f} MiB, limit is "
                      f"{MAX_WEIGHT_BYTES/1024/1024:.0f} MiB")

    # The encoder ties one embedding table across the flat, item and beast
    # paths. The file has to carry a copy per name, and load_state_dict writes
    # them into shared storage, so whichever lands last wins. Disagreeing
    # copies are a silent, order-dependent reinterpretation of the model.
    check_tensor_values(arch, config, blob)

    if verbose:
        print(f"architecture      {arch}")
        print(f"tensors           {len(header)}")
        print(f"parameters        {payload // 4:,}")
        print(f"weight bytes      {payload/1024/1024:.2f} MiB  "
              f"(limit {MAX_WEIGHT_BYTES/1024/1024:.0f} MiB)")
    return {"config": config, "blob": blob}


def unused_prefixes(arch: str, config: dict) -> tuple:
    """Tensor prefixes the reference module defines but never reads.

    LSTMPolicy builds self.lstm unconditionally, but the use_lstm=False forward
    path touches neither the cell nor the gate. Those tensors are half the file
    at some widths, so a stateless submission must not have to spend its byte
    budget on weights that provably never run.
    """
    if arch == "dm_mlp_v1":
        return ("lstm",)   # covers both `lstm.*` and `lstm_gate`
    return ()


def build_reference_model(arch: str, config: dict):
    """Instantiate the evaluator's own module for a validated config."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import train

    if arch not in ARCHITECTURES:
        raise Invalid(f"no reference module for {arch!r}")
    # These live as module globals and would silently change the tensor set, so
    # an architecture name would stop meaning one shape.
    for name, want in (("DECOUPLED_VALUE", False), ("QUANTILE_VALUE", False),
                       ("EQUIVARIANT_HEAD", False), ("SPR_AUX", False)):
        if getattr(train, name, False) != want:
            raise Invalid(f"reference module is configured with {name}="
                          f"{getattr(train, name)!r}; the whitelist assumes {want!r}")
    if arch == "dm_transformer_v1":
        return train.TokenPolicy(
            d_model=config["d_model"], n_layers=config["n_layers"],
            n_heads=config["n_heads"], ff_dim=config["ff_dim"],
            act_dim=config["act_dim"],
        )
    if getattr(train, "MEMORY_TYPE", "lstm") != "lstm":
        raise Invalid(f"reference module is configured for "
                      f"MEMORY_TYPE={train.MEMORY_TYPE!r}, not 'lstm'")
    return train.LSTMPolicy(
        obs_dim=config["obs_dim"],
        act_dim=config["act_dim"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_trunk_blocks=config["num_trunk_blocks"],
        use_lstm=(arch == "dm_lstm_v1"),
    )


def reference_tensor_spec(arch: str, config: dict) -> dict[str, tuple]:
    """Name -> shape for every tensor a submission must carry.

    Built by instantiating the evaluator's own module, so the whitelist cannot
    drift from the code that will consume the weights, minus whatever this
    config never reads.
    """
    model = build_reference_model(arch, config)
    skip = unused_prefixes(arch, config)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()
            if not k.startswith(skip)}


def check_tensor_values(arch: str, config: dict, blob: bytes) -> None:
    """Value-level checks, once the header has proved the file safe to read.

    The dtype whitelist constrains the *type* of every number and says nothing
    about the numbers. A file of NaNs is well-formed F32 and would load and
    score like noise, so finiteness is checked here. Names that share storage
    in the reference module must also share values: load_state_dict writes them
    into one buffer, so disagreeing copies resolve by whichever lands last.

    Safe to read the payload at this point: every header bound has been checked
    and the total is inside MAX_WEIGHT_BYTES.
    """
    import torch
    from safetensors.torch import load

    model = build_reference_model(arch, config)
    groups: dict[int, list] = {}
    for name, tensor in model.state_dict().items():
        groups.setdefault(tensor.data_ptr(), []).append(name)

    tensors = load(blob)
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            n_bad = int((~torch.isfinite(tensor)).sum())
            raise Invalid(f"tensor {name!r} has {n_bad} non-finite values "
                          f"(NaN or inf)")
    for names in groups.values():
        present = [n for n in names if n in tensors]
        if len(present) < 2:
            continue
        first = tensors[present[0]]
        for other in present[1:]:
            if not torch.equal(first, tensors[other]):
                raise Invalid(f"tensors {present[0]!r} and {other!r} are tied in this "
                              f"architecture and must hold identical values")


def load_into_model(result: dict):
    """The evaluator's load path: architecture from the whitelist, weights from
    the file, strict key matching. No contestant code runs at any point."""
    from safetensors.torch import load

    config = result["config"]
    model = build_reference_model(config["architecture"], config)
    # Start from the module's own tensors so anything this config never reads
    # keeps its init values, then overwrite with the submission. The load stays
    # strict: validate() already proved the submitted names are exactly the
    # ones this config requires.
    state = model.state_dict()
    state.update(load(result["blob"]))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--load", action="store_true",
                    help="also build the reference model and load the weights")
    args = ap.parse_args()

    try:
        result = validate(args.submission)
        if args.load:
            model = load_into_model(result)
            n = sum(p.numel() for p in model.parameters())
            print(f"loaded            {type(model).__name__}, {n:,} params")
    except Invalid as exc:
        print(f"REJECTED  {args.submission}: {exc}", file=sys.stderr)
        return 1
    except zipfile.BadZipFile as exc:
        print(f"REJECTED  {args.submission}: not a zip file ({exc})", file=sys.stderr)
        return 1
    print(f"OK        {args.submission}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
