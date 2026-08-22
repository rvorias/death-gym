"""Competition submission format: round-trip, and the rejections that matter.

The point of the format is that a submission is data, never code. These tests
build hostile submissions and assert the validator refuses each one.
"""
from __future__ import annotations

import json
import pickle
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.validate_submission import (  # noqa: E402
    ARCHITECTURES,
    Invalid,
    load_into_model,
    validate,
)

ARCH = "dm_lstm_v1"

# One point in the allowed config space, used as the default for every test
# that is not specifically about config variation.
DEFAULT_CONFIG = {
    "architecture": ARCH,
    **ARCHITECTURES[ARCH]["fixed"],
    "embed_dim": 256,
    "hidden_dim": 256,
    "num_trunk_blocks": 1,
}
DEFAULT_TRAINING = {"reward": {"xp": 1.0}, "params": {"lr": 0.0001, "num_envs": 2048}}
DEFAULT_CONFIG = {**DEFAULT_CONFIG, "format_version": 3, "training": DEFAULT_TRAINING}


def weights_for(config):
    """The reference module's weights for a given config."""
    import tools.validate_submission as v
    model = v.build_reference_model(config["architecture"], config)
    return {k: t.detach().to(torch.float32).clone()
            for k, t in model.state_dict().items()
            if not k.startswith(v.unused_prefixes(config["architecture"], config))}


@pytest.fixture(scope="module")
def reference_weights():
    """A valid weight set, built from the architecture the evaluator owns."""
    return weights_for(DEFAULT_CONFIG)


def write_submission(path: Path, weights=None, config=None, overlay=None,
                     extra_files=None, raw_members=None) -> Path:
    """Build a submission zip. Every argument is an escape hatch for a test."""
    cfg = dict(DEFAULT_CONFIG) if config is None else dict(config)
    if overlay is not None:
        cfg.update(overlay)

    st_path = path.parent / "model.safetensors"
    if weights is not None:
        save_file(weights, str(st_path))
        blob = st_path.read_bytes()
        st_path.unlink()
    else:
        blob = b""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        if raw_members is None:
            z.writestr("model.safetensors", blob)
            z.writestr("config.json", json.dumps(cfg))
        else:
            for name, data in raw_members.items():
                z.writestr(name, data)
        for name, data in (extra_files or {}).items():
            z.writestr(name, data)
    return path


# ── The happy path ──────────────────────────────────────────────────────────

def test_valid_submission_round_trips(tmp_path, reference_weights):
    """A model exported and re-loaded must be the same model, tensor for tensor."""
    zip_path = write_submission(tmp_path / "submission.zip", reference_weights)
    result = validate(zip_path, verbose=False)
    model = load_into_model(result)

    loaded = model.state_dict()
    assert set(loaded) == set(reference_weights)
    for name, want in reference_weights.items():
        assert torch.equal(loaded[name], want), f"{name} changed across the round trip"


def test_tied_embeddings_survive_the_round_trip(tmp_path, reference_weights):
    """The encoder ties one embedding across three paths. The export writes each
    name its own bytes; the tie must be back after the evaluator loads."""
    zip_path = write_submission(tmp_path / "submission.zip", reference_weights)
    model = load_into_model(validate(zip_path, verbose=False))
    flat = model.encoder.type_emb.weight
    item = model.encoder.item_encoder.type_emb.weight
    beast = model.encoder.beast_encoder.type_emb.weight
    assert flat.data_ptr() == item.data_ptr() == beast.data_ptr()


def test_export_script_produces_a_valid_submission(tmp_path):
    """The exporter and the validator must agree, as processes, end to end."""
    ckpt = tmp_path / "fake.pt"
    torch.save({"model_state_dict": weights_for(DEFAULT_CONFIG)}, ckpt)

    out = tmp_path / "submission.zip"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/export_submission.py"), str(ckpt), "-o", str(out)],
        check=True, cwd=ROOT, capture_output=True,
    )
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_submission.py"), str(out), "--load"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr


# ── Rejections: the archive ─────────────────────────────────────────────────

def test_rejects_path_traversal(tmp_path, reference_weights):
    zip_path = write_submission(tmp_path / "s.zip", reference_weights,
                                extra_files={"../escape.txt": b"x"})
    with pytest.raises(Invalid, match="traversal|unexpected"):
        validate(zip_path, verbose=False)


def test_rejects_absolute_path(tmp_path, reference_weights):
    zip_path = write_submission(tmp_path / "s.zip", reference_weights,
                                extra_files={"/etc/cron.d/pwn": b"x"})
    with pytest.raises(Invalid, match="absolute path|unexpected"):
        validate(zip_path, verbose=False)


def test_rejects_symlink_member(tmp_path, reference_weights):
    """config.json as a symlink to somewhere else on the evaluator's disk.

    Built from scratch rather than appended: a second config.json would trip
    the duplicate-member check first and never reach the symlink test.
    """
    st = tmp_path / "m.safetensors"
    save_file(reference_weights, str(st))
    zip_path = tmp_path / "s.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.safetensors", st.read_bytes())
        info = zipfile.ZipInfo("config.json")
        info.external_attr = (0o120777 << 16)   # S_IFLNK
        z.writestr(info, "/etc/passwd")
    with pytest.raises(Invalid, match="regular file"):
        validate(zip_path, verbose=False)


def test_rejects_extra_file(tmp_path, reference_weights):
    zip_path = write_submission(tmp_path / "s.zip", reference_weights,
                                extra_files={"setup.py": b"import os"})
    with pytest.raises(Invalid, match="unexpected file"):
        validate(zip_path, verbose=False)


def test_rejects_zip_bomb(tmp_path):
    """A member that explodes on decompression is refused from the central
    directory, before a single byte is inflated."""
    zip_path = write_submission(
        tmp_path / "s.zip",
        raw_members={
            "model.safetensors": b"\0" * (8 * 1024 * 1024),
            "config.json": json.dumps(DEFAULT_CONFIG),
        })
    with pytest.raises(Invalid, match="compression ratio|uncompressed"):
        validate(zip_path, verbose=False)


def test_rejects_a_pickle_renamed_as_weights(tmp_path):
    """The whole point: a .pt in a trench coat never reaches torch.load."""
    payload = pickle.dumps({"anything": 1})
    zip_path = write_submission(
        tmp_path / "s.zip",
        raw_members={
            "model.safetensors": payload,
            "config.json": json.dumps(DEFAULT_CONFIG),
        })
    with pytest.raises(Invalid):
        validate(zip_path, verbose=False)


# ── Rejections: the JSON ────────────────────────────────────────────────────

def test_rejects_unknown_architecture(tmp_path, reference_weights):
    cfg = {**DEFAULT_CONFIG, "architecture": "contestant_net_v9"}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="not whitelisted"):
        validate(zip_path, verbose=False)


def test_rejects_resource_attack_in_config(tmp_path, reference_weights):
    """A config claiming an enormous dimension must not reach an allocator."""
    cfg = {**DEFAULT_CONFIG, "sequence_length": 1_000_000_000}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="unknown config keys"):
        validate(zip_path, verbose=False)


def test_rejects_altered_architecture_dimension(tmp_path, reference_weights):
    cfg = {**DEFAULT_CONFIG, "hidden_dim": 1_000_000}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="hidden_dim.*must be one of"):
        validate(zip_path, verbose=False)


def test_rejects_deeply_nested_json(tmp_path, reference_weights):
    meta = {"training": DEFAULT_TRAINING}
    node = meta
    for _ in range(40):
        node["n"] = {}
        node = node["n"]
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, overlay=meta)
    with pytest.raises(Invalid, match="nested deeper"):
        validate(zip_path, verbose=False)


def test_rejects_oversized_json(tmp_path, reference_weights):
    meta = {"note": "A" * 100_000}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, overlay=meta)
    with pytest.raises(Invalid, match="bytes, limit is|compression ratio"):
        validate(zip_path, verbose=False)


# ── Rejections: the tensors ─────────────────────────────────────────────────

def test_rejects_wrong_dtype(tmp_path, reference_weights):
    weights = {k: v.to(torch.float64) for k, v in reference_weights.items()}
    zip_path = write_submission(tmp_path / "s.zip", weights)
    with pytest.raises(Invalid, match="dtype"):
        validate(zip_path, verbose=False)


def test_rejects_unknown_tensor_name(tmp_path, reference_weights):
    weights = dict(reference_weights)
    weights["backdoor.weight"] = torch.zeros(4)
    zip_path = write_submission(tmp_path / "s.zip", weights)
    with pytest.raises(Invalid, match="name mismatch"):
        validate(zip_path, verbose=False)


def test_rejects_wrong_shape(tmp_path, reference_weights):
    weights = dict(reference_weights)
    name = next(k for k, v in reference_weights.items() if v.dim() == 2)
    weights[name] = torch.zeros(reference_weights[name].shape[0] + 1,
                                reference_weights[name].shape[1])
    zip_path = write_submission(tmp_path / "s.zip", weights)
    with pytest.raises(Invalid, match="shape"):
        validate(zip_path, verbose=False)


def test_rejects_absurd_dimension_before_allocating(tmp_path, reference_weights):
    """A header claiming a 10^9-element tensor is refused from the header alone,
    with no payload present and nothing allocated."""
    name = next(iter(reference_weights))
    header = {name: {"dtype": "F32", "shape": [1_000_000_000, 1024],
                     "data_offsets": [0, 16]}}
    raw = json.dumps(header).encode()
    blob = struct.pack("<Q", len(raw)) + raw + b"\0" * 16
    zip_path = write_submission(
        tmp_path / "s.zip",
        raw_members={
            "model.safetensors": blob,
            "config.json": json.dumps(DEFAULT_CONFIG),
        })
    with pytest.raises(Invalid, match="name mismatch|elements|rank"):
        validate(zip_path, verbose=False)


def test_rejects_weights_over_the_size_limit(tmp_path, reference_weights):
    """The size gate is on tensor bytes, so compression cannot hide bulk."""
    weights = dict(reference_weights)
    name = next(k for k, v in reference_weights.items() if v.dim() == 2)
    weights[name] = torch.zeros(3_000_000, dtype=torch.float32)
    zip_path = write_submission(tmp_path / "s.zip", weights)
    with pytest.raises(Invalid, match="shape|limit is"):
        validate(zip_path, verbose=False)


def test_size_gate_fires_on_an_otherwise_valid_submission(tmp_path, reference_weights,
                                                          monkeypatch):
    """Isolate the size gate: correct names, shapes and dtypes, budget lowered
    below what the reference model needs. Nothing but size may reject it."""
    import tools.validate_submission as v
    monkeypatch.setattr(v, "MAX_WEIGHT_BYTES", 1024 * 1024)
    zip_path = write_submission(tmp_path / "s.zip", reference_weights)
    with pytest.raises(Invalid, match="limit is 1 MiB"):
        v.validate(zip_path, verbose=False)


def test_size_gate_counts_tensor_bytes_not_compressed_bytes(tmp_path, reference_weights,
                                                            monkeypatch):
    """Zeros compress to almost nothing. The gate must still see the real size."""
    import tools.validate_submission as v
    monkeypatch.setattr(v, "MAX_WEIGHT_BYTES", 1024 * 1024)
    monkeypatch.setattr(v, "MAX_COMPRESSION_RATIO", 10**9)
    zeroed = {k: torch.zeros_like(t) for k, t in reference_weights.items()}
    zip_path = write_submission(tmp_path / "s.zip", zeroed)
    assert zip_path.stat().st_size < 1024 * 1024
    with pytest.raises(Invalid, match="limit is 1 MiB"):
        v.validate(zip_path, verbose=False)


# ── Config variation: what a contestant may and may not change ──────────────

@pytest.mark.parametrize("embed_dim,hidden_dim,num_trunk_blocks", [
    (128, 128, 1),
    (128, 256, 2),
    (512, 512, 3),   # the largest model the whitelist permits
])
def test_contestant_may_vary_the_architecture(tmp_path, embed_dim, hidden_dim,
                                              num_trunk_blocks):
    """Any point in the enumerated config space must validate and load."""
    cfg = {**DEFAULT_CONFIG, "embed_dim": embed_dim, "hidden_dim": hidden_dim,
           "num_trunk_blocks": num_trunk_blocks}
    weights = weights_for(cfg)
    zip_path = write_submission(tmp_path / "s.zip", weights, config=cfg)
    model = load_into_model(validate(zip_path, verbose=False))
    loaded = model.state_dict()
    assert set(loaded) == set(weights)
    for name, want in weights.items():
        assert torch.equal(loaded[name], want)


def test_largest_allowed_model_is_under_the_size_limit():
    """The whitelist must not permit a model the size gate would then reject."""
    import tools.validate_submission as v
    schema = ARCHITECTURES[ARCH]["choices"]
    cfg = {**DEFAULT_CONFIG, "embed_dim": max(schema["embed_dim"]),
           "hidden_dim": max(schema["hidden_dim"]),
           "num_trunk_blocks": max(schema["num_trunk_blocks"])}
    payload = sum(t.numel() * 4 for t in weights_for(cfg).values())
    assert payload <= v.MAX_WEIGHT_BYTES, (
        f"largest allowed config is {payload/1024/1024:.2f} MiB, over the "
        f"{v.MAX_WEIGHT_BYTES/1024/1024:.0f} MiB cap")


def test_rejects_config_value_outside_the_enumerated_set(tmp_path, reference_weights):
    """384 is between two allowed widths, and must still be refused."""
    cfg = {**DEFAULT_CONFIG, "hidden_dim": 384}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="must be one of"):
        validate(zip_path, verbose=False)


def test_rejects_int_where_a_bool_is_required(tmp_path, reference_weights):
    """`True == 1` in Python, so a bare membership test would let this through.

    No current choice list is boolean, so this asserts the guard directly.
    """
    import tools.validate_submission as v
    with pytest.raises(Invalid, match="wrong type"):
        v.check_choice("flag", 1, [True, False])
    cfg = {**DEFAULT_CONFIG, "hidden_dim": True}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="wrong type|must be one of"):
        validate(zip_path, verbose=False)


def test_rejects_changing_an_environment_fixed_dimension(tmp_path, reference_weights):
    """obs_dim and act_dim belong to the env, not the contestant."""
    cfg = {**DEFAULT_CONFIG, "obs_dim": 464}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="obs_dim"):
        validate(zip_path, verbose=False)


def test_rejects_missing_config_key(tmp_path, reference_weights):
    cfg = {k: v for k, v in DEFAULT_CONFIG.items() if k != "hidden_dim"}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="missing config keys"):
        validate(zip_path, verbose=False)


def test_weights_must_match_the_declared_config(tmp_path, reference_weights):
    """A config claiming 512 with 256-wide weights is a shape mismatch, not a
    silent reinterpretation."""
    cfg = {**DEFAULT_CONFIG, "hidden_dim": 512}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match="shape"):
        validate(zip_path, verbose=False)


# ── Stateless (use_lstm=false) submissions ─────────────────────────────────

MLP_CONFIG = {**DEFAULT_CONFIG, "architecture": "dm_mlp_v1"}


def stateless_weights(config):
    """weights_for already drops what the family never reads."""
    return weights_for(config)


def test_stateless_submission_omits_the_lstm_tensors(tmp_path):
    cfg = dict(MLP_CONFIG)
    weights = stateless_weights(cfg)
    assert not any(k.startswith("lstm") for k in weights)
    zip_path = write_submission(tmp_path / "s.zip", weights, config=cfg)
    model = load_into_model(validate(zip_path, verbose=False))
    assert model.use_lstm is False
    loaded = model.state_dict()
    for name, want in weights.items():
        assert torch.equal(loaded[name], want)


def test_stateless_model_ignores_its_hidden_state(tmp_path):
    """The flag must change behaviour, not just metadata."""
    cfg = dict(MLP_CONFIG)
    zip_path = write_submission(tmp_path / "s.zip", stateless_weights(cfg), config=cfg)
    model = load_into_model(validate(zip_path, verbose=False))
    obs = torch.randn(4, cfg["obs_dim"])
    mask = torch.ones(4, cfg["act_dim"], dtype=torch.bool)
    h = cfg["hidden_dim"]
    zero = (torch.zeros(1, 4, h), torch.zeros(1, 4, h))
    noise = (torch.randn(1, 4, h), torch.randn(1, 4, h))
    with torch.no_grad():
        a, _, _ = model.step(obs, mask, zero)
        b, _, _ = model.step(obs, mask, noise)
    assert torch.allclose(a, b)


def test_stateless_submission_is_materially_smaller(tmp_path):
    """The whole point of the omission: budget freed for the rest of the model."""
    cfg = dict(MLP_CONFIG)
    lean = sum(t.numel() for t in stateless_weights(cfg).values())
    full = sum(t.numel() for t in weights_for(DEFAULT_CONFIG).values())
    assert lean < full / 2


def test_rejects_mlp_entry_carrying_lstm_tensors(tmp_path):
    """Declaring dm_mlp_v1 then shipping the cell anyway is a name mismatch."""
    zip_path = write_submission(tmp_path / "s.zip", weights_for(DEFAULT_CONFIG),
                                config=dict(MLP_CONFIG))
    with pytest.raises(Invalid, match="name mismatch"):
        validate(zip_path, verbose=False)


def test_rejects_lstm_entry_missing_lstm_tensors(tmp_path):
    """And the converse: dm_lstm_v1 must carry them."""
    zip_path = write_submission(tmp_path / "s.zip", weights_for(MLP_CONFIG),
                                config=dict(DEFAULT_CONFIG))
    with pytest.raises(Invalid, match="name mismatch"):
        validate(zip_path, verbose=False)


def test_export_drops_lstm_tensors_for_an_mlp_checkpoint(tmp_path):
    """The exporter and the validator must agree on the stateless tensor set."""
    import checkpoint as ckpt_mod
    cfg = dict(MLP_CONFIG)
    ckpt = ckpt_mod.save(tmp_path / "fake", weights_for(cfg),
                         arch={**cfg, "memory_type": "lstm", "decoupled_value": False,
                               "quantile_value": False, "equivariant_head": False,
                               "spr_aux": False})
    out = tmp_path / "submission.zip"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/export_submission.py"), str(ckpt), "-o", str(out)],
        check=True, cwd=ROOT, capture_output=True,
    )
    with zipfile.ZipFile(out) as z:
        cfg = json.loads(z.read("config.json"))
    assert cfg["architecture"] == "dm_mlp_v1"
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_submission.py"), str(out), "--load"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr


# ── Integrity holes found by adversarial review ────────────────────────────

def test_rejects_trailing_bytes_after_the_tensors(tmp_path, reference_weights):
    """The size gate counts referenced bytes, so junk stapled after the last
    tensor would otherwise ride along uncounted."""
    import os
    st = tmp_path / "m.safetensors"
    save_file(reference_weights, str(st))
    padded = st.read_bytes() + os.urandom(1024 * 1024)
    zip_path = write_submission(
        tmp_path / "s.zip",
        raw_members={"model.safetensors": padded,
                     "config.json": json.dumps(DEFAULT_CONFIG)})
    with pytest.raises(Invalid, match="bytes after the last tensor"):
        validate(zip_path, verbose=False)


def test_rejects_duplicate_archive_members(tmp_path, reference_weights):
    """Keying members by filename would keep the last and hide the first."""
    zip_path = tmp_path / "s.zip"
    write_submission(zip_path, reference_weights)
    with zipfile.ZipFile(zip_path, "a") as z:
        z.writestr("config.json", json.dumps(DEFAULT_CONFIG))
    with pytest.raises(Invalid, match="duplicate archive member"):
        validate(zip_path, verbose=False)


def test_rejects_disagreeing_tied_tensors(tmp_path, reference_weights):
    """Tied names share storage on load, so the last write silently wins."""
    weights = dict(reference_weights)
    tied = "encoder.beast_encoder.type_emb.weight"
    weights[tied] = torch.full_like(weights["encoder.type_emb.weight"], 999.0)
    zip_path = write_submission(tmp_path / "s.zip", weights)
    with pytest.raises(Invalid, match="tied in this architecture"):
        validate(zip_path, verbose=False)


def test_pytorch_checkpoint_gets_a_useful_message(tmp_path, reference_weights):
    """A .pt is itself a zip; without a sniff the error is its member count."""
    ckpt = tmp_path / "final.pt"
    torch.save({"model_state_dict": reference_weights}, ckpt)
    with pytest.raises(Invalid, match="PyTorch checkpoint, not a submission"):
        validate(ckpt, verbose=False)


def test_renamed_pickle_gets_a_useful_message(tmp_path, reference_weights):
    """Renaming a .pt to model.safetensors must not surface as 'header length'."""
    import pickle
    zip_path = write_submission(
        tmp_path / "s.zip",
        raw_members={"model.safetensors": pickle.dumps({"a": 1}),
                     "config.json": json.dumps(DEFAULT_CONFIG)})
    with pytest.raises(Invalid, match="not a safetensors file"):
        validate(zip_path, verbose=False)


def test_unexpected_file_message_names_the_expected_set(tmp_path, reference_weights):
    zip_path = write_submission(tmp_path / "s.zip", reference_weights,
                                extra_files={"notes.txt": b"x"})
    with pytest.raises(Invalid, match="a submission contains exactly"):
        validate(zip_path, verbose=False)


# ── Three families, and the declared training record ───────────────────────

TRANSFORMER_CONFIG = {
    "architecture": "dm_transformer_v1",
    **ARCHITECTURES["dm_transformer_v1"]["fixed"],
    "d_model": 128, "n_layers": 2, "n_heads": 4, "ff_dim": 512,
    "format_version": 3, "training": DEFAULT_TRAINING,
}


@pytest.mark.parametrize("config", [
    pytest.param(DEFAULT_CONFIG, id="lstm"),
    pytest.param({**DEFAULT_CONFIG, "architecture": "dm_mlp_v1"}, id="mlp"),
    pytest.param(TRANSFORMER_CONFIG, id="transformer"),
])
def test_every_family_round_trips(tmp_path, config):
    weights = weights_for(config)
    zip_path = write_submission(tmp_path / "s.zip", weights, config=dict(config))
    model = load_into_model(validate(zip_path, verbose=False))
    loaded = model.state_dict()
    for name, want in weights.items():
        assert torch.equal(loaded[name], want)


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_largest_allowed_model_fits_for_every_family(arch):
    """Widening a choice list must fail here, not at an entrant's packing step."""
    import itertools
    import tools.validate_submission as v
    schema = ARCHITECTURES[arch]
    keys = list(schema["choices"])
    worst = 0
    for values in itertools.product(*(schema["choices"][k] for k in keys)):
        cfg = {"architecture": arch, **schema["fixed"], **dict(zip(keys, values))}
        if arch == "dm_transformer_v1" and cfg["d_model"] % cfg["n_heads"]:
            continue
        worst = max(worst, sum(t.numel() * 4 for t in weights_for(cfg).values()))
    assert worst <= v.MAX_WEIGHT_BYTES, (
        f"{arch} worst case is {worst/1024/1024:.2f} MiB, over the cap")


def test_transformer_config_is_rejected_for_an_lstm_entry(tmp_path):
    cfg = {**DEFAULT_CONFIG, "d_model": 128}
    zip_path = write_submission(tmp_path / "s.zip", weights_for(DEFAULT_CONFIG), config=cfg)
    with pytest.raises(Invalid, match="unknown config keys"):
        validate(zip_path, verbose=False)


def test_training_block_is_required(tmp_path, reference_weights):
    cfg = {k: v for k, v in DEFAULT_CONFIG.items() if k != "training"}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, config=cfg)
    with pytest.raises(Invalid, match=r"missing config keys: \['training'\]"):
        validate(zip_path, verbose=False)


@pytest.mark.parametrize("training,pattern", [
    ({"reward": {}, "params": {}, "extra": {}}, "unknown sections"),
    ({"params": {"lr": float("inf")}}, "not finite"),
    ({"params": {"lr": 1e30}}, "out of range"),
    ({"params": {"Bad Key": 1}}, "not a lowercase identifier"),
    # 100 chars: over the training block's 64-char cap, under the global
    # 256-char json string cap, so this reaches the check under test.
    ({"params": {"note": "x" * 100}}, "must be a number"),
    ({"params": {"note": "x" * 500}}, "json string longer than"),
    ({"params": {"nested": {"a": 1}}}, "must be a number"),
    ({"reward": [1, 2, 3]}, "must be an object"),
])
def test_training_block_is_bounded(tmp_path, reference_weights, training, pattern):
    """Declared provenance must never become a resource attack."""
    meta = {"training": training}
    zip_path = write_submission(tmp_path / "s.zip", reference_weights, overlay=meta)
    with pytest.raises(Invalid, match=pattern):
        validate(zip_path, verbose=False)


def test_training_block_survives_a_real_export(tmp_path):
    """The reward weights and hyperparameters must reach the submission."""
    import checkpoint as ckpt_mod
    arch = {**DEFAULT_CONFIG, "memory_type": "lstm", "decoupled_value": False,
            "quantile_value": False, "equivariant_head": False, "spr_aux": False}
    training = {"reward": {"xp": 2.5}, "params": {"lr": 0.0003, "num_envs": 4096}}
    ckpt = ckpt_mod.save(tmp_path / "c", weights_for(DEFAULT_CONFIG), arch=arch,
                         meta={"training": training})
    out = tmp_path / "submission.zip"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/export_submission.py"), str(ckpt), "-o", str(out)],
        check=True, cwd=ROOT, capture_output=True,
    )
    with zipfile.ZipFile(out) as z:
        meta = json.loads(z.read("config.json"))
    assert meta["training"] == training


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_weights(tmp_path, reference_weights, bad):
    """F32 constrains the type of every number, not the numbers. A file of NaNs
    is well-formed and would load and score like noise."""
    weights = dict(reference_weights)
    name = next(iter(weights))
    weights[name] = torch.full_like(weights[name], bad)
    zip_path = write_submission(tmp_path / "s.zip", weights)
    with pytest.raises(Invalid, match="non-finite"):
        validate(zip_path, verbose=False)


def test_accepts_ordinary_finite_weights(tmp_path, reference_weights):
    """The finiteness gate must not reject a normal entry."""
    zip_path = write_submission(tmp_path / "s.zip", reference_weights)
    validate(zip_path, verbose=False)
