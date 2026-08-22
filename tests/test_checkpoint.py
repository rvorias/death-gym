"""Checkpoint round-trip, and the architecture record that makes it unambiguous."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import checkpoint  # noqa: E402
import train  # noqa: E402


def build(embed_dim=256, hidden_dim=256, num_trunk_blocks=1, use_lstm=True):
    model = train.LSTMPolicy(obs_dim=463, act_dim=57, embed_dim=embed_dim,
                             hidden_dim=hidden_dim,
                             num_trunk_blocks=num_trunk_blocks, use_lstm=use_lstm)
    arch = {"architecture": "dm_lstm_v1", "obs_dim": 463, "act_dim": 57,
            "embed_dim": embed_dim, "hidden_dim": hidden_dim,
            "num_trunk_blocks": num_trunk_blocks, "use_lstm": use_lstm,
            "memory_type": "lstm", "decoupled_value": False,
            "quantile_value": False, "equivariant_head": False, "spr_aux": False}
    return model, arch


def test_checkpoint_is_not_a_pickle(tmp_path):
    """The whole point: loading a checkpoint must not execute code."""
    model, arch = build()
    path = checkpoint.save(tmp_path / "c", model.state_dict(), arch=arch)
    assert path.suffix == ".safetensors"
    head = path.read_bytes()[:16]
    assert head[8:9] == b"{"          # safetensors header, not a pickle opcode
    assert not head.startswith(b"PK")


def test_round_trips_weights_and_arch(tmp_path):
    model, arch = build(embed_dim=128, hidden_dim=256, num_trunk_blocks=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.zeros(1, 463)) if False else None
    path = checkpoint.save(tmp_path / "c", model.state_dict(), arch=arch,
                           meta={"iteration": 7}, optimizer_state=opt.state_dict())
    got = checkpoint.load(path)
    assert got["arch"] == arch
    assert got["meta"]["iteration"] == 7
    assert got["legacy"] is False
    for name, want in model.state_dict().items():
        assert torch.equal(got["model"][name], want)


def test_optimizer_state_survives(tmp_path):
    model, arch = build()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.state_dict()["policy_head.weight"]
    loss = sum(p.sum() for p in model.parameters())
    loss.backward()
    opt.step()
    path = checkpoint.save(tmp_path / "c", model.state_dict(), arch=arch,
                           optimizer_state=opt.state_dict())
    got = checkpoint.load(path)
    fresh = train.LSTMPolicy(obs_dim=463, act_dim=57)
    opt2 = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    opt2.load_state_dict(got["optimizer"])
    assert len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"])


@pytest.mark.parametrize("embed_dim,hidden_dim,blocks", [(128, 128, 1), (128, 256, 2),
                                                         (512, 512, 3)])
def test_arch_record_matches_the_weights(tmp_path, embed_dim, hidden_dim, blocks):
    """The record must describe the tensors actually written, at every shape.

    num_trunk_blocks previously bound TRUNK_NUM_BLOCKS at class-definition
    time, so a non-default value was announced but never built.
    """
    model, arch = build(embed_dim, hidden_dim, blocks)
    path = checkpoint.save(tmp_path / "c", model.state_dict(), arch=arch)
    got = checkpoint.load(path)
    built = {k.split(".")[2] for k in got["model"] if k.startswith("input_trunk.blocks.")}
    assert len(built) == got["arch"]["num_trunk_blocks"] == blocks
    assert got["model"]["input_trunk.proj.weight"].shape[1] == embed_dim
    assert got["model"]["lstm.weight_hh_l0"].shape[1] == hidden_dim


def test_legacy_pickle_is_still_readable(tmp_path):
    model, _ = build(embed_dim=128, hidden_dim=256, num_trunk_blocks=2)
    legacy = tmp_path / "old.pt"
    torch.save({"model_state_dict": model.state_dict(), "iteration": 3,
                "avg_xp": 12.5}, legacy)
    got = checkpoint.load(legacy)
    assert got["legacy"] is True
    assert got["meta"]["iteration"] == 3
    assert got["arch"]["embed_dim"] == 128
    assert got["arch"]["hidden_dim"] == 256
    assert got["arch"]["num_trunk_blocks"] == 2
    # use_lstm cannot be recovered from a .pt and must be flagged as a guess.
    assert got["arch"]["use_lstm_inferred"] is True


def test_transformer_checkpoint_gets_a_clear_error(tmp_path):
    """It has no whitelisted architecture; the failure must say so."""
    legacy = tmp_path / "tok.pt"
    torch.save({"model_state_dict": {"blocks.0.attn.weight": torch.zeros(4, 4)}}, legacy)
    with pytest.raises(ValueError, match="Transformer .* not supported"):
        checkpoint.load(legacy)
