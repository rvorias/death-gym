# Extracted from research/zoo.py; see manifest.json for source hash.
import os, importlib, torch
import policy_core as train
import checkpoint, semantic_obs
REQ = dict(obs_dim=463, act_dim=57)

def set_globals(a):
    """Every arch flag must be applied: LSTMPolicy reads them as module globals
    at __init__, so a mismatch silently builds the wrong network."""
    train.EMBED_DIM = a.get("embed_dim", 256)
    train.HIDDEN_DIM = a.get("hidden_dim", 256)
    train.TRUNK_NUM_BLOCKS = a.get("num_trunk_blocks", 1)
    train.EQUIVARIANT_HEAD = a.get("equivariant_head", False)
    train.DECOUPLED_VALUE = a.get("decoupled_value", False)
    train.QUANTILE_VALUE = a.get("quantile_value", False)
    train.SPR_AUX = a.get("spr_aux", False)
    train.ITEM_MECHANICS = a.get("item_mechanics", False)
    if train.ITEM_MECHANICS and a.get("item_mechanics_version") != 1:
        raise ValueError("unsupported item mechanics feature version")
    train.MEMORY_TYPE = a.get("memory_type", "lstm")
    # newer arch flags -- omitting these silently builds the WRONG network
    # (a set-encoder checkpoint has a 271-wide encoder input, not 703).
    for g, k in (("SET_ENCODER", "set_encoder"), ("SET_ATTN", "set_attn"),
                 ("SET_NORM", "set_norm"), ("STAT_HEAD", "stat_head"),
                 ("ITEM_TF", "item_tf"), ("ITEM_TF_AFFORD", "item_tf_afford"),
                 ("ITEM_TF_SLOT_BIAS", "item_tf_slot_bias"),
                 ("ITEM_TF_CONTENT_NORM", "item_tf_content_norm"),
                 ("PRIV_CRITIC", "priv_critic"), ("SLOT_PAIR", "slot_pair")):
        if hasattr(train, g):
            setattr(train, g, a.get(k, False))
    for g, k, d in (("ITEM_TF_DIM", "item_tf_dim", 16),
                    ("ITEM_TF_LAYERS", "item_tf_layers", 1),
                    ("ITEM_TF_HEADS", "item_tf_heads", 4),
                    ("ITEM_TF_ID_SCALE", "item_tf_id_scale", 1.0),
                    ("THINK_STEPS", "think_steps", 1),
                    ("SET_SEEDS", "set_seeds", 2)):
        if hasattr(train, g):
            setattr(train, g, a.get(k, d))
    if hasattr(train, "LSTM_HIDDEN"):
        # pre---lstm-hidden checkpoints: recurrent width WAS hidden_dim
        train.LSTM_HIDDEN = a.get("lstm_hidden", a.get("hidden_dim", 256))
    if hasattr(train, "TRUNK_BLOCK"):
        train.TRUNK_BLOCK = a.get("trunk_block", "res")

def load(path, dev, semantic=None):
    """semantic=None (default) auto-detects: newer checkpoints record
    `semantic_obs` in the arch dict; older ones predate that field, so fall
    back to trying both encoder widths. Pass True/False only to force it."""
    r = (torch.load(path, map_location="cpu", weights_only=False)
         if path.endswith(".pt") else checkpoint.load(path, map_location="cpu"))
    a = r["arch"]
    for k, v in REQ.items():
        if a.get(k) != v:
            raise ValueError(f"{k}={a.get(k)} != {v}")
    # SEMANTIC_OBS must be set BEFORE the encoder is built -- it sizes the input
    # projection. `--semantic` is NOT recorded in the checkpoint arch, so the
    # caller has to know; getting it wrong fails as a 615-vs-676 shape error.
    if semantic is None and "semantic_obs" in a:
        semantic = bool(a["semantic_obs"])
    # v3 widens EXTRA_DIM 61 -> 77, so the flag alone is not enough
    # set the v3 flag BOTH ways: a pre-set env var must not leak into loading a
    # v1+v2 checkpoint, which would fail as a 676-vs-703 shape mismatch.
    want_v3 = "1" if a.get("obs_extra_dim", 0) > 61 else "0"
    want_v4 = "1" if a.get("obs_extra_dim", 0) >= 90 else "0"
    if (os.environ.get("DM_SEM_V3", "0"), os.environ.get("DM_SEM_V4", "1")) != (want_v3, want_v4):
        os.environ["DM_SEM_V3"] = want_v3
        os.environ["DM_SEM_V4"] = want_v4
        importlib.reload(semantic_obs)
    cands = [semantic] if semantic is not None else [False, True]
    err = None
    for sem in cands:
        # SEMANTIC_OBS must be set BEFORE the encoder is built: it sizes the
        # input projection. Older checkpoints do not record it, so try both.
        train.SEMANTIC_OBS = sem
        set_globals(a)
        net = train.LSTMPolicy(obs_dim=train.OBS_DIM, act_dim=train.ACT_DIM,
                               use_lstm=a.get("memory_type", "lstm") == "lstm")
        try:
            net.load_state_dict(r["model"])
        except RuntimeError as e:
            err = e; continue
        a = dict(a); a["semantic_obs"] = sem
        return net.to(dev).eval(), a
    raise RuntimeError(f"could not load {path} as semantic or non-semantic: {err}")
