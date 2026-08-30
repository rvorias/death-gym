"""
RL training for Death Mountain gym using PPO with action masking.
MLP policy with categorical embedding encoder.
Usage: python train.py [--resume path/to/checkpoint.pt]
"""

import argparse
import os

# Make CUDA_VISIBLE_DEVICES mean what nvidia-smi shows. CUDA orders devices
# FASTEST_FIRST by default, so on a mixed-GPU host CUDA_VISIBLE_DEVICES=0 can
# select a different card than `nvidia-smi -i 0` reports. Pin the ordering
# before torch initialises CUDA. Export CUDA_DEVICE_ORDER yourself to override.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import time
import shutil

import numpy as np
import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from rewards import ACTION_DIM, OBS_DIM, GameEnv, Info, item_xp_per_step_config, rich_reward_config

# -- Hyperparameters ----------------------------------------------------------

NUM_ENVS = 2048
MAX_STEPS = 2048
REWARD_CONFIG = {
    **rich_reward_config(),
    # CURRICULUM-STRIP EXPERIMENT (2026-06-13): the economy curriculum was
    # tuned at 10M steps, three regimes ago; "pure-XP loses" was explicitly
    # budget-conditioned to 10M. At 250M/2048 envs the curriculum's economy
    # bonuses may be the cage that holds the fight-shy local optimum.
    # Fundamentals only:
    "xp_shaping_scale": 500.0,
    "death_penalty": -0.25,
    "kill": 1.0,
    "level_up": 1.0,
    # zero out the economy/behavior curriculum:
    "buy_item": 0.0,
    "buy_potion": 0.0,
    "buy_t1_item": 0.0,
    "buy_weapon_t1": 0.0,
    "buy_quality_delta": 0.0,
    "buy_asset_bank_delta": 0.0,
    "switch_survivability": 0.0,
    "stat_upgrade": 0.0,
    "item_xp_gain": 0.0,
    "attack_safe_sim": 0.0,
    "flee_safe_sim_penalty": 0.0,
    "flee_hard_sim": 0.0,
    "flee_penalty": 0.0,
    "phase_very_early_end": 3.0,
    "phase_early_end": 10.0,
    "sim_risk_hard_threshold": 1.35,
}
TOTAL_ENV_STEPS = 10_000_000_000
CHECKPOINT_EVERY_STEPS = 250_000_000   # 40 mid-run checkpoints

ACT_DIM = ACTION_DIM

# Trunk: input proj + ResBlocks + LayerNorm. Pre-norm residual, orth init.
EMBED_DIM = 256
HIDDEN_DIM = 256
TRUNK_NUM_BLOCKS = 1
DECOUPLED_VALUE = False  # tie at 100M, reverted (code kept for A/B)
# Self-predictive representation (SPR-lite): an auxiliary loss makes the
# encoder predict its OWN next latent (given the action taken), densifying the
# learning signal — every transition trains the representation, not just the
# reward-relevant ones. Targets the root diagnosis (sample-bound learning,
# signal-starved deep regime). Stop-grad target; cosine loss; masked at episode
# boundaries. A representation change, not a reward change.
SPR_AUX = False  # representation axis closed (neutral-to-negative); flag kept
# Memory module: "lstm" (default, gated-residual LSTM) | "transformer"
# (bounded causal-attention window over recent features, episode-masked,
# gated residual). Last untested structural idea — swaps ONLY the memory.
MEMORY_TYPE = "lstm"  # transformer tied-to-neg @100M; LSTM is the proven incumbent
ATTN_WINDOW = 32
ATTN_HEADS = 4
AUX_COEF = 0.1
AUX_ACTION_EMB = 16

# Flat item encoder: each item gets the same per-field embedding + MLP proj,
# then all 48 items are flattened. exp420's phase-gating sits on the full
# feature stack before the final projection.
ITEM_OUT = 12

LR = 1e-4
GAMMA = 0.995
GAE_LAMBDA = 0.90
CLIP_EPS = 0.2
ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END = 0.005
VALUE_COEF = 0.08
SIL_COEF = 0.0           # --sil-coef: self-imitation auxiliary (positive-surprise BC)
QUANTILE_VALUE = False   # --quantile-value: distributional critic (quantile regression)
# --equivariant-head: shared pointer scorer over bag/market slots instead of
# positional logits. See LSTMPolicy.__init__ for the rationale.
EQUIVARIANT_HEAD = False
# --transformer: entity-token transformer policy (see TokenPolicy).
TOKEN_POLICY = False
TOKEN_D_MODEL, TOKEN_LAYERS, TOKEN_HEADS, TOKEN_FF = 128, 4, 4, 512
EX_NUM_BASE_ACTIONS = 11   # action slots 0-10 before the bag block
PTR_DIM = 32             # query/key width for the pointer head
N_QUANTILES = 32
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 32
PPO_EPOCHS = 8
MINIBATCH_SIZE = 8192
DEATH_XP_WINDOW = 4096
REWARD_TRANSFORM = "symlog"  # per-step reward squash: symlog | sqrt | none

# Expert iteration (search -> distill), enabled with --ei. The search probe
# on the 1B checkpoint showed 1-step lookahead over the exact engine disagrees
# with the policy on 40% of decisions, worth ~+13 xp/decision. Decision states
# are snapshotted during rollout (POD memcpy), searched with the current
# policy by forcing each top-k candidate in rng-decorrelated clones, and the
# search-best actions are distilled back via cross-entropy.
# Deep-game sample upweighting ("mine good behavior"): transitions from
# episodes that have crossed XP >= 600 (the signal-starved regime where
# crossers still die at ~55% of human depth) get extra gradient weight.
# obs[3] = xp / 32767; 600 xp <=> 0.0183.
DEEP_XP_OBS_THRESHOLD = 600.0 / 32767.0

EI_DECISIONS_PER_ROUND = 16  # decision states searched per round
EI_EVERY_ITERS = 2           # search every N training iterations
EI_CLONES = 64               # clone rollouts per candidate action
EI_TOPK = 3                  # candidate actions searched per decision
EI_HORIZON = 150             # max search rollout steps
EI_COEF = 0.2                # CE distillation weight

# -- Obs encoder with categorical embeddings ----------------------------------


class FlatItemEncoder(nn.Module):
    """Shared per-item MLP + flatten.

    Each item is a 9-dim vector [id, tier, type, slot, sp1, sp2, sp3, xp,
    greatness]. Categoricals are embedded with vocab-sized dims (3/3/3/3/2/2);
    continuous fields (tier, xp, greatness) pass through as-is. Per-item
    feature of 19 dims → Linear(19, ITEM_OUT) with shared weights, then
    flatten across all 48 items for (B, 48*ITEM_OUT).

    Transformer attention was shelved (exp460) — 5× slower fps at equivalent
    early learning. Slot/segment semantics emerge from the downstream linear
    proj anyway since positional structure is baked into the flat ordering.

    type/sp1/sp2/sp3 embeddings are shared with BeastEncoder (owned by
    ObsEncoder). The underlying DMFAST_TYPE_* enum is identical across items
    and beasts (0=None, 1=Magic/Cloth, 2=Blade/Hide, 3=Bludgeon/Metal,
    4=Necklace, 5=Ring; beasts only emit 1-3) and special-power
    vocabularies are the same for both — so tying the weights lets each
    token see ~2× the updates.
    """

    BASE_FEATURE_DIM = 19  # id(3)+tier(1)+type(3)+slot(3)+sp1(3)+sp2(2)+sp3(2)+xp(1)+great(1)

    def __init__(self, num_items: int = 48, out_dim: int = ITEM_OUT,
                 type_emb: nn.Embedding = None,
                 sp1_emb: nn.Embedding = None,
                 sp2_emb: nn.Embedding = None,
                 sp3_emb: nn.Embedding = None):
        super().__init__()
        self.num_items = num_items
        self.out_per_item = out_dim

        self.item_id_emb = nn.Embedding(102, 3)
        self.slot_emb    = nn.Embedding(9, 3)
        self.type_emb = type_emb if type_emb is not None else nn.Embedding(6, 3)
        self.sp1_emb  = sp1_emb  if sp1_emb  is not None else nn.Embedding(17, 3)
        self.sp2_emb  = sp2_emb  if sp2_emb  is not None else nn.Embedding(70, 2)
        self.sp3_emb  = sp3_emb  if sp3_emb  is not None else nn.Embedding(19, 2)
        self.item_proj = nn.Linear(self.BASE_FEATURE_DIM, out_dim)

    @property
    def out_dim(self):
        return self.num_items * self.out_per_item

    def forward(self, items):
        """items: (B, 48, 9) → (B, 48, ITEM_OUT), per item.

        Returns the UNFLATTENED per-item tensor; ObsEncoder flattens it for the
        trunk. Keeping the per-item view is what lets the equivariant action
        head score each market/bag slot with one shared scorer instead of 25
        slot-specific output rows."""
        B = items.shape[0]
        feats = torch.cat(
            [
                self.item_id_emb(items[:, :, 0].long().clamp(0, 101)),  # id  → 3
                items[:, :, 1:2],                                       # tier (continuous) → 1
                self.type_emb(items[:, :, 2].long().clamp(0, 5)),       # type → 3 (shared w/ beast)
                self.slot_emb(items[:, :, 3].long().clamp(0, 8)),       # slot → 3
                self.sp1_emb(items[:, :, 4].long().clamp(0, 16)),       # sp1  → 3 (shared w/ beast)
                self.sp2_emb(items[:, :, 5].long().clamp(0, 69)),       # sp2  → 2 (shared w/ beast)
                self.sp3_emb(items[:, :, 6].long().clamp(0, 18)),       # sp3  → 2 (shared w/ beast)
                items[:, :, 7:8],                                       # xp (continuous) → 1
                items[:, :, 8:9],                                       # greatness (continuous) → 1
            ],
            dim=-1,                                                     # total 19 = BASE_FEATURE_DIM
        )
        return F.silu(self.item_proj(feats))


class BeastEncoder(nn.Module):
    """Dedicated beast encoder — embeds all 7 beast fields properly.

    Beast obs layout (ex_pack_obs): [starting_hp, tier, level, type, sp1, sp2, sp3]
    - [0:3] continuous (starting_hp/MAX, tier_monotone/5, log1p-level)
    - [3] type (cat 0-3)
    - [4] sp1 (cat 0-16) ← was silently dropped by the old ObsEncoder
    - [5] sp2 (cat 0-69)
    - [6] sp3 (cat 0-18)

    type/sp1/sp2/sp3 embeddings are shared with FlatItemEncoder (owned by
    ObsEncoder). The type enum is the same for items and beasts — beasts
    only emit indices 1-3 but the shared vocab is size 6 to cover
    item-only Necklace/Ring.
    """

    def __init__(self, out_dim: int,
                 type_emb: nn.Embedding = None,
                 sp1_emb: nn.Embedding = None,
                 sp2_emb: nn.Embedding = None,
                 sp3_emb: nn.Embedding = None):
        super().__init__()
        self.type_emb = type_emb if type_emb is not None else nn.Embedding(6, 3)
        self.sp1_emb  = sp1_emb  if sp1_emb  is not None else nn.Embedding(17, 3)
        self.sp2_emb  = sp2_emb  if sp2_emb  is not None else nn.Embedding(70, 2)
        self.sp3_emb  = sp3_emb  if sp3_emb  is not None else nn.Embedding(19, 2)
        self.scalar_proj = nn.Sequential(
            nn.Linear(3, 6), nn.SiLU(), nn.Linear(6, 6),
        )
        self.out_proj = nn.Sequential(nn.Linear(16, out_dim), nn.SiLU())

    def forward(self, beast):
        """beast: (B, 7) → (B, out_dim)"""
        scalars = beast[:, :3]
        t  = self.type_emb(beast[:, 3].long().clamp(0, 5))
        s1 = self.sp1_emb(beast[:, 4].long().clamp(0, 16))
        s2 = self.sp2_emb(beast[:, 5].long().clamp(0, 69))
        s3 = self.sp3_emb(beast[:, 6].long().clamp(0, 18))
        return self.out_proj(torch.cat([self.scalar_proj(scalars), t, s1, s2, s3], dim=-1))


class ObsEncoder(nn.Module):
    """Flat obs → embed_dim. Phase-gated (exp420), per-item MLP, BeastEncoder
    for beast, small pre-procs for the rest. All concatenated → gate → proj."""

    PHASE_EMB_DIM = 4
    BEAST_OUT_DIM = 12
    SIM_OUT_DIM = 10

    def __init__(self, embed_dim):
        super().__init__()
        self.phase_emb = nn.Embedding(6, self.PHASE_EMB_DIM)
        # Shared embeddings across items and beasts. The DMFAST_TYPE_* enum
        # is identical across the two (0=None, 1=Magic/Cloth, 2=Blade/Hide,
        # 3=Bludgeon/Metal, 4=Necklace, 5=Ring — beasts only emit 1-3), and
        # the special-power vocabs are the same too. Tying gives each token
        # ~2× the updates.
        self.type_emb = nn.Embedding(6, 3)
        self.sp1_emb  = nn.Embedding(17, 3)
        self.sp2_emb  = nn.Embedding(70, 2)
        self.sp3_emb  = nn.Embedding(19, 2)
        self.item_encoder = FlatItemEncoder(
            num_items=48, out_dim=ITEM_OUT,
            type_emb=self.type_emb,
            sp1_emb=self.sp1_emb, sp2_emb=self.sp2_emb, sp3_emb=self.sp3_emb,
        )
        self.beast_encoder = BeastEncoder(
            out_dim=self.BEAST_OUT_DIM,
            type_emb=self.type_emb,
            sp1_emb=self.sp1_emb, sp2_emb=self.sp2_emb, sp3_emb=self.sp3_emb,
        )
        self.adv_proj = nn.Sequential(nn.Linear(13, 13), nn.LayerNorm(13))
        self.sim_proj = nn.Sequential(
            nn.Linear(self.SIM_OUT_DIM, self.SIM_OUT_DIM),
            nn.SiLU(),
            nn.LayerNorm(self.SIM_OUT_DIM),
        )

        total = (self.PHASE_EMB_DIM + 13 + self.item_encoder.out_dim
                 + self.BEAST_OUT_DIM + self.SIM_OUT_DIM)
        # exp420 breakthrough: phase-dependent sigmoid gate over ALL features
        # so the model can attend different to (e.g.) market vs combat features
        # at different phases.
        self.gate = nn.Linear(self.PHASE_EMB_DIM, total)
        self.proj = nn.Linear(total, embed_dim)

    def forward(self, obs, return_items: bool = False):
        """obs: (..., 463) → (..., embed_dim).

        With return_items=True also returns the per-item embeddings
        (..., 48, ITEM_OUT) — equipment[0:8], bag[8:23], market[23:48] — for
        the equivariant action head. Returned rather than cached on the module
        so it survives torch.compile."""
        shape = obs.shape[:-1]
        flat = obs.reshape(-1, 463)
        B = flat.shape[0]

        phase = self.phase_emb(flat[:, 0].long().clamp(0, 5))
        adv   = self.adv_proj(flat[:, 1:14])
        items = torch.cat(
            [
                flat[:, 14:86].reshape(B, 8, 9),
                flat[:, 86:221].reshape(B, 15, 9),
                flat[:, 221:446].reshape(B, 25, 9),
            ],
            dim=1,
        )
        per_item = self.item_encoder(items)              # (B, 48, ITEM_OUT)
        item_out = per_item.reshape(B, -1)
        beast    = self.beast_encoder(flat[:, 446:453])
        sim      = self.sim_proj(flat[:, 453:463])

        features = torch.cat([phase, adv, item_out, beast, sim], dim=-1)
        gated = features * torch.sigmoid(self.gate(phase))
        enc = self.proj(gated).reshape(*shape, -1)
        if return_items:
            return enc, per_item.reshape(*shape, per_item.shape[-2], per_item.shape[-1])
        return enc


# -- Policy -------------------------------------------------------------------


def _init_linear(layer: nn.Linear, gain: float = float(np.sqrt(2))) -> nn.Linear:
    """Orthogonal init for hidden Linear layers (gain=sqrt(2) is the standard
    for ReLU/SiLU in PPO)."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ResBlock(nn.Module):
    """Pre-norm residual block: x + W2(SiLU(W1(LN(x)))).

    Pre-norm is the stable choice for deep MLPs; post-norm diverges at depth
    without careful LR tuning. Orthogonal init with gain=sqrt(2) on the inner
    Linear and gain=1.0 on the output Linear (so residual contribution starts
    small)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = _init_linear(nn.Linear(dim, dim), gain=float(np.sqrt(2)))
        self.fc2 = _init_linear(nn.Linear(dim, dim), gain=1.0)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class Trunk(nn.Module):
    """Input projection + N residual blocks + final LayerNorm."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.proj = _init_linear(nn.Linear(embed_dim, hidden_dim), gain=float(np.sqrt(2)))
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.silu(self.proj(x))
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class WindowAttnMemory(nn.Module):
    """Bounded causal-attention window as in-episode memory (LSTM replacement).

    Each step attends over the last ATTN_WINDOW feature vectors within the SAME
    episode. State is (buffer (W,N,H), valid (W,N,1)) so the harness's existing
    (h,c)-style done-reset (state*keep) and start_state threading work unchanged:
    zeroing the buffer on death = empty history for the new episode. Gated
    residual (out = feats + gate*attn) with gate init 0, so it starts as the
    feed-forward MLP — same philosophy as the LSTM gate that worked.
    """

    def __init__(self, dim, window=ATTN_WINDOW, heads=ATTN_HEADS):
        super().__init__()
        self.dim = dim
        self.window = window
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def init_state(self, n_envs, device):
        buf = torch.zeros(self.window, n_envs, self.dim, device=device)
        valid = torch.zeros(self.window, n_envs, 1, device=device)
        return (buf, valid)

    def step(self, feat, state):
        """feat (1,N,H), state=(buf (W,N,H), valid (W,N,1)) -> (out (N,H), new_state).
        Rolls feat into the buffer (newest slot valid=1), attends the newest
        query over valid slots. Caller zeroes state on done BEFORE next call."""
        buf, valid = state
        nf = feat.detach()  # detach into memory: same rationale as the LSTM path
        buf = torch.cat([buf[1:], nf], dim=0)                    # (W,N,H)
        valid = torch.cat([valid[1:], torch.ones_like(valid[:1])], dim=0)  # (W,N,1)
        q = self.norm(buf[-1:])                                  # (1,N,H)
        k = self.norm(buf)                                       # (W,N,H)
        key_pad = (valid.squeeze(-1) < 0.5).transpose(0, 1)      # (N,W) True=pad
        attn_out, _ = self.attn(q, k, k, key_padding_mask=key_pad, need_weights=False)
        out = (feat + self.gate * attn_out).squeeze(0)           # (N,H)
        return out, (buf, valid)

    def forward_seq(self, feats, start_state, done_seq):
        """feats (T,N,H), start_state=(hist (W,N,H), hist_valid (W,N,1)),
        done_seq (T,N) [done AFTER step t]. Returns out (T,N,H). Causal +
        window + episode-masked attention; replays the per-step memory exactly.
        """
        T, N, H = feats.shape
        W = self.window
        hist, hist_valid = start_state
        # Detach into attention (same as step() and the LSTM path): memory is
        # bolted on; encoder gradient flows only through the residual `feats`.
        keys = torch.cat([hist, feats], dim=0).detach()         # (W+T,N,H)
        keysn = self.norm(keys)
        qn = self.norm(feats.detach())
        dev = feats.device
        # prior-done count strictly before each segment pos: ep[t] (T,N)
        df = done_seq.float()
        ep = torch.cat([torch.zeros(1, N, device=dev), torch.cumsum(df, 0)[:-1]], 0)
        ti = torch.arange(T, device=dev)
        # seg-seg allowed (N,T,T): j<=t, t-j<W, same episode
        jj = ti.view(1, T)          # key seg pos j
        tt = ti.view(T, 1)          # query seg pos t
        causal = (jj <= tt) & ((tt - jj) < W)                   # (T,T)
        ep_t = ep.transpose(0, 1).unsqueeze(2)                  # (N,T,1)
        ep_j = ep.transpose(0, 1).unsqueeze(1)                  # (N,1,T)
        seg_allowed = causal.unsqueeze(0) & (ep_t == ep_j)      # (N,T,T)
        # seg-hist allowed (N,T,W): slot>t (window), hist valid, query in ep 0
        slot = torch.arange(W, device=dev)
        win_h = (slot.view(1, W) > ti.view(T, 1))              # (T,W)
        hv = (hist_valid.squeeze(-1) > 0.5).transpose(0, 1).unsqueeze(1)  # (N,1,W)
        ep0 = (ep_t == 0)                                       # (N,T,1)
        hist_allowed = win_h.unsqueeze(0) & hv & ep0           # (N,T,W)
        allowed = torch.cat([hist_allowed, seg_allowed], dim=2)  # (N,T,W+T)
        disallowed = ~allowed
        am = disallowed.unsqueeze(1).expand(N, self.heads, T, W + T).reshape(N * self.heads, T, W + T)
        attn_out, _ = self.attn(qn, keysn, keysn, attn_mask=am, need_weights=False)
        return feats + self.gate * attn_out                     # (T,N,H)


def mask_fill_value(dtype):
    """Fill value for illegal actions, largest magnitude the dtype allows.

    Must stay FINITE, not -inf: ppo_update relies on masked actions
    contributing exactly 0 to the KL term (prob underflows to 0, times a
    finite log-ratio). -1e8 overflows fp16, whose max is 65504, so half
    precision gets -1e4 -- still enough that softmax underflows to 0.
    fp32 and bf16 keep -1e8 exactly, so their numerics are unchanged.
    """
    return -1e8 if torch.finfo(dtype).max > 1e8 else -1e4


class LSTMPolicy(nn.Module):
    """Recurrent PPO policy with an LSTM after the obs encoder.

    The LSTM gives the agent in-episode memory: HP/gear trajectory, recent
    decisions, the long-horizon survival context that the feed-forward policy
    can't represent. State is per-env (h, c) of shape (1, N, hidden_dim) and
    must be **zeroed on episode end** (auto-reset boundary) — done by the
    rollout and the sequence evaluator.

    Single LSTM shared between policy and value heads (cleaner / fewer params
    than two; standard recurrent PPO).
    """

    def __init__(
        self,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        embed_dim=None,
        hidden_dim=None,
        num_trunk_blocks=None,
        use_lstm=True,
    ):
        super().__init__()
        # Resolve at CALL time so --hidden-dim (module-global override in
        # main) takes effect; class-def-time defaults froze the old value.
        # num_trunk_blocks had exactly that bug: it bound TRUNK_NUM_BLOCKS at
        # def time, so --trunk-blocks printed the new value and built the old
        # one, and the checkpoint's arch record disagreed with its weights.
        embed_dim = EMBED_DIM if embed_dim is None else embed_dim
        hidden_dim = HIDDEN_DIM if hidden_dim is None else hidden_dim
        num_trunk_blocks = (TRUNK_NUM_BLOCKS if num_trunk_blocks is None
                            else num_trunk_blocks)
        self.hidden_dim = hidden_dim
        # use_lstm=False is the exact MLP ablation: gate stays 0 and the LSTM
        # forward is skipped entirely, so out = feats (the pre-LSTM arch).
        self.use_lstm = use_lstm
        self.encoder = ObsEncoder(embed_dim)
        # Single feed-forward trunk before the LSTM, shared by policy + value.
        self.input_trunk = Trunk(embed_dim, hidden_dim, num_trunk_blocks)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
        # Recurrent weight init: orthogonal hh, xavier ih.
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # forget-gate bias = 1 (helps early training)
                n = p.shape[0] // 4
                p.data[n:2*n].fill_(1.0)

        self.policy_head = _init_linear(nn.Linear(hidden_dim, act_dim), gain=0.01)
        # ── Equivariant (pointer) action head ──────────────────────────────
        # The default head is one Linear over all 57 actions, so each of the 25
        # market slots and 15 bag slots owns its own output row: an item seen
        # in slot 7 teaches nothing about the same item in slot 12. That is
        # also why splice is load-bearing — it is a workaround for a missing
        # symmetry rather than a property of the model.
        #
        # Here every slot is scored by ONE shared function of its item
        # embedding against a phase-aware query from the trunk, so every item
        # observation trains the same weights: ~25x the effective data per
        # parameter for the buy head, and permutation equivariance by
        # construction. Base actions and stat upgrades keep positional logits —
        # those are genuinely distinct actions, not interchangeable slots.
        self.equivariant_head = EQUIVARIANT_HEAD
        if self.equivariant_head:
            self.ptr_dim = PTR_DIM
            self.item_key = nn.Sequential(
                nn.Linear(ITEM_OUT, PTR_DIM), nn.SiLU(),
                nn.Linear(PTR_DIM, PTR_DIM))
            # gain 0.01 mirrors policy_head: near-zero scores at init, so the
            # policy starts near-uniform over slots instead of with a random
            # preference baked in.
            self.market_query = _init_linear(nn.Linear(hidden_dim, PTR_DIM), gain=0.01)
            self.bag_query = _init_linear(nn.Linear(hidden_dim, PTR_DIM), gain=0.01)
            self.market_bias = nn.Parameter(torch.zeros(1))
            self.bag_bias = nn.Parameter(torch.zeros(1))
        self.quantile_value = QUANTILE_VALUE
        self.value_head = _init_linear(
            nn.Linear(hidden_dim, N_QUANTILES if QUANTILE_VALUE else 1), gain=1.0)
        # Decoupled value: a second trunk off the shared encoder feeds the value
        # head, so policy (wants "safe now") and value (wants "long-horizon XP
        # across the regime crossing") stop fighting over one representation.
        # Stateless (no LSTM) — the 463-d obs is near-fully-observed.
        self.decoupled_value = DECOUPLED_VALUE
        if self.decoupled_value:
            self.value_trunk = Trunk(embed_dim, hidden_dim, num_trunk_blocks)
        # Learnable scalar gate on the LSTM residual. Init = 0 so the policy
        # starts as *exactly* the feed-forward MLP — no random-LSTM noise tax.
        # Gradient can grow the gate if memory is actually useful.
        self.lstm_gate = nn.Parameter(torch.zeros(1))
        self.memory_type = MEMORY_TYPE
        if self.memory_type == "transformer":
            self.mem = WindowAttnMemory(hidden_dim)
        self.spr_aux = SPR_AUX
        if self.spr_aux:
            self.action_emb = nn.Embedding(act_dim, AUX_ACTION_EMB)
            self.transition = nn.Sequential(
                nn.Linear(embed_dim + AUX_ACTION_EMB, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
        _init_linear(self.encoder.proj, gain=float(np.sqrt(2)))
        _init_linear(self.encoder.item_encoder.item_proj, gain=float(np.sqrt(2)))
        _init_linear(self.encoder.gate, gain=1.0)

    def init_state(self, n_envs: int, device):
        H = self.hidden_dim
        if self.memory_type == "transformer":
            return self.mem.init_state(n_envs, device)
        return (torch.zeros(1, n_envs, H, device=device),
                torch.zeros(1, n_envs, H, device=device))

    def _features(self, obs):
        return self.input_trunk(self.encoder(obs))

    # Action-index layout (57): base[0:11] bag[11:26] market[26:51] stats[51:57]
    # Item-index layout (48):   equip[0:8]  bag[8:23]  market[23:48]
    BAG_A0, BAG_A1, MKT_A0, MKT_A1 = 11, 26, 26, 51
    BAG_I0, BAG_I1, MKT_I0, MKT_I1 = 8, 23, 23, 48

    def _logits(self, out, per_item, action_mask):
        """Masked action logits. With the equivariant head on, the bag and
        market slices are replaced by shared-scorer scores; everything else
        keeps its positional logit."""
        logits = self.policy_head(out)
        if self.equivariant_head and per_item is not None:
            keys = self.item_key(per_item)                       # (..., 48, P)
            scale = self.ptr_dim ** -0.5
            bag = (keys[..., self.BAG_I0:self.BAG_I1, :]
                   * self.bag_query(out).unsqueeze(-2)).sum(-1) * scale + self.bag_bias
            mkt = (keys[..., self.MKT_I0:self.MKT_I1, :]
                   * self.market_query(out).unsqueeze(-2)).sum(-1) * scale + self.market_bias
            # cat rather than index-assign: keeps the graph functional and
            # avoids an in-place write that torch.compile would have to break on.
            logits = torch.cat([logits[..., :self.BAG_A0], bag, mkt,
                                logits[..., self.MKT_A1:]], dim=-1)
        return logits.masked_fill(~action_mask, mask_fill_value(logits.dtype))

    def _value(self, x):
        """Scalar value from the head; quantile mode stashes the full set in
        self._vq for the quantile-regression loss (mean is used for GAE)."""
        v = self.value_head(x)
        if self.quantile_value:
            self._vq = v
            return v.mean(-1)
        return v.squeeze(-1)

    @staticmethod
    def _reset_state(state, done):
        """Zero (h, c) for envs where `done` is True. done: (N,) bool tensor."""
        if done is None:
            return state
        keep = (~done).to(dtype=state[0].dtype).view(1, -1, 1)
        return (state[0] * keep, state[1] * keep)

    def _run_lstm(self, inp, state):
        """Run the LSTM in fp32 with a state whose dtype matches its input.

        Under autocast, cuDNN casts the LSTM input but NOT the hidden state
        it is handed, so a mixed pair raises "Input and hidden tensors are
        not the same dtype". Pinning both to fp32 also keeps the recurrent
        state at full precision, which is what you want for stability. This
        is a no-op on the fp32 path.
        """
        h, c = state
        with torch.autocast("cuda", enabled=False):
            out, new_state = self.lstm(inp.float(), (h.float(), c.float()))
        return out.to(inp.dtype), new_state

    def step(self, obs, action_mask, state):
        """Single-step inference: returns (logits, value, new_state).

        Caller is responsible for zeroing `state` on done BEFORE the call so the
        new episode's first obs sees a clean LSTM state.
        """
        enc, per_item = self.encoder(obs, return_items=True)
        feats = self.input_trunk(enc).unsqueeze(0)  # (1, N, H)
        if self.memory_type == "transformer":
            out, new_state = self.mem.step(feats, state)
        elif not self.use_lstm:
            out = feats.squeeze(0)
            new_state = state
        else:
            # Detach into LSTM: the recurrent path trains its own params but cannot
            # propagate gradient back through to encoder/trunk. Prevents the (early-
            # noisy) LSTM from corrupting the feed-forward representation that the
            # residual path actually relies on. "Memory bolted on."
            lstm_out, new_state = self._run_lstm(feats.detach(), state)
            # Gated residual: out = feats + gate * lstm_out. gate inits to 0 so the
            # policy starts as the feed-forward MLP and the LSTM is only added in
            # if/when it earns positive gradient through the gate.
            out = (feats + self.lstm_gate * lstm_out).squeeze(0)
        logits = self._logits(out, per_item, action_mask)
        if self.decoupled_value:
            value = self._value(self.value_trunk(enc))
        else:
            value = self._value(out)
        return logits, value, new_state

    def get_action_and_value(self, obs, action_mask, state):
        logits, value, new_state = self.step(obs, action_mask, state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, new_state

    def evaluate_sequence(self, obs_seq, mask_seq, action_seq, done_seq, start_state,
                          return_logits=False):
        """Replay a (T, N) rollout segment through the LSTM with stored start
        state, applying done-resets at each step boundary. Returns (logp,
        entropy, values), each (T, N); with return_logits=True also returns
        the masked logits (T, N, A) for KL-style losses.

        done_seq[t] is the done flag AFTER taking action_seq[t]. So when
        processing step t+1, we zero state using done_seq[t]. Step 0 uses
        start_state unchanged (it was already zeroed when stored, if needed).
        """
        T, N = obs_seq.shape[:2]
        # Batch the feature extraction over the whole sequence in one go.
        enc, per_item = self.encoder(obs_seq.reshape(T * N, -1), return_items=True)
        per_item = per_item.view(T, N, per_item.shape[-2], per_item.shape[-1])
        feats = self.input_trunk(enc).view(T, N, -1)
        if self.decoupled_value:
            dvalues = self._value(self.value_trunk(enc)).view(T, N)
        if self.memory_type == "transformer":
            out = self.mem.forward_seq(feats, start_state, done_seq)  # (T,N,H)
            logits = self._logits(out, per_item, mask_seq)
            values = dvalues if self.decoupled_value else self._value(out)
            dist = Categorical(logits=logits)
            if return_logits:
                return dist.log_prob(action_seq), dist.entropy(), values, logits
            return dist.log_prob(action_seq), dist.entropy(), values
        if not self.use_lstm:
            out = feats
            logits = self._logits(out, per_item, mask_seq)
            values = dvalues if self.decoupled_value else self._value(out)
            dist = Categorical(logits=logits)
            if return_logits:
                return dist.log_prob(action_seq), dist.entropy(), values, logits
            return dist.log_prob(action_seq), dist.entropy(), values
        feats_det = feats.detach()
        h, c = start_state
        # Segment packing: the only reason for per-step LSTM calls is zeroing
        # (h, c) at episode boundaries. Run ONE fused cuDNN call for all envs
        # ignoring dones, then recompute just the boundary envs with a second
        # fused call: each env's window is split into segments at its dones,
        # every segment is shifted to t=0 in a zero-padded buffer (exact —
        # the LSTM is causal, a segment depends only on its inputs and its
        # initial state, which is start_state for t0=0 segments and zeros
        # after a done), and the valid rows overwrite the first call's
        # output. No per-timestep python loop on any path. done_seq[T-1]
        # only affects the NEXT window's start state, so it is excluded.
        lstm_out, _ = self._run_lstm(feats_det, (h, c))  # (T, N, H) fused
        mid_done = done_seq[:-1].any(dim=0)         # (N,)
        if bool(mid_done.any()):
            idx = mid_done.nonzero(as_tuple=True)[0]
            dn_cpu = done_seq[:-1, idx].cpu()
            segs_env, segs_t0, segs_t1, segs_first = [], [], [], []
            for j in range(dn_cpu.shape[1]):
                t0 = 0
                for t in dn_cpu[:, j].nonzero(as_tuple=True)[0].tolist():
                    segs_env.append(j); segs_t0.append(t0); segs_t1.append(t)
                    segs_first.append(t0 == 0)
                    t0 = t + 1
                segs_env.append(j); segs_t0.append(t0); segs_t1.append(T - 1)
                segs_first.append(t0 == 0)
            S = len(segs_env)
            dev = feats_det.device
            env_t = idx[torch.tensor(segs_env, device=dev)]
            t0_t = torch.tensor(segs_t0, device=dev)
            t1_t = torch.tensor(segs_t1, device=dev)
            first_t = torch.tensor(segs_first, device=dev)
            ar = torch.arange(T, device=dev).view(T, 1)
            time_idx = (t0_t.view(1, S) + ar).clamp_max(t1_t.view(1, S))  # (T, S)
            env_idx = env_t.view(1, S).expand(T, S)
            seg_in = feats_det[time_idx, env_idx]                          # (T, S, H)
            h0s = h.new_zeros(1, S, h.shape[-1])
            c0s = c.new_zeros(1, S, c.shape[-1])
            if bool(first_t.any()):
                fidx = first_t.nonzero(as_tuple=True)[0]
                h0s[:, fidx] = h[:, env_t[fidx]]
                c0s[:, fidx] = c[:, env_t[fidx]]
            seg_out, _ = self._run_lstm(seg_in, (h0s, c0s))                 # (T, S, H)
            valid = ar < (t1_t - t0_t + 1).view(1, S)                      # (T, S)
            lstm_out = lstm_out.clone()
            lstm_out[time_idx[valid], env_idx[valid]] = seg_out[valid]
        # Gated residual + heads, batched over the whole sequence at once.
        out = feats + self.lstm_gate * lstm_out  # (T, N, H)
        logits = self._logits(out, per_item, mask_seq)                 # (T, N, A)
        values = dvalues if self.decoupled_value else self._value(out)  # (T, N)
        dist = Categorical(logits=logits)
        if return_logits:
            return dist.log_prob(action_seq), dist.entropy(), values, logits
        return dist.log_prob(action_seq), dist.entropy(), values


    def aux_loss(self, obs_seq, action_seq, done_seq):
        """SPR-lite: predict next-step encoder latent from current latent +
        action; cosine loss vs the (detached) actual next latent. Masked so
        predictions never cross an episode boundary. Gradient flows into the
        encoder via the prediction side, shaping a dynamics-aware representation.
        obs_seq (T,N,O), action_seq (T,N), done_seq (T,N)."""
        T, N = obs_seq.shape[:2]
        enc = self.encoder(obs_seq.reshape(T * N, -1)).view(T, N, -1)  # (T,N,E)
        a_emb = self.action_emb(action_seq)                           # (T,N,A_emb)
        pred = self.transition(torch.cat([enc[:-1], a_emb[:-1]], dim=-1))  # predict enc[1:]
        target = enc[1:].detach()
        cos = F.cosine_similarity(pred, target, dim=-1)              # (T-1, N)
        # done_seq[t] True => obs[t+1] is a fresh episode; drop that prediction.
        valid = (~done_seq[:-1]).float()
        denom = valid.sum().clamp_min(1.0)
        return -(cos * valid).sum() / denom



# ─── Entity/token transformer policy (spec V1) ──────────────────────────────

class TokenPolicy(nn.Module):
    """Entity-token transformer policy.

    The 463-d observation already HAS entity structure — it is
    phase[1] | adventurer[13] | equipment[8x9] | bag[15x9] | market[25x9] |
    beast[7] | sim[10] — so tokenising is a re-view of the same bytes, not new
    data, and needs no engine change.

        [CLS] [PLAYER] [STATS] [EQUIP x8] [BAG x15] [BEAST] [MARKET x25]  = 52

    Item tokens (equipped / bag / market) share ONE projection, so an item
    teaches the same weights wherever it sits; an entity-type embedding tells
    them apart and a slot embedding restores position where it matters.

    Action logits are entity-conditioned: bag actions are scored from bag
    tokens, market actions from market tokens, stat upgrades from the stats
    token, and the phase-level actions from [CLS]. That is the "should I equip
    THIS item" inductive bias rather than "should I emit action #17".

    Logits are emitted in the engine's existing 57-slot layout, so masks,
    rollout buffers and PPO are untouched:
        0 explore | 1 attack | 2 flee | 3 equip | 4 drop | 5 buy_item
        6 buy_potion | 7-10 macros (or multi-potion) | 11-25 bag
        26-50 market | 51-56 stat upgrades

    Stateless by design (the spec defers recurrent memory), but it keeps the
    LSTMPolicy interface — init_state/step/evaluate_sequence/_reset_state — so
    collect_rollout and the PPO update need no special-casing. `state` is
    carried through untouched.
    """

    N_EQUIP, N_BAG, N_MARKET, N_STATS = 8, 15, 25, 6
    N_ITEM_FIELDS = 9
    # token index layout
    I_CLS, I_PLAYER, I_STATS, I_EQUIP0 = 0, 1, 2, 3
    I_BAG0 = I_EQUIP0 + N_EQUIP            # 11
    I_BEAST = I_BAG0 + N_BAG               # 26
    I_MARKET0 = I_BEAST + 1                # 27
    N_TOKENS = I_MARKET0 + N_MARKET        # 52
    # entity type ids
    T_CLS, T_PLAYER, T_STATS, T_EQUIP, T_BAG, T_BEAST, T_MARKET = range(7)
    # obs slices
    O_ADV, O_EQUIP, O_BAG, O_MARKET, O_BEAST, O_SIM = 1, 14, 86, 221, 446, 453

    def __init__(self, d_model=None, n_layers=None, n_heads=None, ff_dim=None,
                 act_dim=ACT_DIM, **_ignored):
        super().__init__()
        d_model = TOKEN_D_MODEL if d_model is None else d_model
        n_layers = TOKEN_LAYERS if n_layers is None else n_layers
        n_heads = TOKEN_HEADS if n_heads is None else n_heads
        ff_dim = TOKEN_FF if ff_dim is None else ff_dim
        self.d_model = d_model
        self.act_dim = act_dim
        self.hidden_dim = d_model          # some callers read this

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Per-entity input projections. Items share one so the same weights see
        # every item, wherever it sits.
        self.player_proj = nn.Linear(6 + 6, d_model)   # 6 scalars + phase one-hot
        self.stats_proj = nn.Linear(7, d_model)
        self.item_proj = nn.Linear(self.N_ITEM_FIELDS, d_model)
        self.beast_proj = nn.Linear(7 + 10, d_model)   # beast fields + sim stats

        self.type_emb = nn.Embedding(7, d_model)
        self.slot_emb = nn.Embedding(self.N_TOKENS, d_model)
        # Bag/market ordering is incidental, so their slot embedding is damped
        # rather than removed (kept learnable, just scaled down at init).
        self.register_buffer("slot_scale", self._slot_scale(), persistent=False)
        self.register_buffer("type_ids", self._type_ids(), persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(d_model)

        # ── heads ──
        # 11 phase-level actions (explore/attack/flee/enter-equip/enter-drop/
        # enter-buy/potion + the four macro-or-multipotion slots) from [CLS].
        self.global_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            _init_linear(nn.Linear(d_model, EX_NUM_BASE_ACTIONS), gain=0.01))
        self.bag_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            _init_linear(nn.Linear(d_model, 1), gain=0.01))
        self.market_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            _init_linear(nn.Linear(d_model, 1), gain=0.01))
        self.upgrade_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            _init_linear(nn.Linear(d_model, self.N_STATS), gain=0.01))
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            _init_linear(nn.Linear(d_model, N_QUANTILES if QUANTILE_VALUE else 1),
                         gain=1.0))
        self.quantile_value = QUANTILE_VALUE
        # Fields other code paths probe for; harmless here.
        self.decoupled_value = False
        self.memory_type = "none"
        self.spr_aux = False
        self.use_lstm = False

    # ── token metadata ──
    def _type_ids(self):
        t = torch.empty(self.N_TOKENS, dtype=torch.long)
        t[self.I_CLS] = self.T_CLS
        t[self.I_PLAYER] = self.T_PLAYER
        t[self.I_STATS] = self.T_STATS
        t[self.I_EQUIP0:self.I_EQUIP0 + self.N_EQUIP] = self.T_EQUIP
        t[self.I_BAG0:self.I_BAG0 + self.N_BAG] = self.T_BAG
        t[self.I_BEAST] = self.T_BEAST
        t[self.I_MARKET0:] = self.T_MARKET
        return t

    def _slot_scale(self):
        s = torch.ones(self.N_TOKENS, 1)
        s[self.I_BAG0:self.I_BAG0 + self.N_BAG] = 0.1
        s[self.I_MARKET0:] = 0.1
        return s

    def tokenize(self, obs):
        """(B, 463) -> (B, 52, d_model). Pure re-view of the flat obs."""
        B = obs.shape[0]
        phase = obs[:, 0].long().clamp(0, 5)
        adv = obs[:, self.O_ADV:self.O_ADV + 13]
        player = torch.cat([adv[:, :6], F.one_hot(phase, 6).to(obs.dtype)], dim=-1)
        stats = adv[:, 6:13]
        equip = obs[:, self.O_EQUIP:self.O_BAG].reshape(B, self.N_EQUIP, 9)
        bag = obs[:, self.O_BAG:self.O_MARKET].reshape(B, self.N_BAG, 9)
        market = obs[:, self.O_MARKET:self.O_BEAST].reshape(B, self.N_MARKET, 9)
        beast = torch.cat([obs[:, self.O_BEAST:self.O_SIM],
                           obs[:, self.O_SIM:self.O_SIM + 10]], dim=-1)

        items = torch.cat([equip, bag, market], dim=1)          # (B, 48, 9)
        toks = torch.cat([
            self.cls_token.expand(B, 1, -1),
            self.player_proj(player).unsqueeze(1),
            self.stats_proj(stats).unsqueeze(1),
            self.item_proj(items[:, :self.N_EQUIP]),
            self.item_proj(items[:, self.N_EQUIP:self.N_EQUIP + self.N_BAG]),
            self.beast_proj(beast).unsqueeze(1),
            self.item_proj(items[:, self.N_EQUIP + self.N_BAG:]),
        ], dim=1)
        return toks + self.type_emb(self.type_ids) \
                    + self.slot_emb.weight * self.slot_scale

    def _encode(self, obs):
        h = self.ln_out(self.encoder(self.tokenize(obs)))
        return h

    def _logits_value(self, obs, action_mask):
        flat = obs.reshape(-1, OBS_DIM)
        h = self._encode(flat)
        cls = h[:, self.I_CLS]
        bag = h[:, self.I_BAG0:self.I_BAG0 + self.N_BAG]
        mkt = h[:, self.I_MARKET0:]
        stats = h[:, self.I_STATS]

        def pair(tokens):
            return torch.cat([cls.unsqueeze(1).expand_as(tokens), tokens], dim=-1)

        logits = torch.cat([
            self.global_head(cls),                                   # 0-10
            self.bag_head(pair(bag)).squeeze(-1),                    # 11-25
            self.market_head(pair(mkt)).squeeze(-1),                 # 26-50
            self.upgrade_head(torch.cat([cls, stats], dim=-1)),      # 51-56
        ], dim=-1)

        v = self.value_head(cls)
        if self.quantile_value:
            self._vq = v.reshape(*obs.shape[:-1], -1)
            value = v.mean(-1)
        else:
            value = v.squeeze(-1)
        logits = logits.reshape(*obs.shape[:-1], self.act_dim)
        value = value.reshape(*obs.shape[:-1])
        return logits.masked_fill(~action_mask,
                                  mask_fill_value(logits.dtype)), value

    # ── LSTMPolicy-compatible interface (stateless) ──
    def init_state(self, n_envs: int, device):
        z = torch.zeros(1, n_envs, 1, device=device)
        return (z, z.clone())

    @staticmethod
    def _reset_state(state, done):
        return state          # nothing to reset

    def step(self, obs, action_mask, state):
        logits, value = self._logits_value(obs, action_mask)
        return logits, value, state

    def get_action_and_value(self, obs, action_mask, state):
        logits, value, new_state = self.step(obs, action_mask, state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, new_state

    def evaluate_sequence(self, obs_seq, mask_seq, action_seq, done_seq,
                          start_state, return_logits=False):
        logits, values = self._logits_value(obs_seq, mask_seq)
        dist = Categorical(logits=logits)
        if return_logits:
            return dist.log_prob(action_seq), dist.entropy(), values, logits
        return dist.log_prob(action_seq), dist.entropy(), values


# Back-compat alias: dashboard side-car and main() refer to `MLPPolicy`.
MLPPolicy = LSTMPolicy


# -- Rollout collection -------------------------------------------------------


def shop_gate_mask(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build-band shopping gate (survivorship-clean finding, 2026-07-18):
    top humans buy 1.67 items/game at L6-11 vs the policy's 0.51 — the ONE
    behavioral gap; it compounds into the entire crossing-rate deficit.
    At market phase, level 6-11, gold-rich, with an affordable item on offer
    (buy_menu maskable), block EXPLORE/MACRO_EXPLORE: convert cash to gear
    (or potions/equip) before walking past the shop. Constraint, not price —
    priced incentives are 0-for-15 against this policy."""
    xp = obs[:, 3] * 32767.0
    gate = ((obs[:, 0] == 1) & (xp >= 36.0) & (xp < 144.0)
            & (obs[:, 4] * 511.0 >= 24.0) & mask[:, 5])
    if gate.any():
        mask[gate, 0] = False
        mask[gate, 10] = False
    return mask


class DoorstepPool:
    """Blob pool for doorstep-reset continuation training.

    Sampled blobs get a fresh rng (uint64 at byte offset 0) so clones of the
    same state explore different futures. Blobs are stored injection-ready
    (done=0, episode_length=0; the pool builder is not in this repo)."""

    def __init__(self, path: str, seed: int = 0):
        data = np.load(path)
        if "blobs" in data:
            self.blobs = data["blobs"]
            print(f"blob pool: {self.blobs.shape[0]} states ({path})")
        else:
            self.blobs = np.concatenate([data["human"], data["self"]], axis=0)
            print(f"doorstep pool: {data['human'].shape[0]} human + "
                  f"{data['self'].shape[0]} self states")
        self.rng = np.random.default_rng(seed ^ 0x5EED)

    def sample(self) -> np.ndarray:
        return self.sample_batch(1)[0]

    def sample_batch(self, k: int) -> np.ndarray:
        idx = self.rng.integers(0, len(self.blobs), size=k)
        out = self.blobs[idx].copy()
        rngs = (self.rng.integers(1, 2**63, size=k, dtype=np.int64) | 1).astype("<u8")
        out[:, 0:8] = rngs.view(np.uint8).reshape(k, 8)
        return out


@torch.no_grad()
def collect_rollout(env, policy, device, lstm_state, prior=None, prior_state=None,
                    snap_count=0, doorstep_pool=None, doorstep_prob=1.0,
                    inject_pending=None, shop_gate=False,
                    fight_drill=False, drill_hp_cost=0.0, explore_eps=0.0,
                    groups=None, world_pool=None):
    """Collect ROLLOUT_STEPS of experience, threading LSTM state across steps.

    `lstm_state` is the (h, c) entering this rollout (already zeroed for envs in
    a fresh episode). Returns the rollout buffers, the start_state (saved here
    so PPO can replay sequences with the same input), and the end_state to
    thread into the next rollout.

    If `prior` is given (frozen BC policy for the KL anchor), its masked logits
    are recorded at every step (its own LSTM state threaded via `prior_state`)
    and returned as an extra (T, N, A) buffer + its end state.

    If `snap_count` > 0, up to that many decision states (market/combat
    phases) are snapshotted: (env blob, h, c, mask) tuples for expert-
    iteration search. Snapshots capture the state BEFORE the step's action.
    """
    n = ROLLOUT_STEPS
    N = env.num_envs
    snapshots = []

    obs_buf = np.empty((n, N, OBS_DIM), dtype=np.float32)
    mask_buf = np.empty((n, N, ACT_DIM), dtype=bool)
    act_buf = np.empty((n, N), dtype=np.int64)
    logp_buf = np.empty((n, N), dtype=np.float32)
    val_buf = np.empty((n, N), dtype=np.float32)
    rew_buf = np.empty((n, N), dtype=np.float32)
    done_buf = np.empty((n, N), dtype=bool)
    world_buf = np.zeros((n, N), dtype=np.int64)
    prior_logits_buf = np.empty((n, N, ACT_DIM), dtype=np.float32) if prior is not None else None
    death_xps = []

    # Save the state entering this rollout so PPO can replay the sequence with
    # the same inputs the rollout used.
    start_state = (lstm_state[0].clone(), lstm_state[1].clone())
    h, c = lstm_state

    for t in range(n):
        obs = env.obs.copy()
        mask = env.action_mask.astype(bool)
        if shop_gate:
            mask = shop_gate_mask(obs, mask)

        if snap_count and len(snapshots) < snap_count:
            cand = np.flatnonzero(np.isin(obs[:, 0].astype(np.int32), (1, 4)))
            if len(cand):
                i = int(cand[np.random.randint(len(cand))])
                snapshots.append({
                    "blob": env._engine.save_state(i),
                    "obs": obs[i].copy(),
                    "mask": mask[i].copy(),
                    "h": h[:, i].detach().clone(),
                    "c": c[:, i].detach().clone(),
                })

        obs_buf[t] = obs
        obs_t = torch.from_numpy(obs).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        if explore_eps > 0.0:
            # VALUE-REPAIR mode: with prob eps take a random valid action so V
            # learns the off-policy branches search steers into. log_prob is the
            # policy's logp OF THE TAKEN action (correct importance base).
            logits, value, (h, c) = policy.step(obs_t, mask_t, (h, c))
            dist = Categorical(logits=logits)
            action = dist.sample()
            forced = torch.rand(action.shape[0], device=device) < explore_eps
            if bool(forced.any()):
                rnd = torch.rand_like(logits).masked_fill(~mask_t, -1.0)
                action = torch.where(forced, rnd.argmax(dim=-1), action)
            log_prob = dist.log_prob(action)
        else:
            action, log_prob, value, (h, c) = policy.get_action_and_value(obs_t, mask_t, (h, c))
        if prior is not None:
            p_logits, _, prior_state = prior.step(obs_t, mask_t, prior_state)
            prior_logits_buf[t] = p_logits.cpu().numpy()

        actions_np = action.cpu().numpy().astype(np.int32)
        if groups is not None:
            # Stamp the world BEFORE the step: on_reset() below reassigns
            # world_id for envs that just died, and this step still belongs to
            # the episode that was being played.
            world_buf[t] = groups.world_id
        env.step(actions_np)
        # Read done BEFORE anything calls load_state: the engine's
        # dmfast_exact_load_state clears terminated/truncated, so re-reading
        # the flags afterwards silently loses every death that got its world
        # overwritten (with --group-size 4 that is 3 of every 4 envs).
        done = env.terminated.astype(bool) | env.truncated.astype(bool)
        if groups is not None:
            groups.on_reset(env, done)
        if world_pool is not None:
            world_pool.on_reset(env, done)

        mask_buf[t] = mask
        act_buf[t] = actions_np
        logp_buf[t] = log_prob.cpu().numpy()
        val_buf[t] = value.cpu().numpy()
        rew = env.reward.copy()
        if fight_drill:
            # Fight-drill: each combat is one episode. Resolution (phase left
            # combat, or env done) is a synthetic terminal so values never
            # bridge unrelated fights. HP spent is priced every step —
            # drill episodes end at resolution, so without this the policy
            # is indifferent to damage taken (execution quality invisible).
            hp_drop = np.maximum(0.0, obs[:, 1] - env.obs[:, 1]) * 1023.0
            rew = rew - drill_hp_cost * hp_drop
            resolved = (env.obs[:, 0] != 4) | done
            done = resolved
        done_buf[t] = done

        died = env.terminated.astype(bool) & ~env.truncated.astype(bool)
        for i in np.flatnonzero(died):
            death_xps.append(float(env.last_episode_info[i, Info.XP]))

        if doorstep_pool is not None:
            # The state-diff reward kernel's prev buffers are inconsistent for
            # one transition after an injection (obs/mask refresh immediately,
            # info repacks on the next engine step) — zero that one reward.
            rew[inject_pending] = 0.0
            inject_pending[:] = False
            # Redirect a fraction of episode starts to doorstep states: the
            # env auto-reset gave a fresh start; overwrite it with a pool blob.
            # In fight-drill mode `done` includes synthetic combat-resolution
            # terminals, so every resolved fight is replaced by a fresh one.
            targets = np.flatnonzero(done)
            if doorstep_prob < 1.0 and len(targets):
                targets = targets[np.random.random(len(targets)) < doorstep_prob]
            if len(targets):
                batch = doorstep_pool.sample_batch(len(targets))
                for j, i in enumerate(targets):
                    env._engine.load_state(int(i), batch[j])
                inject_pending[targets] = True
        rew_buf[t] = rew

        # Reset LSTM state for envs whose episode just ended (env auto-reset,
        # so the NEXT obs is a fresh episode and the LSTM should start clean).
        if done.any():
            keep = torch.from_numpy(~done).to(device=device, dtype=h.dtype).view(1, -1, 1)
            h = h * keep
            c = c * keep
            if prior is not None:
                prior_state = (prior_state[0] * keep, prior_state[1] * keep)

    # Bootstrap value from the post-rollout state.
    obs_t = torch.from_numpy(env.obs.copy()).to(device)
    mask_t = torch.from_numpy(env.action_mask.astype(bool)).to(device)
    _, _, next_value, _ = policy.get_action_and_value(obs_t, mask_t, (h, c))
    end_state = (h, c)

    return (
        obs_buf,
        mask_buf,
        act_buf,
        logp_buf,
        val_buf,
        rew_buf,
        done_buf,
        world_buf,
        next_value.cpu().numpy(),
        death_xps,
        start_state,
        end_state,
        prior_logits_buf,
        prior_state,
        snapshots,
    )


def current_arch(use_lstm=True, transformer=False):
    """The architecture record written into every checkpoint.

    These fields fully determine the tensor set, so a reader can rebuild the
    exact module that produced the weights instead of guessing. The family is
    part of the name -- dm_mlp_v1, dm_lstm_v1, dm_transformer_v1 -- so there is
    no separate flag a reader has to interpret.
    """
    common = {
        "obs_dim": int(OBS_DIM),
        "act_dim": int(ACT_DIM),
        "decoupled_value": bool(DECOUPLED_VALUE),
        "quantile_value": bool(QUANTILE_VALUE),
        "equivariant_head": bool(EQUIVARIANT_HEAD),
        "spr_aux": bool(SPR_AUX),
    }
    if transformer:
        return {
            "architecture": "dm_transformer_v1",
            "d_model": int(TOKEN_D_MODEL),
            "n_layers": int(TOKEN_LAYERS),
            "n_heads": int(TOKEN_HEADS),
            "ff_dim": int(TOKEN_FF),
            **common,
        }
    return {
        "architecture": "dm_lstm_v1" if use_lstm else "dm_mlp_v1",
        "embed_dim": int(EMBED_DIM),
        "hidden_dim": int(HIDDEN_DIM),
        "num_trunk_blocks": int(TRUNK_NUM_BLOCKS),
        "memory_type": MEMORY_TYPE,
        **common,
    }


# Training hyperparameters and reward weights are recorded alongside the
# weights. Nothing reads them back at eval time -- a policy is scored purely on
# how it plays -- but a checkpoint that cannot say how it was produced is not
# reproducible.
TRAINING_PARAM_GLOBALS = (
    "NUM_ENVS", "MAX_STEPS", "ROLLOUT_STEPS", "LR", "GAMMA", "GAE_LAMBDA",
    "CLIP_EPS", "ENTROPY_COEF_START", "ENTROPY_COEF_END", "MAX_GRAD_NORM",
    "TOTAL_ENV_STEPS",
)


def current_training(seed=None):
    """The reward weights and training hyperparameters this run used."""
    params = {name.lower(): globals()[name] for name in TRAINING_PARAM_GLOBALS
              if name in globals()}
    if seed is not None:
        params["seed"] = int(seed)
    return {
        "reward": {k: float(v) for k, v in REWARD_CONFIG.items()
                   if isinstance(v, (int, float))},
        "params": params,
    }


# -- Fresh-reset eval ---------------------------------------------------------


# The eval set: a fixed bank of starting worlds every policy is scored on.
# reset(seed) hands env i the world seed `seed + i * 1315423911` (see
# dmfast_exact_reset_all), so worlds 0..EVAL_WORLDS-1 are identical across
# runs, machines and checkpoints.
EVAL_WORLDS = 1000
EVAL_SEED = 7

# The full eval: a much larger bank than the trainer's per-checkpoint one,
# for when you want a number you can quote. Wider than any batch, so
# eval_policy runs it in chunks -- see WORLD_SEED_STRIDE.
BANK_WORLDS = 65_536 // 4
# The score is the mean over THREE banks, not one. reset() is a pure function
# of its seed, so a single bank is one draw, and a lucky seed is worth real
# points. Override with DM_EVAL_SEEDS to score on worlds you have not tuned
# against -- the defaults below are the ones every run here reports, so a
# policy selected on them is selected on a test set it has already seen.
DEFAULT_BANK_SEEDS = (3930, 7717, 20477)
BANK_SEEDS = tuple(
    int(x) for x in os.environ.get(
        "DM_EVAL_SEEDS", ",".join(map(str, DEFAULT_BANK_SEEDS))).split(",")
)
BANK_SEED = BANK_SEEDS[0]   # the single-bank default, for --eval-only
# The full eval gets its own env at a pinned width, not the training batch:
# batch width changes the action-sampling stream, so two runs are only
# comparable at the same width. 65_536 is exactly 32 chunks of 2_048.
BANK_BATCH = 2_048

# ex_reset_env hands env i the world seed `seed + i * WORLD_SEED_STRIDE`
# (dmfast_exact.c). The stride is affine in the env index, so worlds
# c*B .. c*B+B-1 of a bank are exactly the worlds a B-wide env produces when
# reset with `seed + c*B*WORLD_SEED_STRIDE`. That is what lets a 2048-wide env
# score a 100k-world bank without allocating 100k envs.
WORLD_SEED_STRIDE = 1315423911


# -- Episode stats ------------------------------------------------------------

# Read straight off the final observation (layout: docs/environment.md), so
# these cost nothing in the engine and need no rebuild.
ADV_STATS = slice(7, 13)      # str dex vit int wis cha, each /31
ADV_LUCK = 13                 # /100
EQUIP_BLOCK = slice(14, 86)   # 8 slots x 9 item fields
ITEM_FIELDS = 9
ARMOR_SLOTS = slice(1, 6)     # chest head waist foot hand (0 is weapon)
I_ID, I_TYPE, I_GREATNESS = 0, 2, 8
ARMOR_MATERIALS = {"cloth": 1, "hide": 2, "metal": 3}
STAT_NAMES = ("str", "dex", "vit", "int", "wis", "cha", "luck")
GEAR_KEYS = ("gear_lvl15", "gear_lvl20", "full_cloth", "full_hide", "full_metal")


def episode_stats(terminal_obs):
    """Per-episode gear milestones and final stats, one row per world.

    Gear keys are booleans (mean = fraction of adventurers); stat keys are the
    denormalised values at death, base + item specials as the obs packs them.
    """
    items = terminal_obs[:, EQUIP_BLOCK].reshape(len(terminal_obs), -1, ITEM_FIELDS)
    equipped = items[:, :, I_ID] != 0
    # greatness is packed as (g - 1) / 20; rint because float32 round-trips.
    greatness = np.rint(items[:, :, I_GREATNESS] * 20.0) + 1.0
    armor, armor_on = items[:, ARMOR_SLOTS, :], equipped[:, ARMOR_SLOTS]

    out = {
        "gear_lvl15": (equipped & (greatness >= 15)).any(axis=1),
        "gear_lvl20": (equipped & (greatness >= 20)).any(axis=1),
    }
    for name, code in ARMOR_MATERIALS.items():
        # "full" = every armour slot filled AND all of one material.
        out[f"full_{name}"] = (armor_on & (armor[:, :, I_TYPE] == code)).all(axis=1)
    stats = terminal_obs[:, ADV_STATS] * 31.0
    for i, name in enumerate(STAT_NAMES[:-1]):
        out[name] = stats[:, i]
    out["luck"] = terminal_obs[:, ADV_LUCK] * 100.0
    return out


def merge_stats(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def print_episode_stats(stats):
    for key in GEAR_KEYS:
        print(f"eval_{key}:{'':<{max(1, 14 - len(key))}}{stats[key].mean() * 100:.1f}%")
    print("eval_final_stats:  " +
          "  ".join(f"{n} {stats[n].mean():.1f}" for n in STAT_NAMES))


@torch.no_grad()
def _eval_chunk(env, policy, device, n, seed, mask_transform=None):
    """Score `n` worlds from one reset. Requires n <= env.num_envs.

    The batch is always env.num_envs wide; only the first `n` lanes are
    scored. Returns (xps, n_truncated).
    """
    env.reset(seed=seed)
    core = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    h, c = core.init_state(env.num_envs, device)

    xps = np.full(n, np.nan, dtype=np.float64)
    truncated = np.zeros(n, dtype=bool)
    term_obs = np.zeros((n, env.obs.shape[1]), dtype=np.float32)
    pending = n
    # +1 because an episode ends ON the step that hits the cap.
    for _ in range(MAX_STEPS + 1):
        if pending == 0:
            break
        obs_np = env.obs.copy()
        mask_np = env.action_mask.astype(bool)
        if mask_transform is not None:
            mask_np = mask_transform(obs_np, mask_np)
        obs_t = torch.from_numpy(obs_np).to(device)
        mask_t = torch.from_numpy(mask_np).to(device)
        action, _, _, (h, c) = policy.get_action_and_value(obs_t, mask_t, (h, c))
        env.step(action.cpu().numpy().astype(np.int32))

        term = env.terminated.astype(bool)
        done = term | env.truncated.astype(bool)
        # np.isnan picks out worlds still on their first episode; a world
        # that already has a score never counts again, however often it
        # replays.
        fresh = np.flatnonzero(done[:n] & np.isnan(xps))
        if fresh.size:
            xps[fresh] = env.last_episode_info[fresh, Info.XP]
            truncated[fresh] = ~term[fresh]
            term_obs[fresh] = env.last_terminal_obs[fresh]
            pending -= fresh.size
        if done.any():
            keep = torch.from_numpy(~done).to(device=device, dtype=h.dtype).view(1, -1, 1)
            h = h * keep
            c = c * keep

    if pending:  # unreachable: MAX_STEPS truncation ends every episode
        raise RuntimeError(f"{pending}/{n} eval worlds unfinished after "
                           f"{MAX_STEPS} steps")
    return xps, int(truncated.sum()), episode_stats(term_obs)


@torch.no_grad()
def eval_policy(env, policy, device, worlds=EVAL_WORLDS, seed=EVAL_SEED,
                mask_transform=None, progress=False):
    """Score the policy on a fixed bank of `worlds` starting worlds.

    Two things are pinned, and both are needed for a score that depends only
    on the policy:

    * the worlds -- reset(seed) hands env i the world seed
      `seed + i * WORLD_SEED_STRIDE`, so worlds 0..worlds-1 are identical
      across runs, machines and checkpoints;
    * the action-sampling stream -- the policy samples from a Categorical, so
      without a fixed torch seed the same checkpoint scores differently
      depending on when eval runs. The previous RNG state is restored on the
      way out, leaving a caller's stream untouched.

    `worlds` may exceed env.num_envs. The bank is then scored in chunks of
    env.num_envs, each chunk reset with `seed + done * WORLD_SEED_STRIDE` so
    it lands on exactly the worlds a single env that wide would have produced.
    Each chunk reseeds torch to `seed + chunk_index`, so a chunk's score does
    not depend on the chunks before it.

    The batch WIDTH is part of the protocol, not an implementation detail: the
    policy samples one action per env per step from a shared stream, so the
    same bank scored with a different env.num_envs draws different actions and
    returns a different number. Compare numbers only at equal width.

    Each world counts its FIRST episode only. On death the engine draws the
    next world from a batch-wide counter (`ex_reset_env(b, idx,
    ++b->seed_counter)`), so which world an env sees second depends on the
    order every env in the batch happened to die -- policy-dependent, and
    therefore not a fixed test set. First episodes are the part that is pinned:
    every policy meets the same `worlds` openings, in the same order.

    Runs until every world's first episode has ended, by death or by
    truncation at MAX_STEPS, so the result always has exactly `worlds` entries.

    LSTM state is threaded and zeroed on done, matching deployment semantics
    (every game starts clean). mask_transform is an optional client-side mask
    edit (e.g. shop_gate_mask) so eval sees the same policy+mask system that
    deployment would.

    Returns (xps, n_truncated).
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    parts, stat_parts, n_truncated, done, chunk = [], [], 0, 0, 0
    n_chunks = (worlds + env.num_envs - 1) // env.num_envs
    try:
        while done < worlds:
            n = min(worlds - done, env.num_envs)
            chunk_seed = (seed + done * WORLD_SEED_STRIDE) % 2**64
            torch.manual_seed(seed + chunk)
            xps, trunc, stats = _eval_chunk(env, policy, device, n, chunk_seed,
                                            mask_transform)
            parts.append(xps)
            stat_parts.append(stats)
            n_truncated += trunc
            done += n
            chunk += 1
            if progress and n_chunks > 1:
                print(f"  eval chunk {chunk}/{n_chunks}  {done}/{worlds} worlds  "
                      f"running mean {np.concatenate(parts).mean():.1f}", flush=True)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    return np.concatenate(parts), n_truncated, merge_stats(stat_parts)


def bank_score(policy, device, seeds=None, worlds=BANK_WORLDS,
                      mask_transform=None, progress=False, env_factory=None):
    """Score a policy over every eval bank and average the banks.

    Each bank gets a clean env at the pinned width: batch width is part of the
    protocol, and the training env is both the wrong width and liable to carry
    curriculum knobs. Returns (per_seed, mean_xp, total_truncated) where
    per_seed is a list of (seed, xps, truncated).
    """
    seeds = tuple(BANK_SEEDS if seeds is None else seeds)
    make_env = env_factory or (lambda seed: GameEnv(
        num_envs=BANK_BATCH, seed=seed, max_steps=MAX_STEPS,
        reward_config=REWARD_CONFIG))
    per_seed, truncated = [], 0
    for i, seed in enumerate(seeds, 1):
        if progress:
            print(f"bank {i}/{len(seeds)}: {worlds} worlds at seed {seed}", flush=True)
        env = make_env(seed)
        try:
            xps, trunc, stats = eval_policy(env, policy, device, worlds=worlds,
                                            seed=seed, mask_transform=mask_transform,
                                            progress=progress)
        finally:
            env.close()
        per_seed.append((seed, xps, trunc, stats))
        truncated += trunc
    # Mean of the bank means. Banks are equal-sized, so this is also the mean
    # over every world; stated this way because the per-bank numbers are what
    # get published alongside it.
    mean_xp = float(np.mean([xps.mean() for _, xps, _, _ in per_seed]))
    return per_seed, mean_xp, truncated


def print_bank_result(per_seed, mean_xp, truncated, batch):
    for seed, xps, trunc, _ in per_seed:
        print(f"  seed {seed:<12} avg {xps.mean():8.1f}   max {xps.max():7.1f}   "
              f"median {np.median(xps):7.1f}   truncated {trunc}")
    print(f"eval_avg_xp:       {mean_xp:.1f}")
    print(f"eval_banks:        {len(per_seed)} x {len(per_seed[0][1])} worlds")
    print(f"eval_seeds:        {','.join(str(s) for s, _, _, _ in per_seed)}")
    print(f"eval_batch:        {batch}")
    print(f"eval_truncated:    {truncated}")
    print_episode_stats(merge_stats([st for _, _, _, st in per_seed]))


# -- GAE ----------------------------------------------------------------------


class GroupWorlds:
    """Common random numbers for training: envs in the same group replay the
    SAME starting world.

    Why this is worth doing (measured on the 32B policy, 96 worlds x 16
    repeats): 43% of final-xp variance is fixed by the starting state, and the
    critic explains only ~3% of return variance at episode start. That is not a
    critic failure — roughly half the starting-world effect is the env's rng
    seed, which lives in DMFastExactEnv but never enters the 463-d observation,
    so V *cannot* predict it. Cloned rollouts share it exactly, which is the
    one regime where a paired baseline beats a learned one.

    Note the engine's randomness is a STREAM (ex_step_explore pulls ~15 draws
    from e->rng per call), so group members desynchronize the moment they act
    differently. This shares the starting world only — the 43% — not the whole
    trajectory.

    Scheme: the first env of each group is the leader. When the leader dies its
    fresh start is captured as the group's world; when any other member dies it
    is reloaded into that world. Members therefore replay the leader's world
    until the leader itself dies and draws a new one.
    """

    def __init__(self, num_envs: int, group_size: int):
        if group_size < 2 or num_envs % group_size:
            raise ValueError(
                f"--group-size {group_size} must be >=2 and divide num_envs "
                f"({num_envs})")
        self.group_size = group_size
        self.num_groups = num_envs // group_size
        self.leader = np.arange(self.num_groups) * group_size
        self.blob = [None] * self.num_groups
        self.env_group = np.arange(num_envs) // group_size
        # Which world each env is currently playing. Members die and reload at
        # DIFFERENT times, so at a given rollout timestep they sit at different
        # points of different episodes — the group baseline has to be matched
        # by world id, never by timestep.
        self.world_id = np.arange(num_envs) // group_size
        self._next_world = self.num_groups

    @staticmethod
    def _eng(env):
        """The raw BatchEnv. GameEnv wraps it and does not forward the
        state-snapshot API (same unwrap the doorstep-reset path uses)."""
        return getattr(env, "_engine", env)

    def prime(self, env):
        """Seed every group from its leader's current (post-reset) state."""
        eng = self._eng(env)
        for g in range(self.num_groups):
            self.blob[g] = eng.save_state(int(self.leader[g]))
            for j in range(1, self.group_size):
                eng.load_state(int(self.leader[g] + j), self.blob[g])

    def on_reset(self, env, done: np.ndarray) -> None:
        """Called right after env.step(): auto_reset has already given each
        dead env a fresh world, so leaders define and members inherit."""
        if not done.any():
            return
        eng = self._eng(env)
        idx = np.flatnonzero(done)
        # Leaders first, so a member resetting on the same step inherits the
        # NEW world rather than the retired one.
        for i in idx:
            g = self.env_group[i]
            if i == self.leader[g]:
                self.blob[g] = eng.save_state(int(i))
                self.world_id[i] = self._next_world
                self._next_world += 1
        for i in idx:
            g = self.env_group[i]
            if i != self.leader[g] and self.blob[g] is not None:
                eng.load_state(int(i), self.blob[g])
                self.world_id[i] = self.world_id[self.leader[g]]


class BuyBoostCurriculum:
    """Reward curriculum: multiply the item-buying rewards for the first
    `frac` of training, then snap back to the configured values.

    The idea is that shopping is a long-horizon, low-frequency behaviour that
    the agent has to discover before it can pay off — BUY is only ~8% of steps
    — so an early bonus buys exploration of the market, and the normal reward
    then decides what is actually worth buying.

    Only ITEM buying is boosted. buy_potion is deliberately left alone: potions
    are already the most-used action by a wide margin (~45 per life at the 32B
    checkpoint) and need no encouragement.

    Two risks worth stating, since neither is hypothetical:
      - This is reward geometry, which is 0-for-15+ in the experiment log. The
        difference here is that it is SCHEDULED rather than a permanent
        reshaping, but that is a difference in kind, not proven to matter.
      - The revert is a step change. The critic has been fitting returns under
        the boosted reward and is suddenly wrong at the boundary, which shows
        up as a value-loss spike and a transient policy dip. --buy-boost-ramp
        anneals the multiplier down instead if that turns out to bite.
    """

    KEYS = ("buy_item", "buy_t1_item", "buy_weapon_t1",
            "buy_quality_delta", "buy_asset_bank_delta")

    def __init__(self, env, mult: float, frac: float, ramp: bool = False):
        eng = getattr(env, "_engine", env)
        self.model = eng._reward_model
        self.mult, self.frac, self.ramp = mult, frac, ramp
        self.base = {k: float(self.model.params.get(k, 0.0)) for k in self.KEYS}
        self._last = None
        self.set_progress(0.0)

    def _factor(self, progress: float) -> float:
        if progress >= self.frac:
            return 1.0
        if not self.ramp:
            return self.mult
        # Linear ramp from mult down to 1.0 across the boosted window.
        return 1.0 + (self.mult - 1.0) * (1.0 - progress / max(self.frac, 1e-9))

    def set_progress(self, progress: float) -> None:
        f = self._factor(float(progress))
        if self._last is not None and abs(f - self._last) < 1e-6:
            return
        for k, v in self.base.items():
            self.model.set_param(k, v * f)
        if self._last is None:
            print(f"BUY BOOST: item-buy rewards x{f:.2f} for the first "
                  f"{100*self.frac:.0f}% of training "
                  f"({'ramped' if self.ramp else 'then hard revert'}); "
                  f"boosted keys {', '.join(self.KEYS)}", flush=True)
        elif f == 1.0:
            print(f"  >> buy boost OFF (progress {progress:.2f}) — "
                  f"reward reverted to configured values", flush=True)
        self._last = f


class WorldPool:
    """Stochasticity curriculum: draw episode starts from a small FIXED pool of
    worlds early, then anneal toward fresh worlds.

    The bet is that repeated exposure to the same starting worlds makes the
    early gradient cheaper — the agent is not paying to re-estimate a new world
    every episode — and that the competence transfers once randomisation opens
    up.

    Two things bound how much this can do, both measured on this engine:

    1. It fixes the START only. The rng is a stream (ex_step_explore pulls ~15
       draws per call), so two episodes in the same world diverge as soon as
       the policy acts differently. A "fixed world" is not a deterministic
       episode; it is a fixed opening. That start is worth 43% of final-xp
       variance, which is the ceiling here.
    2. Unlike group cloning, this is BIASED. Training on K worlds and evaluating
       on all of them is a train/test mismatch, and a small pool can be
       memorised — the agent can learn world-specific lines that do not
       transfer. The anneal is what is supposed to wash that out, and whether
       it does is exactly what the experiment has to show.

    Composes with --group-size: the pool decides WHICH world an episode plays,
    grouping decides how many episodes share it.
    """

    def __init__(self, env, size: int, p_start: float, p_end: float):
        eng = getattr(env, "_engine", env)
        self.size = min(size, env.num_envs)
        if size > env.num_envs:
            print(f"  world pool capped at num_envs ({env.num_envs})", flush=True)
        # The env has num_envs distinct worlds right after reset; take the
        # first `size` of them as the fixed pool.
        self.blobs = [eng.save_state(i) for i in range(self.size)]
        self.p_start, self.p_end = p_start, p_end
        self.p = p_start
        self.rng = np.random.default_rng(12345)

    def set_progress(self, progress: float) -> None:
        """progress in [0, 1] over the run; p anneals p_start -> p_end."""
        self.p = self.p_start + (self.p_end - self.p_start) * float(progress)

    def on_reset(self, env, done: np.ndarray) -> None:
        """Overwrite a fraction of fresh starts with pool worlds. MUST be
        called after `done` has been read — load_state clears the engine's
        terminated/truncated flags."""
        if self.p <= 0.0 or not done.any():
            return
        eng = getattr(env, "_engine", env)
        idx = np.flatnonzero(done)
        take = idx[self.rng.random(len(idx)) < self.p]
        if not len(take):
            return
        picks = self.rng.integers(0, self.size, size=len(take))
        for i, k in zip(take, picks):
            eng.load_state(int(i), self.blobs[int(k)])


def group_center_advantages(advantages, dones, world_buf):
    """Remove the shared starting-world offset from each episode's advantages.

    Episodes are matched BY WORLD, not by timestep: group members desync the
    moment they act differently and reset at different times, so at a given
    rollout step they are at different points of different episodes. Centring
    column-wise would subtract an unrelated episode's advantage.

    For each episode that completes inside the window we take its mean
    advantage, then subtract the LEAVE-ONE-OUT mean over the other episodes
    that played the same world. Leave-one-out keeps the baseline independent of
    the episode it is applied to, so the gradient stays unbiased. Within-episode
    shape is untouched — GAE still does the per-step credit assignment; only
    the constant per-episode offset the world imposed is removed.

    Episodes whose world has no sibling in this window are left alone.
    """
    T, N = advantages.shape
    out = advantages.copy()

    segs = []           # (env, t0, t1, world)
    for i in range(N):
        t0 = 0
        for t in np.flatnonzero(dones[:, i]):
            segs.append((i, t0, int(t), int(world_buf[t, i])))
            t0 = int(t) + 1
    if not segs:
        return out

    abar = np.array([out[t0:t1 + 1, i].mean() for (i, t0, t1, _) in segs])
    by_world = {}
    for k, (_, _, _, w) in enumerate(segs):
        by_world.setdefault(w, []).append(k)

    for ks in by_world.values():
        if len(ks) < 2:
            continue
        total = abar[ks].sum()
        denom = len(ks) - 1
        for k in ks:
            loo = (total - abar[k]) / denom
            i, t0, t1, _ = segs[k]
            out[t0:t1 + 1, i] -= loo
    return out


def compute_gae(rewards, values, dones, next_values):
    n_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        next_val = next_values if t == n_steps - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + GAMMA * next_val * non_terminal - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# -- PPO update ---------------------------------------------------------------


def ppo_update(
    policy,
    optimizer,
    obs,
    masks,
    actions,
    dones,
    old_logp,
    advantages,
    returns,
    start_state,
    device,
    entropy_coef=0.01,
    prior_logits=None,
    kl_coef=0.0,
    deep_weight=0.0,
):
    """Sequence-aware PPO update for the recurrent policy.

    Buffers are shape (T, N_envs, ...). Minibatches are by *env index* (not by
    individual timestep) so the LSTM can be replayed in temporal order within
    each minibatch starting from the stored start_state for those envs.

    With prior_logits + kl_coef > 0, adds kl_coef * KL(pi_theta || pi_prior)
    on the rollout states (VPT-style anchor to a frozen BC prior). Both logits
    are masked with the same finite -1e8 fill, so masked actions contribute
    exactly 0 to the KL (prob ~0 times a finite log-ratio).
    """
    T, N_envs = obs.shape[:2]
    # Advantage normalization over the full rollout (flat).
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    # Translate MINIBATCH_SIZE (in samples) into envs-per-minibatch.
    mb_envs = max(1, MINIBATCH_SIZE // T)

    # Upload the whole rollout once; minibatches are sliced on-GPU. The old
    # per-minibatch numpy uploads pushed ~PPO_EPOCHS x the rollout (~0.5 GB
    # per iteration) over PCIe.
    obs_g = torch.from_numpy(obs).to(device)
    masks_g = torch.from_numpy(masks).to(device)
    act_g = torch.from_numpy(actions).long().to(device)
    done_g = torch.from_numpy(dones).to(device)
    old_logp_g = torch.from_numpy(old_logp).to(device)
    adv_g = torch.from_numpy(advantages).to(device)
    ret_g = torch.from_numpy(returns).to(device)
    prior_g = (torch.from_numpy(prior_logits).to(device)
               if prior_logits is not None and kl_coef > 0.0 else None)

    pg_loss_total = v_loss_total = entropy_total = 0.0
    n_updates = 0
    h0, c0 = start_state

    for _ in range(PPO_EPOCHS):
        idx = np.random.permutation(N_envs)
        for start in range(0, N_envs, mb_envs):
            mb = idx[start : start + mb_envs]
            mb_t = torch.from_numpy(mb).to(device)

            obs_mb = obs_g[:, mb_t]
            masks_mb = masks_g[:, mb_t]
            act_mb = act_g[:, mb_t]
            done_mb = done_g[:, mb_t]
            old_logp_mb = old_logp_g[:, mb_t]
            adv_mb = adv_g[:, mb_t]
            ret_mb = ret_g[:, mb_t]
            start_mb = (h0[:, mb_t].detach(), c0[:, mb_t].detach())

            use_kl = prior_logits is not None and kl_coef > 0.0
            if use_kl:
                new_logp, entropy, values, new_logits = policy.evaluate_sequence(
                    obs_mb, masks_mb, act_mb, done_mb, start_mb, return_logits=True
                )
            else:
                new_logp, entropy, values = policy.evaluate_sequence(
                    obs_mb, masks_mb, act_mb, done_mb, start_mb
                )

            ratio = (new_logp - old_logp_mb).exp()
            pg1 = -adv_mb * ratio
            pg2 = -adv_mb * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_per = torch.max(pg1, pg2)
            if QUANTILE_VALUE:
                vq = policy._vq.view(*ret_mb.shape, N_QUANTILES)
                u = ret_mb.unsqueeze(-1) - vq
                huber = torch.where(u.abs() <= 1.0, 0.5 * u * u, u.abs() - 0.5)
                taus = (torch.arange(N_QUANTILES, device=u.device) + 0.5) / N_QUANTILES
                v_per = ((taus - (u.detach() < 0).float()).abs() * huber).sum(-1)
            else:
                v_per = (values - ret_mb) ** 2
            if deep_weight != 0.0:
                # positive: upweight deep-game (xp >= thresh); NEGATIVE value:
                # upweight EARLY game (xp < thresh) by |deep_weight| — the
                # anti-forgetting ingredient (opening-skill erosion, 2026-07-30)
                if deep_weight > 0.0:
                    w = 1.0 + deep_weight * (obs_mb[..., 3] >= DEEP_XP_OBS_THRESHOLD).float()
                else:
                    w = 1.0 + (-deep_weight) * (obs_mb[..., 3] < DEEP_XP_OBS_THRESHOLD).float()
                w = w / w.mean()
                pg_loss = (pg_per * w).mean()
                v_loss = (v_per * w).mean()
            else:
                pg_loss = pg_per.mean()
                v_loss = v_per.mean()
            ent_loss = -entropy.mean()
            loss = pg_loss + VALUE_COEF * v_loss + entropy_coef * ent_loss
            if SIL_COEF > 0.0:
                # Self-imitation (Oh et al. 2018, n-step variant): extra
                # gradient only on positive surprises R > V — reinforce what
                # worked more than PPO's symmetric update does. Targets
                # rare-success replication (crossings) directly.
                a_plus = (ret_mb - values).detach().clamp(min=0.0)
                sil_pg = (-new_logp * a_plus).mean()
                sil_v = 0.5 * ((ret_mb - values).clamp(min=0.0) ** 2).mean()
                loss = loss + SIL_COEF * (sil_pg + 0.5 * sil_v)
            if getattr(policy, "spr_aux", False) and AUX_COEF > 0.0:
                loss = loss + AUX_COEF * policy.aux_loss(obs_mb, act_mb, done_mb)
            if use_kl:
                prior_mb = prior_g[:, mb_t]
                cur_lsm = F.log_softmax(new_logits, dim=-1)
                kl = (cur_lsm.exp() * (cur_lsm - F.log_softmax(prior_mb, dim=-1))).sum(-1)
                loss = loss + kl_coef * kl.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            pg_loss_total += pg_loss.item()
            v_loss_total += v_loss.item()
            entropy_total += entropy.mean().item()
            n_updates += 1

    return (
        pg_loss_total / n_updates,
        v_loss_total / n_updates,
        entropy_total / n_updates,
    )


# -- Expert iteration: search + distill ---------------------------------------


@torch.no_grad()
def ei_search(policy, search_env, snapshots, device):
    """Search each snapshot: force each top-k candidate action in EI_CLONES
    rng-decorrelated clones, roll the policy to death/horizon, score by XP.
    All decisions x candidates x clones run as ONE batched rollout.

    Returns distillation targets: list of (obs, mask, h, c, best_action).
    """
    D = len(snapshots)
    if D == 0:
        return []
    K, C = EI_TOPK, EI_CLONES
    width = D * K * C
    assert width <= search_env.num_envs

    # Candidate actions per decision: top-k valid by current policy logits.
    obs_b = torch.from_numpy(np.stack([s["obs"] for s in snapshots])).to(device)
    mask_b = torch.from_numpy(np.stack([s["mask"] for s in snapshots])).to(device)
    h_b = torch.cat([s["h"].unsqueeze(1) for s in snapshots], dim=1).to(device)
    c_b = torch.cat([s["c"].unsqueeze(1) for s in snapshots], dim=1).to(device)
    logits, _, _ = policy.step(obs_b, mask_b, (h_b, c_b))
    logits_np = logits.cpu().numpy()
    cands = np.zeros((D, K), dtype=np.int32)
    for d in range(D):
        valid = np.flatnonzero(snapshots[d]["mask"])
        order = valid[np.argsort(-logits_np[d][valid])]
        picks = order[:K]
        # pad with the best action if fewer than K valid
        cands[d] = np.pad(picks, (0, K - len(picks)), mode="edge")

    # Load every clone: same blob per decision block, fresh rng stamped into
    # the blob head (uint64 rng is the struct's first field).
    rng = np.random.default_rng(np.random.randint(2**31))
    for d in range(D):
        base = bytearray(snapshots[d]["blob"].tobytes())
        for k in range(K):
            for j in range(C):
                idx = (d * K + k) * C + j
                base[0:8] = int(rng.integers(1, 2**63)).to_bytes(8, "little")
                search_env._engine.load_state(idx, np.frombuffer(bytes(base), dtype=np.uint8))

    # Tile hidden state: decision d -> its K*C clones.
    h0 = h_b.repeat_interleave(K * C, dim=1).contiguous()
    c0 = c_b.repeat_interleave(K * C, dim=1).contiguous()
    first = np.repeat(cands.reshape(-1), C).astype(np.int32)

    eng = search_env._engine
    score = np.full(width, np.nan, dtype=np.float64)
    full_acts = np.zeros(eng.num_envs, dtype=np.int32)
    full_acts[:width] = first
    h, c = h0, c0
    for step in range(EI_HORIZON):
        eng.step(full_acts, auto_reset=False)
        died = eng.terminated.astype(bool)[:width] & np.isnan(score)
        for i in np.flatnonzero(died):
            score[i] = float(eng.last_episode_info[i, Info.XP])
        alive = np.isnan(score)
        if not alive.any():
            break
        m = eng.action_mask.astype(bool)[:width]
        m[~alive] = False
        m[~alive, 0] = True
        obs_t = torch.from_numpy(eng.obs[:width].copy()).to(device)
        mask_t = torch.from_numpy(m).to(device)
        a, _, _, (h, c) = policy.get_action_and_value(obs_t, mask_t, (h, c))
        full_acts[:width] = a.cpu().numpy().astype(np.int32)
    alive = np.isnan(score)
    if alive.any():
        for i in np.flatnonzero(alive):
            score[i] = float(eng.info[i, Info.XP])

    # v2: emit a target ONLY when search disagrees with the policy's argmax
    # AND the win is statistically significant (mean gap > combined SE of the
    # two candidate estimates) — the winner's-curse filter. v1 distilled every
    # round winner from a stale FIFO and lost -12.5 eval: noisy labels acted
    # as entropy injection, not teaching.
    targets = []
    sc = score.reshape(D, K, C)
    means = sc.mean(axis=2)
    ses = sc.std(axis=2) / np.sqrt(C)
    for d in range(D):
        b = int(np.argmax(means[d]))
        best = int(cands[d][b])
        pol = int(cands[d][0])  # candidates are logit-ordered; [0] = argmax
        if best == pol:
            continue
        gap = means[d][b] - means[d][0]
        if gap <= ses[d][b] + ses[d][0]:
            continue
        targets.append((snapshots[d]["obs"], snapshots[d]["mask"],
                        snapshots[d]["h"], snapshots[d]["c"], best))
    return targets


def ei_distill(policy, optimizer, buffer, device):
    """One CE gradient step pulling the policy toward search-best actions."""
    if not buffer:
        return 0.0
    obs = torch.from_numpy(np.stack([b[0] for b in buffer])).to(device)
    mask = torch.from_numpy(np.stack([b[1] for b in buffer])).to(device)
    h = torch.cat([b[2].unsqueeze(1) for b in buffer], dim=1).to(device)
    c = torch.cat([b[3].unsqueeze(1) for b in buffer], dim=1).to(device)
    target = torch.tensor([b[4] for b in buffer], dtype=torch.long, device=device)
    logits, _, _ = policy.step(obs, mask, (h, c))
    loss = EI_COEF * F.cross_entropy(logits, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return float(loss.item())


# -- Main ---------------------------------------------------------------------


def select_device():
    """The first visible CUDA device, else CPU.

    Which GPU that is comes from CUDA_VISIBLE_DEVICES, the standard knob:

        CUDA_VISIBLE_DEVICES=1 just train        # one card
        CUDA_VISIBLE_DEVICES=1 just baseline     # a whole sweep

    Ids are the ones nvidia-smi prints (see CUDA_DEVICE_ORDER at the top of
    this file).

    Falling back to CPU is ~40x slower, and the usual cause is a torch build
    that does not match the driver rather than a missing GPU -- pip resolves
    `torch>=2.4` to the newest CUDA build, which needs a newer driver. Say so
    instead of quietly running 40x slow.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("!! No CUDA device available -- training on CPU, roughly 40x slower.")
    if shutil.which("nvidia-smi"):
        print(f"   This machine has an NVIDIA driver, so torch {torch.__version__} "
              f"(CUDA {torch.version.cuda}) probably does not match it.")
        print("   Install a matching build, e.g.:")
        print("     uv pip install torch==2.8.0 --index-url "
              "https://download.pytorch.org/whl/cu128")
    print(flush=True)
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--fresh-optimizer", action="store_true",
                        help="Initialize optimizer from scratch instead of loading checkpoint state")
    parser.add_argument("--reset-iteration", action="store_true",
                        help="Start iteration counter at 0 (required for fine-tunes where "
                             "the new TOTAL_ENV_STEPS is smaller than the source checkpoint)")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override TOTAL_ENV_STEPS (env steps to train for)")
    parser.add_argument("--lr", type=float, default=None, help="Override peak learning rate")
    parser.add_argument("--entropy-start", type=float, default=None,
                        help="Override starting entropy coefficient (use the checkpoint's "
                             "endpoint value for conservative fine-tuning)")
    parser.add_argument("--entropy-end", type=float, default=None,
                        help="Override ending entropy coefficient")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Override CHECKPOINT_EVERY_STEPS")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run fresh-reset eval on --resume checkpoint")
    parser.add_argument("--full-eval", action="store_true",
                        help=f"Score on the full banks instead of the training "
                             f"one: {len(BANK_SEEDS)} x {BANK_WORLDS} worlds "
                             f"at seeds {','.join(map(str, BANK_SEEDS))}. "
                             f"Set DM_EVAL_SEEDS to score elsewhere. "
                             f"Implies --eval-only.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (torch/np/env). Eval seed stays fixed at 7 so all "
                             "configs are scored on the same test set. Vary this to get error bars.")
    parser.add_argument("--xp-shaping", type=float, default=None,
                        help="Override REWARD_CONFIG['xp_shaping_scale'] (for fast reward sweeps)")
    parser.add_argument("--reward-override", action="append", default=[], metavar="KEY=VAL",
                        help="Override any REWARD_CONFIG term, e.g. --reward-override death_penalty=-0.5 "
                             "(repeatable; for fast multi-seed reward sweeps without editing the file)")
    parser.add_argument("--reward-transform", type=str, default=None, choices=["symlog", "sqrt", "none"],
                        help="Override REWARD_TRANSFORM (per-step reward squash)")
    parser.add_argument("--mlp", action="store_true",
                        help="Exact MLP ablation: skip the LSTM (gate frozen at 0, out = feats)")
    parser.add_argument("--deep-weight", type=float, default=0.0,
                        help="Extra gradient weight on transitions with episode XP >= 600 "
                             "(mine the rare deep-game experience; 2.0 = 3x weight)")
    parser.add_argument("--t1-weapon-starts", type=float, default=0.0,
                        help="Probability of starting an episode with a T1 weapon (cost-side "
                             "fight-EV lever; eval always clean)")
    parser.add_argument("--max-level", type=int, default=0,
                        help="Armor curriculum stage 1: truncate episodes at this adventurer level")
    parser.add_argument("--armor-coef", type=float, default=0.0,
                        help="Armor curriculum stage 1: reward = armor_score delta (this coef) "
                             "+ death_penalty + level_up; overrides the reward config")
    parser.add_argument("--curriculum-snapshots", type=float, default=0.0,
                        help="Go-Explore: probability of starting an episode from a captured "
                             "mid-game snapshot (engine pool, XP window 169-675 = levels 13-26). "
                             "Eval always runs with this OFF.")
    parser.add_argument("--engine-param", action="append", default=[], metavar="KEY=VAL",
                        help="Set an exact-engine config param by name (strcmp chain in "
                             "dmfast_exact.c), e.g. mask_block_flee_safe_threshold=1.5. "
                             "Repeatable. Applied to the TRAINING env only (eval resets "
                             "state but engine params persist — also applies at eval).")
    parser.add_argument("--shop-gate", action="store_true",
                        help="Build-band shopping gate: at L6-11 market phase with gold and "
                             "an affordable item, block explore until the market is used "
                             "(applied in rollouts AND final eval — policy+mask system)")
    parser.add_argument("--doorstep-resets", type=str, default=None,
                        help="Path to doorstep blob pool (.npz; the pool builder is "
                             "not in this repo). Episode resets are redirected "
                             "to mid-game doorstep states — continuation training.")
    parser.add_argument("--doorstep-prob", type=float, default=1.0,
                        help="Fraction of resets redirected to doorstep states")
    parser.add_argument("--deep-xp-thresh", type=float, default=None,
                        help="Override DEEP_XP_OBS_THRESHOLD (raw XP units)")
    parser.add_argument("--sil-coef", type=float, default=0.0,
                        help="Self-imitation aux coefficient (0 = off)")
    parser.add_argument("--transformer", action="store_true",
                        help="Entity-token transformer policy: the obs is "
                             "re-viewed as 52 typed tokens (CLS/player/stats/"
                             "equipment/bag/beast/market) and action logits are "
                             "entity-conditioned (bag actions from bag tokens, "
                             "market from market tokens). Stateless. NOTE: a "
                             "transformer trunk was shelved before (exp460) at "
                             "~5x slower fps for equal early learning.")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--buy-boost", type=float, default=0.0,
                        help="Reward curriculum: multiply the ITEM-buying "
                             "rewards (buy_item, buy_t1_item, buy_weapon_t1, "
                             "buy_quality_delta, buy_asset_bank_delta) by this "
                             "for the first --buy-boost-frac of training, then "
                             "revert. buy_potion is NOT boosted. 0/1 = off.")
    parser.add_argument("--buy-boost-frac", type=float, default=0.2,
                        help="Fraction of training the buy boost is active")
    parser.add_argument("--buy-boost-ramp", action="store_true",
                        help="Anneal the multiplier down across the window "
                             "instead of a hard revert (gentler on the critic)")
    parser.add_argument("--world-pool", type=int, default=0,
                        help="Stochasticity curriculum: keep a fixed pool of N "
                             "starting worlds and draw episode starts from it, "
                             "annealing toward fresh worlds. 0 = off. Capped at "
                             "NUM_ENVS. BIASED (unlike --group-size): a small "
                             "pool can be memorised.")
    parser.add_argument("--world-pool-start", type=float, default=0.4,
                        help="Fraction of episode starts drawn from the pool at "
                             "the START of training")
    parser.add_argument("--world-pool-end", type=float, default=0.0,
                        help="...and at the END (linear anneal)")
    parser.add_argument("--group-size", type=int, default=0,
                        help="Common random numbers: clone the starting world "
                             "across each group of N envs and centre advantages "
                             "by the group mean. Removes the ~43%% of return "
                             "variance fixed by the start that the critic "
                             "structurally cannot see (the rng seed is not in "
                             "the obs). 0 = off. Must divide NUM_ENVS. "
                             "Costs world diversity: N x fewer distinct worlds "
                             "per rollout.")
    parser.add_argument("--equivariant-head", action="store_true",
                        help="Permutation-equivariant action head: score the 15 bag and "
                             "25 market slots with ONE shared scorer over item embeddings "
                             "instead of per-slot positional logits. Every item observation "
                             "then trains the same weights (~25x effective data for the buy "
                             "head) and slot permutation stops mattering by construction.")
    parser.add_argument("--quantile-value", action="store_true",
                        help="Distributional critic: 32-quantile value head, quantile-"
                             "regression (huber) loss; GAE uses the quantile mean")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Trunk/LSTM width. Also sets --embed-dim unless that is "
                             "given explicitly (parameter-scaling axis)")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Encoder output width, independently of --hidden-dim")
    parser.add_argument("--trunk-blocks", type=int, default=None,
                        help="Number of residual blocks in the trunk")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Checkpoint dir name (enables concurrent experiments)")
    parser.add_argument("--explore-eps", type=float, default=0.0,
                        help="Forced-exploration prob per step (value repair for search)")
    parser.add_argument("--fight-drill", action="store_true",
                        help="Combat-drill mode: with --doorstep-resets pointing at a fight "
                             "pool, every combat resolution is a synthetic episode terminal "
                             "and a fresh fight is injected. Trains combat execution only.")
    parser.add_argument("--drill-hp-cost", type=float, default=0.1,
                        help="Reward cost per HP lost during drill fights")
    parser.add_argument("--ei", action="store_true",
                        help="Expert iteration: search decision states over the exact engine "
                             "(snapshot/broadcast) and distill search-best actions via CE")
    parser.add_argument("--bc-anchor", type=str, default=None,
                        help="Path to a frozen BC policy checkpoint; adds a KL(pi||prior) loss "
                             "on rollout states (VPT-style anchor), linearly annealed to 0")
    parser.add_argument("--kl-coef", type=float, default=0.1,
                        help="Starting coefficient for the --bc-anchor KL term")
    args = parser.parse_args()

    global REWARD_TRANSFORM
    if args.reward_transform is not None:
        REWARD_TRANSFORM = args.reward_transform
        print(f"reward transform: {REWARD_TRANSFORM}")

    if args.xp_shaping is not None:
        REWARD_CONFIG["xp_shaping_scale"] = args.xp_shaping
    for kv in args.reward_override:
        key, _, val = kv.partition("=")
        if key not in REWARD_CONFIG:
            raise SystemExit(f"--reward-override: unknown reward key '{key}'")
        REWARD_CONFIG[key] = float(val)
        print(f"reward override: {key} = {float(val)}")

    if args.armor_coef > 0:
        REWARD_CONFIG.clear()
        REWARD_CONFIG.update({"armor_score": args.armor_coef, "death_penalty": -0.25,
                              "level_up": 1.0})
        print(f"ARMOR CURRICULUM: reward = {args.armor_coef}*armor_score_delta + death + level_up")

    global TOTAL_ENV_STEPS, LR, CHECKPOINT_EVERY_STEPS
    global ENTROPY_COEF_START, ENTROPY_COEF_END
    global HIDDEN_DIM, EMBED_DIM, QUANTILE_VALUE, SIL_COEF, DEEP_XP_OBS_THRESHOLD
    global EQUIVARIANT_HEAD, TOKEN_D_MODEL, TOKEN_LAYERS, TOKEN_HEADS
    global TRUNK_NUM_BLOCKS, MEMORY_TYPE, DECOUPLED_VALUE, SPR_AUX, TOKEN_FF
    if args.deep_xp_thresh is not None:
        DEEP_XP_OBS_THRESHOLD = args.deep_xp_thresh / 32767.0
        print(f"deep/early xp threshold: {args.deep_xp_thresh}")
    if args.sil_coef > 0.0:
        SIL_COEF = args.sil_coef
        print(f"SIL: coef {SIL_COEF} (positive-surprise self-imitation)")
    if args.quantile_value:
        QUANTILE_VALUE = True
        print(f"QUANTILE VALUE: {N_QUANTILES} quantiles, huber pinball loss")
    if args.d_model is not None: TOKEN_D_MODEL = args.d_model
    if args.n_layers is not None: TOKEN_LAYERS = args.n_layers
    if args.n_heads is not None: TOKEN_HEADS = args.n_heads
    if args.equivariant_head:
        EQUIVARIANT_HEAD = True
        print(f"EQUIVARIANT HEAD: shared pointer scorer over bag+market "
              f"(ptr_dim={PTR_DIM}); slot-positional logits retired")
    if args.hidden_dim is not None:
        HIDDEN_DIM = args.hidden_dim
        # --hidden-dim historically moved both widths together. It still does,
        # unless --embed-dim says otherwise, so the two are independently
        # reachable without changing what the old flag means on its own.
        if args.embed_dim is None:
            EMBED_DIM = args.hidden_dim
        print(f"param scaling: HIDDEN_DIM = {HIDDEN_DIM}")
    if args.embed_dim is not None:
        EMBED_DIM = args.embed_dim
        print(f"param scaling: EMBED_DIM = {EMBED_DIM}")
    if args.trunk_blocks is not None:
        TRUNK_NUM_BLOCKS = args.trunk_blocks
        print(f"param scaling: TRUNK_NUM_BLOCKS = {TRUNK_NUM_BLOCKS}")
    if args.total_steps is not None:
        TOTAL_ENV_STEPS = args.total_steps
    if args.lr is not None:
        LR = args.lr
    if args.entropy_start is not None:
        ENTROPY_COEF_START = args.entropy_start
    if args.entropy_end is not None:
        ENTROPY_COEF_END = args.entropy_end
    if args.entropy_start is not None or args.entropy_end is not None:
        print(f"entropy schedule: {ENTROPY_COEF_START} -> {ENTROPY_COEF_END}")
    if args.checkpoint_every is not None:
        CHECKPOINT_EVERY_STEPS = args.checkpoint_every

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # TF32 tensor cores for fp32 GEMMs + cuDNN (Ada). Off by default in
    # torch 2.x; numerics shift is within seed-noise for this workload.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    # Accepting .pt here would mean the scoring path executes whatever the
    # file contains. Eval is the one place that must stay safe to point at a
    # checkpoint someone else produced.
    if (args.eval_only or args.full_eval) and args.resume \
            and not str(args.resume).endswith(".safetensors"):
        raise SystemExit(
            f"eval only accepts .safetensors checkpoints, got {args.resume}\n"
            f"  .pt is a pickle: loading it runs arbitrary code.\n"
            f"  Convert it:  python checkpoint.py {args.resume}")

    device = select_device()
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")
    if args.full_eval:
        print(f"Train seed: {seed} (full eval: {len(BANK_SEEDS)} banks of "
              f"{BANK_WORLDS} worlds, seeds {','.join(map(str, BANK_SEEDS))})")
    else:
        print(f"Train seed: {seed} (eval: {EVAL_WORLDS} worlds at seed {EVAL_SEED})")

    env = GameEnv(
        num_envs=NUM_ENVS, seed=seed, max_steps=MAX_STEPS, reward_config=REWARD_CONFIG
    )
    if args.t1_weapon_starts > 0:
        env.set_engine_param("curriculum_t1_weapon_prob", args.t1_weapon_starts)
        print(f"T1-weapon curriculum: {args.t1_weapon_starts:.0%} of episodes start with a T1 weapon; eval runs clean")
    if args.curriculum_snapshots > 0:
        env.set_engine_param("curriculum_snapshot_prob", args.curriculum_snapshots)
        print(f"Go-Explore curriculum: {args.curriculum_snapshots:.0%} of resets from "
              f"mid-game snapshots (XP 169-675); eval runs clean")
    if args.max_level > 0:
        env.set_engine_param("max_level", float(args.max_level))
        print(f"Episodes truncate at adventurer level {args.max_level}")
    search_env = None
    if args.ei:
        search_width = EI_DECISIONS_PER_ROUND * EI_TOPK * EI_CLONES
        search_env = GameEnv(num_envs=search_width, seed=seed + 991,
                             max_steps=MAX_STEPS, reward_config=REWARD_CONFIG)
        search_env.reset(seed=seed + 991)
        print(f"Expert iteration ON (v2 fresh+significant): {EI_DECISIONS_PER_ROUND} decisions "
              f"x {EI_TOPK} cands x {EI_CLONES} clones every {EI_EVERY_ITERS} iters, "
              f"horizon {EI_HORIZON}, coef {EI_COEF}")
    # A resumed checkpoint records the architecture that produced it, so the
    # policy is rebuilt to match rather than to whatever the globals say today.
    # Without this you could train a shape you were then unable to load.
    resumed, use_lstm = None, not args.mlp
    if args.resume:
        resumed = checkpoint.load(args.resume, map_location="cpu")
        arch = resumed["arch"]
        if resumed["legacy"]:
            print(f"  {args.resume} is a legacy pickle checkpoint; architecture "
                  f"inferred from tensor shapes")
        for name, value in (("EMBED_DIM", arch.get("embed_dim")),
                            ("HIDDEN_DIM", arch.get("hidden_dim")),
                            ("TRUNK_NUM_BLOCKS", arch.get("num_trunk_blocks")),
                            ("TOKEN_D_MODEL", arch.get("d_model")),
                            ("TOKEN_LAYERS", arch.get("n_layers")),
                            ("TOKEN_HEADS", arch.get("n_heads")),
                            ("TOKEN_FF", arch.get("ff_dim"))):
            if value is not None and globals()[name] != value:
                print(f"  checkpoint architecture: {name} {globals()[name]} -> {value}")
                globals()[name] = value
        for name, key in (("MEMORY_TYPE", "memory_type"),
                          ("DECOUPLED_VALUE", "decoupled_value"),
                          ("QUANTILE_VALUE", "quantile_value"),
                          ("EQUIVARIANT_HEAD", "equivariant_head"),
                          ("SPR_AUX", "spr_aux")):
            if key in arch and globals()[name] != arch[key]:
                print(f"  checkpoint architecture: {name} {globals()[name]} -> {arch[key]}")
                globals()[name] = arch[key]
        # --mlp on the command line still wins; otherwise trust the record,
        # except on a legacy file where use_lstm was a guess.
        family = arch.get("architecture", "dm_lstm_v1")
        if family == "dm_transformer_v1":
            args.transformer = True
        elif not args.mlp and not arch.get("use_lstm_inferred"):
            use_lstm = family != "dm_mlp_v1"

    if args.transformer:
        policy = TokenPolicy().to(device)
        n_par = sum(p.numel() for p in policy.parameters())
        print(f"Arch mode: TOKEN TRANSFORMER  d_model={TOKEN_D_MODEL} "
              f"layers={TOKEN_LAYERS} heads={TOKEN_HEADS} ff={TOKEN_FF} | "
              f"{TokenPolicy.N_TOKENS} entity tokens | {n_par:,} params | "
              f"stateless (no recurrence)", flush=True)
    else:
        policy = MLPPolicy(use_lstm=use_lstm).to(device)
    if not use_lstm:
        print("Arch mode: MLP ablation (LSTM skipped, gate frozen)")
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    start_iteration = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        policy.load_state_dict(resumed["model"])
        if args.fresh_optimizer:
            print("  --fresh-optimizer: skipping optimizer state load")
        elif resumed["optimizer"] is None:
            print("  no optimizer state in the checkpoint: starting fresh")
        else:
            optimizer.load_state_dict(resumed["optimizer"])
        start_iteration = resumed["meta"].get("iteration", 0)
        if args.reset_iteration:
            print(f"  --reset-iteration: discarding source iteration {start_iteration}")
            start_iteration = 0
        print(f"  loaded model (avg_xp={resumed['meta'].get('avg_xp', '?')}), "
              f"resuming from iteration {start_iteration}")

    # Compile the feed-forward hot path in place (in-place .compile() keeps
    # state_dict keys intact, unlike torch.compile(policy) which also was a
    # silent no-op here: it only wraps .forward, and this policy is driven via
    # get_action_and_value/evaluate_sequence). The LSTM cell stays eager —
    # it's a single fused cuDNN call per step already.
    if args.transformer:
        # TokenPolicy has no encoder/input_trunk split; compile the whole
        # transformer stack in place (same .compile() trick, state_dict keys
        # stay intact).
        policy.encoder.compile()
    else:
        policy.encoder.compile()
        policy.input_trunk.compile()

    for kv in args.engine_param:
        key, val = kv.split("=", 1)
        rc = env._engine.set_engine_param(key, float(val))
        if rc != 1:
            raise SystemExit(f"--engine-param: engine rejected '{key}'")
        print(f"engine param: {key} = {float(val)}")

    doorstep_pool = None
    inject_pending = np.zeros(NUM_ENVS, dtype=bool)
    if args.doorstep_resets:
        doorstep_pool = DoorstepPool(args.doorstep_resets, seed=args.seed)
        print(f"DOORSTEP RESETS: prob={args.doorstep_prob} — continuation training")
        # Start the very first episodes from doorstep states too (otherwise the
        # opening ~200 steps per env are all fresh starts).
        for i in range(NUM_ENVS):
            if args.doorstep_prob >= 1.0 or np.random.random() < args.doorstep_prob:
                env._engine.load_state(i, doorstep_pool.sample())
                inject_pending[i] = True  # first-step reward invalid (stale prevs)

    prior = None
    if args.bc_anchor:
        print(f"KL anchor: {args.bc_anchor} (kl_coef={args.kl_coef}, linear anneal to 0)")
        # Anchor ckpts are LSTMPolicy since the mid-game BC thread (il/
        # bc_midgame_v3.pt); the original MLP BC anchors are historical.
        prior = LSTMPolicy().to(device)
        prior_ckpt = torch.load(args.bc_anchor, map_location=device, weights_only=False)
        prior.load_state_dict(prior_ckpt["model_state_dict"])
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False)
        prior.encoder.compile()
        prior.input_trunk.compile()

    if args.full_eval:
        mask = shop_gate_mask if args.shop_gate else None
        per_seed, mean_xp, trunc = bank_score(
            policy, device, mask_transform=mask, progress=True)
        print("---")
        print_bank_result(per_seed, mean_xp, trunc, BANK_BATCH)
        env.close()
        return

    if args.eval_only:
        eval_xps, eval_trunc, eval_stats = eval_policy(
            env, policy, device, worlds=EVAL_WORLDS, seed=EVAL_SEED,
            mask_transform=shop_gate_mask if args.shop_gate else None)
        print("---")
        print(f"eval_avg_xp:       {eval_xps.mean():.1f}")
        print(f"eval_max_xp:       {eval_xps.max():.1f}")
        print(f"eval_median_xp:    {np.median(eval_xps):.1f}")
        print(f"eval_worlds:       {len(eval_xps)}")
        print(f"eval_seed:         {EVAL_SEED}")
        print(f"eval_batch:        {env.num_envs}")
        print(f"eval_truncated:    {eval_trunc}")
        print_episode_stats(eval_stats)
        env.close()
        return

    n_params = sum(p.numel() for p in policy.parameters())
    steps_per_iter = ROLLOUT_STEPS * NUM_ENVS
    num_iterations = TOTAL_ENV_STEPS // steps_per_iter
    print(
        f"Policy: {n_params:,} params | {NUM_ENVS} envs | {ROLLOUT_STEPS} steps/rollout | {num_iterations} iters | {TOTAL_ENV_STEPS:,} env steps"
    )
    print(
        f"Arch: embed={EMBED_DIM}, hidden={HIDDEN_DIM}, trunk_blocks={TRUNK_NUM_BLOCKS} | "
        f"FlatItemEnc item_out={ITEM_OUT} (phase-gated)"
    )

    # local/ is gitignored: training output never enters the committed tree.
    ckpt_root = os.environ.get("DM_CHECKPOINTS", "local/checkpoints")
    ckpt_dir = (f"{ckpt_root}/{args.run_name}" if args.run_name
                else f"{ckpt_root}/{TOTAL_ENV_STEPS // 1_000_000}M")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(tag, iteration=0, avg_xp=0.0, max_xp=0.0):
        state = policy._orig_mod.state_dict() if hasattr(policy, '_orig_mod') else policy.state_dict()
        # safetensors, not a pickle: loading a checkpoint must not be able to
        # execute code, and the architecture travels with the weights.
        path = checkpoint.save(
            os.path.join(ckpt_dir, tag),
            state,
            arch=current_arch(use_lstm=use_lstm, transformer=args.transformer),
            meta={"total_env_steps": TOTAL_ENV_STEPS, "iteration": iteration,
                  "avg_xp": avg_xp, "max_xp": max_xp,
                  "training": current_training(seed=seed)},
            optimizer_state=optimizer.state_dict(),
        )
        print(f"  >> saved {path}", flush=True)

    env.reset(seed=seed)
    groups = None
    if args.group_size:
        groups = GroupWorlds(env.num_envs, args.group_size)
        groups.prime(env)
        print(f"GROUP WORLDS: {groups.num_groups} worlds x {args.group_size} "
              f"clones; advantages centred per group", flush=True)
    buy_boost = None
    if args.buy_boost and args.buy_boost != 1.0:
        buy_boost = BuyBoostCurriculum(env, args.buy_boost, args.buy_boost_frac,
                                       args.buy_boost_ramp)
    world_pool = None
    if args.world_pool:
        world_pool = WorldPool(env, args.world_pool,
                               args.world_pool_start, args.world_pool_end)
        print(f"WORLD POOL: {world_pool.size} fixed worlds, "
              f"p(from pool) {args.world_pool_start:.2f} -> "
              f"{args.world_pool_end:.2f} over training", flush=True)
    t_start = time.time()
    t_prev_iter_end = t_start

    death_xp_buf = deque(maxlen=DEATH_XP_WINDOW)
    # Persistent LSTM state, threaded across rollouts. Episodes that finish
    # inside a rollout get their state zeroed by collect_rollout.
    core_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    lstm_state = core_policy.init_state(NUM_ENVS, device)
    prior_state = prior.init_state(NUM_ENVS, device) if prior is not None else None

    for iteration in range(start_iteration + 1, num_iterations + 1):
        progress = iteration / num_iterations
        if world_pool is not None:
            world_pool.set_progress(progress)
        if buy_boost is not None:
            buy_boost.set_progress(progress)
        warmup_iters = 5
        if iteration <= warmup_iters:
            lr = LR * iteration / warmup_iters
        else:
            lr = LR * (1.0 - (iteration - warmup_iters) / (num_iterations - warmup_iters))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        ent_coef = (
            ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress
        )

        ei_round = args.ei and iteration % EI_EVERY_ITERS == 0
        (obs_buf, masks, actions, logp, values, rewards, dones, worlds,
         next_val, death_xps, start_state, lstm_state, prior_logits,
         prior_state, snapshots) = collect_rollout(
            env, policy, device, lstm_state, prior=prior, prior_state=prior_state,
            snap_count=EI_DECISIONS_PER_ROUND if ei_round else 0,
            doorstep_pool=doorstep_pool, doorstep_prob=args.doorstep_prob,
            inject_pending=inject_pending, shop_gate=args.shop_gate,
            fight_drill=args.fight_drill, drill_hp_cost=args.drill_hp_cost,
            explore_eps=args.explore_eps, groups=groups,
            world_pool=world_pool,
        )

        death_xp_buf.extend(death_xps)

        # Reward transform (squashes large rewards; see REWARD_TRANSFORM)
        if REWARD_TRANSFORM == "symlog":
            rewards = np.sign(rewards) * np.log1p(np.abs(rewards))
        elif REWARD_TRANSFORM == "sqrt":
            rewards = np.sign(rewards) * np.sqrt(np.abs(rewards))
        elif REWARD_TRANSFORM == "none":
            pass
        else:
            raise ValueError(f"unknown REWARD_TRANSFORM: {REWARD_TRANSFORM}")

        advantages, returns = compute_gae(rewards, values, dones, next_val)
        if groups is not None:
            # Remove the shared starting-world component the critic cannot see.
            # Advantages only — `returns` stays the critic's unbiased target.
            advantages = group_center_advantages(advantages, dones, worlds)

        # v3: distill BEFORE the PPO epochs so the clipped PPO objective
        # immediately re-anchors the policy around the distilled point —
        # v2's post-update CE left an unconstrained step as the iteration's
        # last word (-6.2 vs bar).
        ei_loss = 0.0
        if ei_round and snapshots:
            core = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
            fresh = ei_search(core, search_env, snapshots, device)
            if fresh:
                ei_loss = ei_distill(core, optimizer, fresh, device)

        pg_loss, vl, ent = ppo_update(
            policy,
            optimizer,
            obs_buf,
            masks,
            actions,
            dones,
            logp,
            advantages,
            returns,
            start_state,
            device,
            entropy_coef=ent_coef,
            prior_logits=prior_logits,
            kl_coef=args.kl_coef * (1.0 - progress) if prior is not None else 0.0,
            deep_weight=args.deep_weight,
        )

        iter_end = time.time()
        elapsed = iter_end - t_start
        iter_seconds = iter_end - t_prev_iter_end
        iter_fps = steps_per_iter / max(iter_seconds, 1e-9)
        t_prev_iter_end = iter_end
        avg_death_xp = np.mean(death_xp_buf) if death_xp_buf else 0.0
        print(
            f"it {iteration:04d}/{num_iterations} | {elapsed:.0f}s | "
            f"fps: {iter_fps:.0f} | "
            f"pg: {pg_loss:.4f} | v: {vl:.4f} | ent: {ent:.2f} | "
            f"death_xp: {avg_death_xp:.0f} ({len(death_xp_buf)})"
            + (f" | ei: {ei_loss:.3f}" if args.ei else ""),
            flush=True,
        )

        # Emit a checkpoint every CHECKPOINT_EVERY_STEPS env steps
        iters_per_ckpt = max(1, CHECKPOINT_EVERY_STEPS // steps_per_iter)
        if iteration % iters_per_ckpt == 0 and iteration != num_iterations:
            step_M = (iteration * steps_per_iter) // 1_000_000
            save_checkpoint(f"step{step_M}M", iteration=iteration, avg_xp=avg_death_xp)

    total_seconds = time.time() - t_start
    fps = (num_iterations * steps_per_iter) / max(total_seconds, 1e-9)
    print(f"\nTraining done: {num_iterations} iterations in {total_seconds:.1f}s")

    xps = np.array(death_xp_buf) if death_xp_buf else np.array([0.0])
    print("---")
    print(f"avg_xp:            {xps.mean():.1f}")
    print(f"max_xp:            {xps.max():.1f}")
    print(f"median_xp:         {np.median(xps):.1f}")
    print(f"training_seconds:  {total_seconds:.1f}")
    print(f"fps:               {fps:.1f}")

    save_checkpoint("final", iteration=num_iterations, avg_xp=float(xps.mean()), max_xp=float(xps.max()))

    if args.curriculum_snapshots > 0:
        env.set_engine_param("curriculum_snapshot_prob", 0.0)  # eval hygiene
    if args.t1_weapon_starts > 0:
        env.set_engine_param("curriculum_t1_weapon_prob", 0.0)  # eval hygiene
    eval_xps, eval_trunc, eval_stats = eval_policy(
        env, policy, device,
        mask_transform=shop_gate_mask if args.shop_gate else None)
    print(f"eval_avg_xp:       {eval_xps.mean():.1f}")
    print(f"eval_max_xp:       {eval_xps.max():.1f}")
    print(f"eval_median_xp:    {np.median(eval_xps):.1f}")
    print(f"eval_worlds:       {len(eval_xps)}")
    print(f"eval_truncated:    {eval_trunc}")
    print_episode_stats(eval_stats)

    env.close()


if __name__ == "__main__":
    main()
