# Numeric interface constants; privileged critic is unsupported.
OBS_DIM = 463
ACTION_DIM = 57
PRIV_DIM = 0
SEM_DIM = 90
fused_itf = None
"""
RL training for Death Mountain gym using PPO with action masking.
MLP policy with categorical embedding encoder.
Usage: python train.py [--resume path/to/checkpoint.pt]
"""

import argparse

import os

import sys

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

import importlib

import semantic_obs

SEMANTIC_OBS = True

NUM_ENVS = 512

MAX_STEPS = 2048

TOTAL_ENV_STEPS = 10_000_000_000

CHECKPOINT_EVERY_STEPS = 250_000_000

ACT_DIM = ACTION_DIM

EMBED_DIM = 256

HIDDEN_DIM = 256

LSTM_HIDDEN = 32

TRUNK_NUM_BLOCKS = 1

DECOUPLED_VALUE = False

SPR_AUX = False

MEMORY_TYPE = "lstm"

ATTN_WINDOW = 32

ATTN_HEADS = 4

AUX_COEF = 0.1

AUX_ACTION_EMB = 16

ITEM_OUT = 12

LR = 1e-4

GAMMA = 0.995

GAE_LAMBDA = 0.90

CLIP_EPS = 0.2

ENTROPY_COEF_START = 0.03

ENTROPY_COEF_END = 0.005

VALUE_COEF = 0.08

SIL_COEF = 0.0

QUANTILE_VALUE = False

EQUIVARIANT_HEAD = True

PTR_RESIDUAL = False

STAT_HEAD = False

SET_ENCODER = False

ITEM_TF = True

ITEM_TF_DIM = 32

ITEM_TF_LAYERS = 1

ITEM_TF_HEADS = 1

ITEM_TF_COMPILE = True

ITEM_TF_CONTENT_NORM = False

ITEM_TF_SLOT_BIAS = False

ITEM_TF_AFFORD = False

ITEM_TF_ID_SCALE = 1.0

SET_ATTN = False

SET_NORM = False

THINK_STEPS = 1

SET_SEEDS = 2

MASK_ACTIONS = (7, 8, 9, 10)

LSTM_GRAD = True

SIM_WIDTH = 10

BEAST_WIDTH = 12

TOKEN_POLICY = False

TOKEN_D_MODEL, TOKEN_LAYERS, TOKEN_HEADS, TOKEN_FF = 128, 4, 4, 512

EX_NUM_BASE_ACTIONS = 11

PTR_DIM = 32

N_QUANTILES = 32

MAX_GRAD_NORM = 0.5

ROLLOUT_STEPS = 256

PPO_EPOCHS = 8

MINIBATCH_SIZE = 8192

GRAD_ACCUM = 1

DEATH_XP_WINDOW = 4096

DEATH_LENS = deque(maxlen=DEATH_XP_WINDOW)

REWARD_TRANSFORM = "symlog"

BF16 = False

PRIV_CRITIC = False

BF16_ENC = True

ASYNC_ROLLOUT = True

ASYNC_LAG_EPOCHS = 2

ROLLOUT_GRAPH = True

CUDA_ENV = True

FUSED_ITF = True

SLOT_PAIR = False

ITEM_MECHANICS = False

VALUE_ANCHOR = 0.0

WORLD = int(os.environ.get("WORLD_SIZE", "1"))

RANK = int(os.environ.get("RANK", "0"))

TERMINAL_ONLY = False

DEEP_XP_OBS_THRESHOLD = 600.0 / 32767.0

EI_DECISIONS_PER_ROUND = 16

EI_EVERY_ITERS = 2

EI_CLONES = 64

EI_TOPK = 3

EI_HORIZON = 150

EI_COEF = 0.2

class SetNorm(nn.Module):
    """Set Norm (Zhang et al., ICML 2022, arXiv 2206.11925).

    LayerNorm normalises each set ELEMENT over its own features, which erases
    the relative differences BETWEEN elements -- exactly the signal a set
    encoder exists to read (is this market item better than that one?). Set
    Norm normalises over the set and feature dims jointly, so element-to-element
    structure survives while the activation scale is still controlled.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):                       # (B, N, D)
        mu = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - mu) * torch.rsqrt(var + self.eps) * self.weight + self.bias

class SetPool(nn.Module):
    """Pooling by Multihead Attention (Set Transformer, Lee et al. 2019).

    mean+max pooling is permutation-invariant but FIXED: it cannot learn which
    items deserve attention, so a single strong market item is averaged in with
    24 irrelevant ones. PMA replaces it with `n_seeds` learned query vectors
    that attend over the set, so the pooled summary is itself learned while
    staying order-invariant. n_seeds * dim is kept equal to the mean+max width,
    so this is a drop-in swap and the comparison isolates pooling, not capacity.
    """

    def __init__(self, dim: int, n_seeds: int = None, heads: int = 2):
        n_seeds = SET_SEEDS if n_seeds is None else n_seeds
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(n_seeds, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = SetNorm(dim) if SET_NORM else nn.LayerNorm(dim)

    def forward(self, x):                      # (B, N, dim) -> (B, n_seeds*dim)
        q = self.seeds.unsqueeze(0).expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.ln(out).reshape(x.shape[0], -1)

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
    BEAST_OUT_DIM = BEAST_WIDTH  # --beast-width
    SIM_OUT_DIM = SIM_WIDTH   # --sim-width

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
        # sim carries 52.2% of the decision signal (occlusion, 20k states) but
        # was projected 10->10, i.e. 1.6% of the encoder's width. beast carried
        # 0.1% and got 12. Widen sim, shrink beast.
        self.sim_proj = nn.Sequential(
            nn.Linear(10, self.SIM_OUT_DIM),
            nn.SiLU(),
            nn.LayerNorm(self.SIM_OUT_DIM),
        )

        # 8 positional equip slots + the two pooled groups. mean+max gives 2
        # outputs per group; PMA gives SET_SEEDS (register slots) per group.
        pooled = 2 * (SET_SEEDS if SET_ATTN else 2)
        item_dim = (self.item_encoder.out_per_item * (8 + pooled) if SET_ENCODER
                    else self.item_encoder.out_dim)
        self.item_tf = None
        self.slot_pair = None
        if SLOT_PAIR:
            # Typed pairs instead of all-pairs attention: a bag/market item competes
            # with exactly one thing, the item equipped in its slot. One shared MLP
            # over [item, equipped-in-slot, adventurer ctx, beast], residual on the
            # item embedding; equipped items see an empty partner. Equivariant
            # over bag and market, ~1 ms per minibatch instead of the transformer's 4-7.
            d = self.item_encoder.out_per_item
            self.slot_pair = nn.Sequential(
                nn.Linear(2 * d + self.PHASE_EMB_DIM + 13 + self.BEAST_OUT_DIM, 32), nn.SiLU(),
                _init_linear(nn.Linear(32, d), gain=0.1))
            item_dim = d * (8 + 4)
        elif ITEM_TF:
            # Context token = phase embedding + the 13 adventurer scalars, so an
            # item is judged against the adventurer carrying it.
            self.item_tf = ItemTransformer(
                self.item_encoder.out_per_item,
                ctx_dim=self.PHASE_EMB_DIM + 13, beast_dim=self.BEAST_OUT_DIM)
            # The block is launch-bound, not FLOP-bound: 50 tokens at d=16 is
            # a long chain of tiny LayerNorm/GELU/residual kernels, and fusing
            # them is worth more than halving d. Measured on one minibatch:
            # 4.16x -> 2.03x vs the flat encoder.
            #
            # OFF by default, and the reason is a genuine conflict rather than
            # taste. The compiled callable is kept in a LIST so assigning it
            # cannot register OptimizedModule as a submodule and rename every
            # key to item_tf._orig_mod.* -- but that leaves the module
            # reachable by two paths (self.item_tf and _tf_call[0]._orig_mod),
            # and Dynamo refuses to trace a module it is already tracking:
            #   AssertionError: UnspecializedNNModuleVariable(ItemTransformer)
            #   is already tracked for mutation
            # Compiling is worth ~2x (d32L1: 4.16x -> 2.03x vs the flat
            # encoder), so this is worth fixing with state_dict hooks that
            # strip the prefix instead. Until then, correctness over speed.
            self._tf_call = (torch.compile(_item_tf_forward, dynamic=False)
                             if ITEM_TF_COMPILE else _item_tf_forward)
            # fused Triton path: one program per sample, nothing materialised
            # between the item embedding and the pooled output (fused_itf.py)
            self.fused = FUSED_ITF and fused_itf.supports(self.item_tf, self.item_encoder.out_per_item)
            item_dim = self.item_tf.out_dim
        if SET_ENCODER and SET_ATTN:
            d = self.item_encoder.out_per_item
            self.bag_pool = SetPool(d)
            self.mkt_pool = SetPool(d)
        total = (self.PHASE_EMB_DIM + 13 + item_dim
                 + self.BEAST_OUT_DIM + self.SIM_OUT_DIM)
        if SEMANTIC_OBS:
            total += semantic_obs.EXTRA_DIM
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
        flat = obs.reshape(-1, obs.shape[-1])
        B = flat.shape[0]
        # With --semantic the obs is 463 + EXTRA: the first 463 parse exactly as
        # before, the tail is appended to the feature stack (no embedding -- it
        # is already continuous and normalised).
        extra = flat[:, 463:] if flat.shape[-1] > 463 else None
        flat = flat[:, :463]

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
        beast    = self.beast_encoder(flat[:, 446:453])
        sim      = self.sim_proj(flat[:, 453:463])
        if self.slot_pair is not None:
            d = per_item.shape[-1]
            eq = per_item[:, :8]                                        # (B, 8, d)
            slot = items[:, 8:, 3].long().clamp(1, 8) - 1               # (B, 40): equipment index
            partner = torch.cat([torch.zeros_like(eq),
                                 torch.gather(eq, 1, slot.unsqueeze(-1).expand(-1, -1, d))], 1)
            ctx = torch.cat([phase, adv, beast], dim=-1).unsqueeze(1).expand(-1, 48, -1)
            per_item = per_item + self.slot_pair(torch.cat([per_item, partner, ctx], dim=-1))
            bag, mkt = per_item[:, 8:23], per_item[:, 23:]
            item_out = torch.cat([per_item[:, :8].reshape(B, -1), bag.mean(1), bag.amax(1),
                                  mkt.mean(1), mkt.amax(1)], dim=-1)
        elif self.item_tf is not None:
            afford = None
            if ITEM_TF_AFFORD:
                # ex_compute_market_prices: price = max((6-tier)*4 - cha, 1);
                # tier_monotone = (6-tier)/5 is already field 1. Gold is /511
                # and stats are /100 in the observation.
                gold = (flat[:, 4] * 511.0).unsqueeze(1)
                cha = (flat[:, 12] * 100.0).unsqueeze(1)
                price = torch.clamp(items[..., 1] * 20.0 - cha, min=1.0)
                mkt = torch.zeros_like(price); mkt[:, 23:] = 1.0
                mkt = mkt * (items[..., 0] > 0).to(price.dtype)
                afford = torch.stack([
                    torch.clamp(gold / price, 0.0, 3.0) / 3.0 * mkt,
                    (gold >= price).to(price.dtype) * mkt], dim=-1)
            if self.fused and per_item.is_cuda:
                item_out, per_item = fused_itf.fused_item_tf(
                    self.item_tf, per_item, torch.cat([phase, adv], dim=-1), beast)
            else:
                item_out, per_item = self._tf_call(
                    self.item_tf, per_item, torch.cat([phase, adv], dim=-1), beast,
                    items[..., 3] if ITEM_TF_SLOT_BIAS else None, afford)
        elif SET_ENCODER:
            # The flat reshape is positionally indexed: the SAME item in market
            # slot 7 lands in a different span of the trunk's input than in
            # slot 12, so every slot must be learned separately. That is exactly
            # the defect the equivariant ACTION head removed (+48 XP) -- on the
            # output side. This removes it on the input side.
            #
            # Equipment (8) stays positional: weapon / 5 armour / neck / ring
            # are distinct roles, not interchangeable slots. Bag (15) and
            # market (25) ARE interchangeable, so they become order-invariant
            # mean+max summaries. Nothing is lost for selection: the pointer
            # head scores per-item from `per_item` directly, so the trunk only
            # needs to know WHAT is on offer, not in which slot.
            eqp = per_item[:, :8].reshape(B, -1)
            bag = per_item[:, 8:23]
            mkt = per_item[:, 23:48]
            if SET_ATTN:
                item_out = torch.cat([eqp, self.bag_pool(bag),
                                      self.mkt_pool(mkt)], dim=-1)
            else:
                item_out = torch.cat([eqp,
                                      bag.mean(1), bag.amax(1),
                                      mkt.mean(1), mkt.amax(1)], dim=-1)
        else:
            item_out = per_item.reshape(B, -1)

        parts = [phase, adv, item_out, beast, sim]
        if extra is not None:
            parts.append(extra)
        features = torch.cat(parts, dim=-1)
        gated = features * torch.sigmoid(self.gate(phase))
        enc = self.proj(gated).reshape(*shape, -1)
        if return_items:
            return enc, per_item.reshape(*shape, per_item.shape[-2], per_item.shape[-1])
        return enc

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

def _init_nobias(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    return layer

class SwiGLUBlock(nn.Module):
    """Current-practice residual block: RMSNorm + SwiGLU + LayerScale.

    ResBlock above is x + W2(SiLU(W1(LN(x)))) with expansion 1 -- inner dim
    equals outer dim. That is the one clearly dated thing about it: every
    modern equivalent widens the inner dim (a transformer FFN is this block at
    4x). Three changes, all standard:

      RMSNorm   -- no centering, cheaper, same or better than LayerNorm
      SwiGLU    -- W_down(SiLU(W_gate x) * W_up x), Shazeer 2020. Inner dim is
                   2/3 * expansion * dim so the gated block costs the same
                   parameters an ungated `expansion`x FFN would.
      LayerScale-- learnable per-channel scale on the residual branch (CaiT).
                   Same trick this file already uses scalar-wise for lstm_gate.

    The shape follows SimBa (Lee et al. 2024), which exists because plain MLP
    trunks in RL fail to gain from added parameters -- the exact symptom this
    repo documents ("bigger is not better": 512/512/3 scores under 256/256/1).

    NOT SUBMITTABLE: different tensor names and shapes from dm_lstm_v1. This is
    a teacher architecture; the legal student is reached by distillation.
    """

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        inner = max(8, int(2 * expansion * dim / 3) // 8 * 8)
        self.norm = nn.RMSNorm(dim)
        # gain 1.0, NOT sqrt(2): the sqrt(2) ReLU-family correction is for a
        # single linear feeding a nonlinearity. Here the two branches are
        # MULTIPLIED, so applying it to both compounds the variance into
        # w_down (~4x too hot) and the policy loss blows up. LLaMA-style
        # SwiGLU uses standard init on all three projections.
        self.w_gate = _init_nobias(nn.Linear(dim, inner, bias=False), 1.0)
        self.w_up = _init_nobias(nn.Linear(dim, inner, bias=False), 1.0)
        self.w_down = _init_nobias(nn.Linear(inner, dim, bias=False), 1.0)
        # 0.1, not CaiT's 1e-4: that is tuned for dozens of blocks, and the
        # competition grid tops out at 3 -- too small here and the block is
        # effectively absent for most of a short run.
        self.scale = nn.Parameter(torch.full((dim,), 0.1))

    def forward(self, x):
        h = self.norm(x)
        h = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x + self.scale * h

TRUNK_BLOCK = "res"

TRUNK_BLOCKS = {"res": ResBlock, "swiglu": SwiGLUBlock}

class Trunk(nn.Module):
    """Input projection + N residual blocks + final LayerNorm."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.proj = _init_linear(nn.Linear(embed_dim, hidden_dim), gain=float(np.sqrt(2)))
        block_cls = TRUNK_BLOCKS[TRUNK_BLOCK]
        self.blocks = nn.ModuleList([block_cls(hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.silu(self.proj(x))
        # THINK_STEPS: run the SAME blocks repeatedly (weight-tied recurrent
        # depth). Depth-6 was worth +19.9 XP at 500M but num_trunk_blocks is
        # capped at 3 for submission -- iterating gives the compute of depth
        # N*blocks while the tensor set, and so the eligibility grid, is
        # unchanged. Adds zero parameters; costs compute per decision only.
        for _ in range(THINK_STEPS):
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
        return feats + self.gate * attn_out

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
        # The LSTM was 62% of the whole model at hidden=256 (two 1024x256
        # matrices, 4 gates each way). A narrower recurrent state with a linear
        # read-back keeps the residual at hidden_dim while cutting that cost:
        # 256 -> 64 takes the block from 526k params to ~99k.
        self.lstm_hidden = LSTM_HIDDEN or hidden_dim
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=self.lstm_hidden, num_layers=1)
        self.lstm_out = (nn.Linear(self.lstm_hidden, hidden_dim, bias=False)
                         if self.lstm_hidden != hidden_dim else None)
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
        self.item_mechanics = bool(ITEM_MECHANICS)
        if self.item_mechanics:
            if not self.equivariant_head:
                raise ValueError("item mechanics requires the shared item-action head")
            from item_mechanics import ItemMechanics, FEATURE_DIM
            self.mechanics_features = ItemMechanics()
            self.mechanics_dim = FEATURE_DIM
            # Adding the residual must not shift initialization/sampling RNG
            # for the unchanged parent network in paired experiments.
            with torch.random.fork_rng(devices=[]):
                self.mechanics_global = nn.Linear(3 * FEATURE_DIM, embed_dim)
                self.mechanics_key = nn.Linear(FEATURE_DIM, PTR_DIM, bias=False)
                nn.init.zeros_(self.mechanics_global.weight)
                nn.init.zeros_(self.mechanics_global.bias)
                nn.init.zeros_(self.mechanics_key.weight)
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
            # Retrofit mode. Replacing the positional bag/market rows on a
            # TRAINED policy throws away everything those rows learned: the
            # pointer starts near-uniform, measured at -173 XP, which 500M
            # steps did not repay. As a gated residual the policy is
            # bit-identical at init (gate=0) and the shared scorer mixes in
            # only as it earns gradient -- the same argument lstm_gate makes
            # for bolting on memory. From scratch leave this OFF: there are no
            # trained positional rows worth keeping and the pure pointer is the
            # stronger inductive bias (+48 XP).
            self.ptr_residual = PTR_RESIDUAL
            if self.ptr_residual:
                self.ptr_gate = nn.Parameter(torch.zeros(1))
        # ── Stat-scoring head ─────────────────────────────────────────────
        # Same argument as the equivariant head, one step further. The six stat
        # actions each own a row of policy_head, so the value of +1 DEX must be
        # learned separately from the value of +1 VIT, even though both are
        # "spend a point, move one avoid-probability".
        #
        # Stats are NOT interchangeable, so weights cannot be shared outright.
        # Instead each stat gets a small learned identity vector and ONE shared
        # scorer maps (identity, current value, level-relative headroom) to a
        # logit. Every upgrade then trains the same scorer, and the engine cliff
        # -- P(avoid) = min(1, stat/level) from dmfast_internal.h:184-192, so
        # points past your level are worth exactly zero -- becomes expressible
        # once instead of six separate times.
        self.stat_head = STAT_HEAD
        if self.stat_head:
            self.stat_id = nn.Parameter(torch.randn(6, PTR_DIM) * 0.02)
            self.stat_score = nn.Sequential(
                nn.Linear(PTR_DIM + 2, PTR_DIM), nn.SiLU(),
                nn.Linear(PTR_DIM, PTR_DIM))
            self.stat_query = _init_linear(nn.Linear(hidden_dim, PTR_DIM), gain=0.01)
            self.stat_bias = nn.Parameter(torch.zeros(1))
            self.ptr_dim = PTR_DIM
        self.quantile_value = QUANTILE_VALUE
        # Asymmetric critic: a residual value term from the latent plus the
        # engine's peek at the next dice (explore/attack/flee outcomes). Last
        # layer inits to 0 so the critic starts exactly as the plain one.
        self.priv_critic = PRIV_CRITIC
        if self.priv_critic:
            self.priv_mlp = nn.Sequential(
                nn.Linear(hidden_dim + PRIV_DIM, hidden_dim), nn.SiLU(),
                _init_linear(nn.Linear(hidden_dim, 1), gain=1e-3))
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
        if self.memory_type == "transformer":
            return self.mem.init_state(n_envs, device)
        H = getattr(self, "lstm_hidden", self.hidden_dim)
        return (torch.zeros(1, n_envs, H, device=device),
                torch.zeros(1, n_envs, H, device=device))

    @staticmethod
    def _stat_ctx(obs):
        """(current stat / 31, level headroom) for the six stats.

        headroom = clamp((level - stat) / level, 0, 1): 0 once the stat has
        reached the adventurer's level, which is exactly where further points
        in it stop buying any avoid-probability."""
        stats = obs[..., 7:13] * 31.0
        xp = obs[..., 3] * 32767.0
        lvl = torch.clamp(torch.floor(torch.sqrt(torch.clamp(xp, min=0.0))), min=1.0)
        head = torch.clamp((lvl.unsqueeze(-1) - stats) / lvl.unsqueeze(-1), 0.0, 1.0)
        return obs[..., 7:13], head

    def _features(self, obs):
        return self.input_trunk(self._encode(obs))

    def _encode(self, obs, return_items=False):
        """The encoder under bf16 autocast (flash attention instead of the
        fp32 cutlass kernel, tensor-core GEMMs); outputs come back fp32 so the
        LSTM, heads and losses are untouched. -20% per minibatch, and the
        rollout and the update share the path, so the PPO ratio is consistent."""
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=BF16_ENC and obs.is_cuda):
            r = self.encoder(obs, return_items=return_items)
        if self.item_mechanics:
            mechanical = self.mechanics_features(obs)
            pooled = torch.cat((mechanical[..., :8, :].mean(-2),
                                mechanical[..., 8:23, :].mean(-2),
                                mechanical[..., 23:48, :].amax(-2)), -1)
            correction = self.mechanics_global(pooled)
            if return_items:
                return r[0].float() + correction, torch.cat((r[1].float(), mechanical), -1)
            return r.float() + correction
        if return_items:
            return r[0].float(), r[1].float()
        return r.float()

    # Action-index layout (57): base[0:11] bag[11:26] market[26:51] stats[51:57]
    # Item-index layout (48):   equip[0:8]  bag[8:23]  market[23:48]
    BAG_A0, BAG_A1, MKT_A0, MKT_A1 = 11, 26, 26, 51
    STAT_A0 = 51
    BAG_I0, BAG_I1, MKT_I0, MKT_I1 = 8, 23, 23, 48

    def _logits(self, out, per_item, action_mask, stat_ctx=None):
        """Masked action logits. With the equivariant head on, the bag and
        market slices are replaced by shared-scorer scores; everything else
        keeps its positional logit."""
        logits = self.policy_head(out)
        if self.equivariant_head and per_item is not None:
            if self.item_mechanics:
                keys = (self.item_key(per_item[..., :-self.mechanics_dim])
                        + self.mechanics_key(per_item[..., -self.mechanics_dim:]))
            else:
                keys = self.item_key(per_item)                   # (..., 48, P)
            scale = self.ptr_dim ** -0.5
            bag = (keys[..., self.BAG_I0:self.BAG_I1, :]
                   * self.bag_query(out).unsqueeze(-2)).sum(-1) * scale + self.bag_bias
            mkt = (keys[..., self.MKT_I0:self.MKT_I1, :]
                   * self.market_query(out).unsqueeze(-2)).sum(-1) * scale + self.market_bias
            if self.ptr_residual:
                bag = logits[..., self.BAG_A0:self.BAG_A1] + self.ptr_gate * bag
                mkt = logits[..., self.MKT_A0:self.MKT_A1] + self.ptr_gate * mkt
            # cat rather than index-assign: keeps the graph functional and
            # avoids an in-place write that torch.compile would have to break on.
            logits = torch.cat([logits[..., :self.BAG_A0], bag, mkt,
                                logits[..., self.MKT_A1:]], dim=-1)
        if self.stat_head and stat_ctx is not None:
            cur, head = stat_ctx                          # (..., 6) each
            ident = self.stat_id.expand(*cur.shape[:-1], 6, self.ptr_dim)
            key = self.stat_score(torch.cat(
                [ident, cur.unsqueeze(-1), head.unsqueeze(-1)], dim=-1))
            sc = ((key * self.stat_query(out).unsqueeze(-2)).sum(-1)
                  * (self.ptr_dim ** -0.5) + self.stat_bias)
            logits = torch.cat([logits[..., :self.STAT_A0], sc], dim=-1)
        return logits.masked_fill(~action_mask, mask_fill_value(logits.dtype))

    def _value(self, x, priv=None):
        """Scalar value from the head; quantile mode stashes the full set in
        self._vq for the quantile-regression loss (mean is used for GAE).
        `priv` (..., PRIV_DIM) is the critic-only peek; None (eval) drops the
        privileged term, which is fine because eval never reads the value."""
        v = self.value_head(x)
        if self.quantile_value:
            self._vq = v
            return v.mean(-1)
        v = v.squeeze(-1)
        if priv is not None and getattr(self, "priv_critic", False):
            v = v + self.priv_mlp(torch.cat([x, priv.to(x.dtype)], dim=-1)).squeeze(-1)
        return v

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
            if self.lstm_out is not None:
                out = self.lstm_out(out)
        return out.to(inp.dtype), new_state

    def step(self, obs, action_mask, state, priv=None):
        """Single-step inference: returns (logits, value, new_state).

        Caller is responsible for zeroing `state` on done BEFORE the call so the
        new episode's first obs sees a clean LSTM state.
        """
        enc, per_item = self._encode(obs, return_items=True)
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
            #
            # --lstm-grad removes the detach, so the encoder gets gradient FROM the
            # recurrent path and can learn to emit features worth remembering
            # rather than only features good for the current step. Riskier early
            # (that is why it is off by default); the point of a warm start is that
            # the representation is already good, so there is nothing left to
            # corrupt.
            lstm_out, new_state = self._run_lstm(
                feats if LSTM_GRAD else feats.detach(), state)
            # Gated residual: out = feats + gate * lstm_out. gate inits to 0 so the
            # policy starts as the feed-forward MLP and the LSTM is only added in
            # if/when it earns positive gradient through the gate.
            out = (feats + self.lstm_gate * lstm_out).squeeze(0)
        logits = self._logits(out, per_item, action_mask,
                              self._stat_ctx(obs) if self.stat_head else None)
        if self.decoupled_value:
            value = self._value(self.value_trunk(enc), priv)
        else:
            value = self._value(out, priv)
        return logits, value, new_state

    def get_action_and_value(self, obs, action_mask, state, priv=None):
        logits, value, new_state = self.step(obs, action_mask, state, priv)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, new_state

    def evaluate_sequence(self, obs_seq, mask_seq, action_seq, done_seq, start_state,
                          return_logits=False, priv_seq=None):
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
        enc, per_item = self._encode(obs_seq.reshape(T * N, -1), return_items=True)
        per_item = per_item.view(T, N, per_item.shape[-2], per_item.shape[-1])
        feats = self.input_trunk(enc).view(T, N, -1)
        if self.decoupled_value:
            dvalues = self._value(self.value_trunk(enc), priv_seq).view(T, N)
        if self.memory_type == "transformer":
            out = self.mem.forward_seq(feats, start_state, done_seq)  # (T,N,H)
            logits = self._logits(out, per_item, mask_seq,
                                  self._stat_ctx(obs_seq) if self.stat_head else None)
            values = dvalues if self.decoupled_value else self._value(out, priv_seq)
            dist = Categorical(logits=logits)
            if return_logits:
                return dist.log_prob(action_seq), dist.entropy(), values, logits
            return dist.log_prob(action_seq), dist.entropy(), values
        if not self.use_lstm:
            out = feats
            logits = self._logits(out, per_item, mask_seq,
                                  self._stat_ctx(obs_seq) if self.stat_head else None)
            values = dvalues if self.decoupled_value else self._value(out, priv_seq)
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
        logits = self._logits(out, per_item, mask_seq,                 # (T, N, A)
                              self._stat_ctx(obs_seq) if self.stat_head else None)
        values = dvalues if self.decoupled_value else self._value(out, priv_seq)  # (T, N)
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

def _item_tf_forward(tf, per_item, ctx, beast, slot, afford):
    """Free function so torch.compile has something to wrap that is NOT the
    module. Compiling the module directly returns an OptimizedModule; storing
    it anywhere leaves ItemTransformer reachable by two paths (as the
    registered submodule and through the wrapper's _orig_mod) and Dynamo
    refuses: "UnspecializedNNModuleVariable(ItemTransformer) is already
    tracked for mutation". Wrapping a plain function keeps exactly one path to
    the module and leaves state_dict keys untouched, so a compiled run and an
    uncompiled one load each other's checkpoints."""
    return tf(per_item, ctx, beast, slot, afford)

class _AttnBlock(nn.Module):
    """Pre-norm block on F.scaled_dot_product_attention.

    nn.TransformerEncoderLayer turns its own fused path OFF under
    norm_first=True (it says so at construction), so it runs the slow generic
    kernel. Calling SDPA directly gets the fused/flash kernel back. The
    feed-forward is 2x, not the stock 4x: with 50 tokens the block is
    projection-bound, and 4x bought nothing measurable.
    """

    def __init__(self, d, heads, ff_mult=2):
        super().__init__()
        assert d % heads == 0, f"d={d} not divisible by heads={heads}"
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                nn.Linear(ff_mult * d, d))

    def forward(self, x, bias=None):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        shape = (B, N, self.h, self.dh)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.n2(x))

class ItemTransformer(nn.Module):
    """Self-attention over every item the adventurer can see at once.

    The flat encoder embeds each of the 48 items with shared weights but the
    items never MEET: bag and market are pooled separately, so nothing in the
    network can compute "is this market chestpiece better than the one I am
    wearing". That comparison is the game's central decision, and right now it
    has to be reconstructed downstream from two summaries that were taken
    independently.

    Here all 48 become tokens of one sequence -- equipped(8), bag(15),
    market(25) -- plus two context tokens so an item can be judged against WHO
    IS WEARING IT and WHAT IT IS FIGHTING rather than in the abstract. After L
    attention layers every item embedding is relative to the whole offer.

    Provenance and role are added, not baked into separate weights: one
    embedding says equipped/bag/market, another says which of the 8 body slots
    (bag and market share a single "unplaced" id, because their positions are
    arbitrary -- that is the same permutation argument the equivariant head
    won +48 XP on, applied to the input side).
    """
    N_EQUIP, N_BAG, N_MARKET = 8, 15, 25
    N_ITEMS = N_EQUIP + N_BAG + N_MARKET          # 48
    T_EQUIP, T_BAG, T_MARKET, T_CTX, T_BEAST = range(5)
    I_CTX, I_BEAST = N_ITEMS, N_ITEMS + 1         # context tokens sit last

    def __init__(self, item_out, ctx_dim, beast_dim, d=None, layers=None, heads=None):
        super().__init__()
        d = d or ITEM_TF_DIM
        self.d = d
        self.item_in = nn.Linear(item_out, d)
        self.ctx_in = nn.Linear(ctx_dim, d)
        self.beast_in = nn.Linear(beast_dim, d)
        self.prov_emb = nn.Embedding(5, d)
        self.role_emb = nn.Embedding(9, d)         # 0-7 body slots, 8 = unplaced
        # Measured on canon-itf-1b: the identity embeddings (prov+role) carry
        # 1.43x the activation norm of the item content they are added to
        # (market items 1.71x). So every token is more "which group am I in"
        # than "which item am I", q.k is near-constant within a group, and the
        # attention comes out uniform and phase-invariant -- exactly what the
        # attention maps showed. These two knobs rebalance it: normalise the
        # content up, and/or scale the identity down.
        self.content_norm = nn.LayerNorm(d) if ITEM_TF_CONTENT_NORM else None
        self.id_scale = ITEM_TF_ID_SCALE
        # Same-slot prior. The decision that matters is "is this better than
        # what is in THAT body slot", and attention has to discover that
        # relation from scratch. State it instead: one learned scalar added to
        # the attention logit whenever two tokens share a wearable slot type.
        # Same move the pointer head made on the output side -- give the
        # structure, let it learn the magnitude. Init 0 so it starts inert.
        self.slot_bias = nn.Parameter(torch.zeros(1)) if ITEM_TF_SLOT_BIAS else None
        # Affordability. Price is NOT in the observation -- it is a function of
        # tier and charisma that the agent must reconstruct by relating two
        # distant obs regions, and today it learns "can I buy this" only from
        # the action mask, never "is it worth it". Two scalars per token
        # (gold/price, can-afford) make the question local to the item.
        self.afford_in = nn.Linear(2, d) if ITEM_TF_AFFORD else None
        self.tf = nn.Sequential(*[
            _AttnBlock(d, heads or ITEM_TF_HEADS)
            for _ in range(layers or ITEM_TF_LAYERS)])
        self.norm = nn.LayerNorm(d)
        # Back to ITEM_OUT so the equivariant pointer head is untouched -- it
        # now scores items that already know what else is on offer.
        self.to_item_out = nn.Linear(d, item_out)

        prov = torch.full((self.N_ITEMS + 2,), self.T_MARKET, dtype=torch.long)
        prov[:self.N_EQUIP] = self.T_EQUIP
        prov[self.N_EQUIP:self.N_EQUIP + self.N_BAG] = self.T_BAG
        prov[self.I_CTX] = self.T_CTX
        prov[self.I_BEAST] = self.T_BEAST
        role = torch.full((self.N_ITEMS + 2,), 8, dtype=torch.long)
        role[:self.N_EQUIP] = torch.arange(self.N_EQUIP)
        self.register_buffer("prov_ids", prov, persistent=False)
        self.register_buffer("role_ids", role, persistent=False)

    @property
    def out_dim(self):
        # 8 equipped (positional) + bag mean/max + market mean/max + context
        return self.d * (self.N_EQUIP + 4 + 1)

    def forward(self, per_item, ctx, beast, slot=None, afford=None):
        B = per_item.shape[0]
        toks = torch.cat([self.item_in(per_item),
                          self.ctx_in(ctx).unsqueeze(1),
                          self.beast_in(beast).unsqueeze(1)], dim=1)
        if self.afford_in is not None and afford is not None:
            pad = afford.new_zeros(B, 2, afford.shape[-1])       # ctx/beast tokens
            toks = toks + self.afford_in(torch.cat([afford, pad], dim=1))
        if self.content_norm is not None:
            toks = self.content_norm(toks)
        ident = self.prov_emb(self.prov_ids) + self.role_emb(self.role_ids)
        toks = toks + self.id_scale * ident
        bias = None
        if self.slot_bias is not None and slot is not None:
            # slot id is field 3 of the item vector: 1..8 wearable, 0 = empty.
            s = torch.cat([slot, slot.new_zeros(B, 2)], dim=1)      # ctx/beast = 0
            same = (s.unsqueeze(2) == s.unsqueeze(1)) & (s.unsqueeze(2) > 0)
            bias = same.to(toks.dtype).unsqueeze(1) * self.slot_bias
        h = toks
        for blk in self.tf:
            h = blk(h, bias)
        h = self.norm(h)
        items = h[:, :self.N_ITEMS]
        eqp = items[:, :self.N_EQUIP].reshape(B, -1)
        bag = items[:, self.N_EQUIP:self.N_EQUIP + self.N_BAG]
        mkt = items[:, self.N_EQUIP + self.N_BAG:]
        summary = torch.cat([eqp, bag.mean(1), bag.amax(1),
                             mkt.mean(1), mkt.amax(1), h[:, self.I_CTX]], dim=-1)
        return summary, self.to_item_out(items)
