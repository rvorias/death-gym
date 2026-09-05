"""Semantic observation augmentation, derived from the existing 463-dim obs.

The engine already exposes the hard part -- obs[453:463] is a full battle
SIMULATION (survivability, risk, expected damage), which is stronger than any
hand-derived combat formula. What it does not expose is the cheap semantic
scaffolding an MLP would otherwise have to rediscover:

  * level        -- absent entirely; level = floor(sqrt(xp)) and everything
                    in the game scales with it
  * log scaling  -- xp is given as xp/32767, so our whole operating range
                    (0-1500 xp) occupies 0..0.046 of that field
  * ratios       -- hp and max_hp are given separately, never their quotient
  * build shape  -- stats are absolute, never per-level
  * matchup      -- weapon type and beast type are both present, but the
                    cyclic blade/bludgeon/magic x cloth/hide/metal table is not
  * market item stats -- EQUIPPED suffix bonuses are folded into obs[7:13],
                    but a market item's suffix is a raw id, so the net must
                    decode what buying it would grant

NOT SUBMITTABLE: this widens obs beyond 463, which validate_submission pins.

PORTED TO C (2026-09-02): engine/src/dmfast_semantic.c computes the v3 set into
`env.sem` with every obs pack, and train.semantic_features() reads that. This
module is the reference (tests/test_semantic_parity.py) and the fallback for
legacy v1/v2 widths. Change both together.
"""
import os
import numpy as np

OBS_DIM = 463
PHASE, HP, MAXHP, XP, GOLD, UPG, BEAST_HP = 0, 1, 2, 3, 4, 5, 6
STATS = slice(7, 13)            # str dex vit int wis cha  (each /31)
LUCK = 13
EQUIP = slice(14, 86)           # 8 slots x 9
MARKET = slice(221, 446)        # 25 slots x 9
BEAST = slice(446, 453)         # start_hp, tier, level(log), type, sp1..3
SIM = slice(453, 463)           # surv_adv, surv_beast, risk, dmg_beast, dmg_adv, per-slot

# dmfast_internal.h:194 -- suffix id -> stat bonus, in obs stat order
_SUF = {1:(3,0,0,0,0,0), 3:(2,0,0,0,0,1), 4:(0,3,0,0,0,0), 5:(1,1,1,0,0,0),
        6:(0,0,0,3,0,0), 7:(0,0,0,0,3,0), 8:(0,1,2,0,0,0), 9:(2,1,0,0,0,0),
        10:(1,0,0,0,1,1), 11:(0,0,1,1,0,1), 12:(0,0,0,2,1,0), 13:(0,2,0,0,0,1),
        14:(0,1,0,0,2,0), 15:(0,0,0,1,2,0), 2:(0,0,3,0,0,0), 16:(0,0,0,0,0,3)}
SUFFIX_STATS = np.zeros((17, 6), dtype=np.float32)
for k, v in _SUF.items():
    SUFFIX_STATS[k] = v

# Cyclic matchup (README/loot-survivor): rows weapon 1=magic 2=blade 3=bludgeon,
# cols armour 1=cloth 2=hide 3=metal. 1.5 strong / 1.0 fair / 0.5 weak.
_M = np.ones((4, 4), dtype=np.float32)
_M[1, 1], _M[1, 2], _M[1, 3] = 1.0, 0.5, 1.5      # magic: fair cloth, weak hide, strong metal
_M[2, 1], _M[2, 2], _M[2, 3] = 1.5, 1.0, 0.5      # blade: strong cloth, fair hide, weak metal
_M[3, 1], _M[3, 2], _M[3, 3] = 0.5, 1.5, 1.0      # bludgeon: weak cloth, strong hide, fair metal

V1_ONLY = bool(int(__import__("os").environ.get("DM_SEM_V1", "0")))
V1_DIM = 39                       # progression/resources/build/enemy/matchup/combat
DMFAST_HEALTH_PER_VITALITY  = 15  # dmfast_internal.h:92
DMFAST_CHARISMA_POTION_DISC = 2   # dmfast_internal.h:96
V2_DIM = 22                       # economy (6) + per-slot upgrade deltas (16)
# v3 is ON by default (DM_SEM_V3=0 opts out). It changes EXTRA_DIM, so a
# checkpoint written before it recorded obs_extra_dim=61; train.py's resume
# path reads that and switches v3 back off rather than dying on a width
# mismatch.
V3 = os.environ.get("DM_SEM_V3", "1") == "1"
V3_DIM = 27                       # repair(6) + stat(10) + value/gold(6) + luck(5)
# v4 (2026-09-03): the two survivability counts UNCAPPED, log-scaled. Default on;
# DM_SEM_V4=0 rebuilds the 88-wide encoder of an older checkpoint.
V4 = V3 and os.environ.get("DM_SEM_V4", "1") == "1"
V4_DIM = 2
EXTRA_DIM = (V1_DIM if V1_ONLY else V1_DIM + V2_DIM) + (V3_DIM if V3 else 0) + (V4_DIM if V4 else 0)


def _slog(x, scale):
    return np.log1p(np.maximum(x, 0.0)) / np.log1p(scale)


def semantic_features(obs: np.ndarray) -> np.ndarray:
    """(B, 463) -> (B, EXTRA_DIM) float32. Pure function of the observation."""
    b = len(obs)
    hp = obs[:, HP] * 1023.0
    mx = np.maximum(obs[:, MAXHP] * 1023.0, 1.0)
    xp = obs[:, XP] * 32767.0
    gold = obs[:, GOLD] * 511.0
    level = np.floor(np.sqrt(np.maximum(xp, 0.0)))
    lvl_safe = np.maximum(level, 1.0)
    stats = obs[:, STATS] * 31.0                      # (B, 6)

    eq = obs[:, EQUIP].reshape(b, 8, 9)
    mk = obs[:, MARKET].reshape(b, 25, 9)
    wtype = eq[:, 0, 2].astype(np.int32).clip(0, 3)   # weapon slot type
    btype = obs[:, 449].astype(np.int32).clip(0, 3)   # beast armour type
    beast_present = (obs[:, 448] > 0).astype(np.float32)
    beast_hp = obs[:, BEAST_HP] * 1023.0
    beast_lvl = np.expm1(obs[:, 448] * np.log1p(640.0))

    sim = obs[:, SIM]
    dmg_to_beast = np.maximum(sim[:, 3], 1e-3)
    dmg_to_adv = np.maximum(sim[:, 4], 1e-3)

    # best suffix on offer in the market, as stat bonuses (the net otherwise
    # sees a raw suffix id and must decode it)
    msuf = mk[:, :, 4].astype(np.int32).clip(0, 16)
    mbonus = SUFFIX_STATS[msuf]                        # (B, 25, 6)
    occupied = (mk[:, :, 0] > 0)[:, :, None]
    mbest = (mbonus * occupied).max(axis=1)            # (B, 6)

    f = np.empty((b, EXTRA_DIM), dtype=np.float32)
    i = 0
    def put(col):
        nonlocal i
        f[:, i] = col; i += 1
    # progression (5)
    put(_slog(level, 100)); put(_slog(xp, 3000)); put(_slog(gold, 500))
    put(np.clip((xp - level**2) / np.maximum(2*level + 1, 1), 0, 1))   # xp into level
    put(_slog(obs[:, UPG] * 15.0, 15))
    # resources (4)
    put(hp / mx); put(1.0 - hp / mx); put(_slog(hp, 1023)); put(_slog(mx, 1023))
    # build composition (6)
    for j in range(6):
        put(np.clip(stats[:, j] / lvl_safe, 0, 3))
    # enemy, relative (5)
    put(beast_present)
    put(np.clip(beast_lvl / lvl_safe, 0, 5))
    put(_slog(beast_lvl, 640))
    put(np.clip(beast_hp / mx, 0, 3))
    put(np.clip(beast_hp / np.maximum(hp, 1.0), 0, 5))
    # matchup (5)
    mult = _M[wtype, btype] * beast_present
    put(mult)
    put((mult > 1.2).astype(np.float32))
    put((mult < 0.8).astype(np.float32))
    put(np.clip(eq[:, 0, 1], 0, 1))                    # weapon tier (already inverted)
    put(np.clip(obs[:, 447], 0, 1))                    # beast tier
    # combat resolution (8)
    hits_to_kill = beast_hp / dmg_to_beast
    hits_to_die = hp / dmg_to_adv
    put(np.clip(hits_to_kill / 20.0, 0, 1))
    put(np.clip(hits_to_die / 20.0, 0, 1))
    put(np.clip(hits_to_die / np.maximum(hits_to_kill, 1e-3) / 5.0, 0, 1))
    put((hits_to_die > hits_to_kill).astype(np.float32) * beast_present)
    put(np.clip(dmg_to_adv / mx, 0, 1))
    put(np.clip(dmg_to_beast / np.maximum(beast_hp, 1.0), 0, 1))
    put((dmg_to_adv >= hp).astype(np.float32) * beast_present)   # dies to one hit
    put((dmg_to_beast >= beast_hp).astype(np.float32) * beast_present)  # kills in one
    # market opportunity (6)
    for j in range(6):
        put(mbest[:, j] / 3.0)
    if V1_ONLY:
        assert i == EXTRA_DIM, (i, EXTRA_DIM)
        return f
    # --- v2 -------------------------------------------------------------
    # ECONOMY (6). Prices are computable but absent from the observation, so
    # the agent can currently reason about affordability only via the action
    # mask -- never about VALUE. Formulas: ex_potion_cost / ex_compute_market_prices.
    cha = stats[:, 5]
    potion = np.maximum(level - cha * 2.0, 1.0)                 # CHARISMA_POTION_DISC=2
    put(_slog(potion, 50))
    put(np.clip(gold / potion, 0, 20) / 20.0)                   # potions affordable
    put((gold >= potion).astype(np.float32))
    mtier6 = mk[:, :, 1] * 5.0                                  # (6-tier), tier is inverted
    mprice = np.maximum(mtier6 * 4.0 - cha[:, None] * 1.0, 1.0)  # ITEM_DISC=1, TIER_PRICE=4
    occ = mk[:, :, 0] > 0
    afford = occ & (mprice <= gold[:, None])
    put(np.clip(afford.sum(axis=1) / 25.0, 0, 1))               # affordable items on offer
    put(np.clip(np.where(occ, mprice, 1e9).min(axis=1) / np.maximum(gold, 1.0), 0, 3) / 3.0)
    put(_slog(cha, 31))

    # PER-SLOT UPGRADE DELTAS (16). The item blocks are 432 of 463 dims and the
    # buy/equip decisions require a 25-way and 15-way comparison against the
    # equipped slot -- a scan an MLP must do inside a linear projection. Give
    # it the answer: for each of the 8 equipment slots, the best quality gain
    # available from the market and from the bag. Quality = tier + greatness,
    # both already normalised in the item vector.
    eq_slot = eq[:, :, 3].astype(np.int32)                      # (B,8) slot ids 1..8
    eq_q = eq[:, :, 1] + eq[:, :, 8]                            # tier + greatness
    bag = obs[:, 86:221].reshape(b, 15, 9)
    for src, n_slots in ((mk, 25), (bag, 15)):
        s_slot = src[:, :, 3].astype(np.int32)
        s_q = src[:, :, 1] + src[:, :, 8]
        s_occ = src[:, :, 0] > 0
        for k in range(8):
            # equipped item occupying equipment index k, and its slot id
            tgt = eq_slot[:, k]                                  # (B,)
            same = (s_slot == tgt[:, None]) & s_occ & (tgt[:, None] > 0)
            gain = np.where(same, s_q - eq_q[:, k][:, None], -1.0).max(axis=1)
            put(np.clip(gain, -1.0, 1.0))
    if not V3:
        assert i == EXTRA_DIM, (i, EXTRA_DIM)
        return f

    # --- v3 --------------------------------------------------------------
    # REPAIR (3). Measured on 204,800 states, three v1 combat features carry no
    # signal at all: idx 26 hits_to_die/20 pins at 1.000, idx 31 "dies to one
    # hit" fires 0.0%, idx 32 "kills in one" is CONSTANT ZERO. Cause: the SIM
    # block is on its own scale (dmg_to_beast mean 0.303, surv scores 0-10),
    # NOT in HP units, so beast_hp / dmg_to_beast computes ~400 and every
    # absolute comparison saturates. The engine's own survivability scores are
    # the live signal those features were trying to reconstruct. Repaired here
    # rather than in place: v1 is frozen because 13 existing checkpoints have
    # already learned to ignore those constants.
    # The engine ALREADY computes both quantities (dmfast_combat.c:459,466):
    #     survivability_beast      = beast_health / average_damage_to_beast  = hits to KILL
    #     survivability_adventurer = adv_health   / average_damage_to_adv    = hits SURVIVED
    # v1 recomputed them from sim[3]/sim[4], which are on a different scale --
    # giving hits_to_kill ~400 and pinning all three features. Use the engine's.
    sa, sb, rk = sim[:, 0], sim[:, 1], sim[:, 2]
    put(np.clip(sa / 10.0, 0, 1) * beast_present)                # hits I survive
    put(np.clip(sb / 10.0, 0, 1) * beast_present)                # hits to kill it
    put(np.clip(rk / 5.0, 0, 1) * beast_present)                 # engine's own risk
    put(((sb <= 1.0) & (beast_present > 0)).astype(np.float32))  # KILLS IN ONE
    put(((sa <= 1.0) & (beast_present > 0)).astype(np.float32))  # dies to one hit
    put(np.clip((sa - sb) / 20.0 + 0.5, 0, 1) * beast_present)   # who wins the race

    # STAT MARGINALS (10). dmfast_internal.h:184-192 -- flee (dex), obstacle
    # dodge (int) and ambush-avoid (wis) all resolve as
    #     stat >= level            -> always avoid
    #     stat >  level*rnd/255    -> otherwise,  rnd ~ U[0,255]
    # i.e. P(avoid) = min(1, stat/level), with a HARD CLIFF: once a stat
    # reaches the adventurer's level, further points in it are worth exactly
    # zero. That threshold is nonlinear in the raw stat value the net is given,
    # so it currently has to be inferred; here it is handed over directly.
    for j in (1, 3, 4):                       # dex, int, wis
        p_av = np.clip(stats[:, j] / lvl_safe, 0.0, 1.0)
        put(p_av)                                            # current P(avoid)
        put(((stats[:, j] < level) / lvl_safe).astype(np.float32))  # marginal of +1
    # VIT: +15 max HP per point, so the marginal is in HITS survived, not HP.
    put(np.clip(DMFAST_HEALTH_PER_VITALITY / mx, 0, 1))
    # gate on beast_present: dmg_to_adv floors at 1e-3 out of combat, which
    # would saturate this at 1 everywhere and make it a constant.
    put(np.clip(DMFAST_HEALTH_PER_VITALITY / np.maximum(dmg_to_adv, 1e-3) / 20.0,
                0, 1) * beast_present)
    # STR: +10% damage per point -> marginal is in hits-to-kill saved.
    hk = beast_hp / dmg_to_beast
    hk2 = beast_hp / np.maximum(dmg_to_beast * 1.10, 1e-3)
    put(np.clip((hk - hk2) / 5.0, 0, 1) * beast_present)
    # CHA: price discounts, as a FRACTION of what is currently paid.
    put(np.clip(DMFAST_CHARISMA_POTION_DISC / potion, 0, 1))

    # VALUE PER GOLD (6). v2 gives price and quality-delta separately but never
    # their ratio -- yet buying is fundamentally a value-per-gold comparison,
    # and the net would have to divide two of its own inputs to get it.
    mq = mk[:, :, 1] + mk[:, :, 8]                      # tier + greatness
    eq_slot2 = eq[:, :, 3].astype(np.int32)
    eq_q2 = eq[:, :, 1] + eq[:, :, 8]
    s_slot2 = mk[:, :, 3].astype(np.int32)
    best_eq = np.zeros_like(mq)
    for k in range(8):
        same = (s_slot2 == eq_slot2[:, k][:, None]) & (eq_slot2[:, k][:, None] > 0)
        best_eq = np.where(same, eq_q2[:, k][:, None], best_eq)
    gain = np.where(occ, mq - best_eq, -1.0)            # quality gain per slot
    vpg = np.where(occ & (mprice > 0), gain / np.maximum(mprice, 1.0), -1.0)
    aff = occ & (mprice <= gold[:, None])
    put(np.clip(vpg.max(axis=1), -1, 1))                       # best value/gold on offer
    put(np.clip(np.where(aff, vpg, -1.0).max(axis=1), -1, 1))  # best AFFORDABLE value/gold
    put(np.clip((gain > 0).sum(axis=1) / 25.0, 0, 1))          # upgrades on offer
    put(np.clip((aff & (gain > 0)).sum(axis=1) / 25.0, 0, 1))  # affordable upgrades
    put(np.clip(np.where(aff & (gain > 0), mprice, 0.0).max(axis=1)
                / np.maximum(gold, 1.0), 0, 1))                # priciest affordable upgrade
    put(np.clip(gain.max(axis=1), -1, 1))                      # best raw quality gain

    # LUCK / CRIT (5). dmfast_inventory_transition.c:66 --
    #     luck = neck_greatness + ring_greatness + SUM(bag jewelry greatness)
    # and dmfast_combat.c:445 -- crit_chance = luck/100, where a crit adds a
    # full extra damage roll. Jewelry sits in DEDICATED slots (6 neck, 7 ring),
    # so it costs no armour slot -- only gold. Bag items never gain xp, so a
    # levelled jewel parked in the bag keeps its greatness forever.
    # Measured: the champion has luck 0.00 at EVERY level; small1b reaches 45.
    # The net is given raw luck at obs[13] but nothing about where it comes
    # from or that any is available, so it cannot act on the mechanic.
    jew_slot = lambda blk: (np.round(blk[:, :, 3]) >= 7) & (blk[:, :, 0] > 0)
    eq_j = jew_slot(eq); bag_j = jew_slot(bag)
    put(np.clip(obs[:, 13] * 100.0 / 50.0, 0, 1))              # current crit chance
    put(np.clip((eq[:, :, 8] * 20.0 * eq_j).sum(1) / 40.0, 0, 1))    # equipped jewel greatness
    put(np.clip((bag[:, :, 8] * 20.0 * bag_j).sum(1) / 40.0, 0, 1))  # bagged jewel greatness
    put(np.clip(jew_slot(mk).sum(1) / 4.0, 0, 1))              # jewellery on offer now
    put((eq_j.sum(1) < 2).astype(np.float32))                  # a jewellery slot is EMPTY
    if not V4:
        assert i == EXTRA_DIM, (i, EXTRA_DIM)
        return f

    # --- v4 --------------------------------------------------------------
    # The sim block clips both survivability counts at 10 (dmfast_combat.c:508),
    # so "I survive 16, it survives 10" and the reverse look identical there.
    # The raw expected damages (obs 456/457, /100) give the engine's own
    # uncapped ratios; log-scaled so 12 vs 40 still differ.
    hs = hp / np.maximum(sim[:, 4] * 100.0, 1e-3)
    hk = beast_hp / np.maximum(sim[:, 3] * 100.0, 1e-3)
    put(np.clip(np.log1p(hs) / np.log1p(200.0), 0, 1) * beast_present)
    put(np.clip(np.log1p(hk) / np.log1p(200.0), 0, 1) * beast_present)

    assert i == EXTRA_DIM, (i, EXTRA_DIM)
    return f
