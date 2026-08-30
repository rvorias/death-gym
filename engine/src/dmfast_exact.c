/*
 * dmfast_exact.c — Full-parity batch engine using exact native kernels.
 *
 * This replaces the simplified prototype in dmfast.c with an engine that
 * uses the parity-tested kernel functions (combat, explore, inventory, etc.)
 * to match the Python EngineEnv exactly.
 *
 * Observation layout matches Python EngineEnv (463-dim):
 *   game_phase[1] | adventurer[13] | equipment[72] | bag[135]
 *   | market[225] | beast[7] | sim_stats[10]
 */

#include "dmfast.h"
#include "dmfast_loot_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── Constants ──────────────────────────────────────────────────────── */

enum {
    EX_NUM_STATS              = 7,
    EX_NUM_EQUIPMENT_SLOTS    = 8,
    EX_NUM_BAG_SLOTS          = 15,
    EX_NUM_MARKET_SLOTS       = 25,
    EX_NUM_SPECIAL_FIELDS     = 3,
    EX_NUM_ITEM_FIELDS        = 9,
    EX_ACTION_DIM             = 57,
    EX_OBS_DIM                = 463,
    EX_INFO_DIM               = 9,  /* per-env info fields for PPO logging */
    /* Info field indices */
    EX_INFO_ADVENTURER_LEVEL  = 0,
    EX_INFO_XP                = 1,
    EX_INFO_HEALTH            = 2,
    EX_INFO_GOLD              = 3,
    EX_INFO_BEAST_LEVEL       = 4,
    EX_INFO_BEASTS_KILLED     = 5,
    EX_INFO_ITEMS_BOUGHT      = 6,
    EX_INFO_POTIONS_BOUGHT    = 7,
    EX_INFO_CURRICULUM_START  = 8,  /* 1.0 if episode started from curriculum (T1 weapon or snapshot) */
    EX_PHASE_UPGRADE          = 0,
    EX_PHASE_MARKET           = 1,
    EX_PHASE_DROP             = 2,
    EX_PHASE_EQUIP            = 3,
    EX_PHASE_COMBAT           = 4,
    EX_PHASE_BUY              = 5,
    EX_BASE_DAMAGE_REDUCTION  = 25,
    EX_STARTING_HEALTH        = 100,
    EX_MAX_HEALTH             = 1023,
    EX_MAX_XP                 = 32767,
    EX_MAX_GOLD               = 511,
    EX_MAX_STAT_UPGRADES      = 15,
    EX_MAX_BEAST_HEALTH       = 1023,
    EX_MAX_OBS_COMBAT_LEVEL   = 640,
    EX_HEALTH_PER_VITALITY    = 15,
    EX_POTION_HEALTH_AMOUNT   = 10,
    EX_CHARISMA_POTION_DISC   = 2,
    EX_CHARISMA_ITEM_DISC     = 1,
    EX_MIN_POTION_PRICE       = 1,
    EX_MIN_ITEM_PRICE         = 1,
    EX_MARKET_TIER_PRICE      = 4,
    /* Action indices (match Python ActionVector) */
    EX_ACT_EXPLORE            = 0,
    EX_ACT_ATTACK             = 1,
    EX_ACT_FLEE               = 2,
    EX_ACT_EQUIP              = 3,
    EX_ACT_DROP               = 4,
    EX_ACT_BUY_ITEM           = 5,
    EX_ACT_BUY_POTION         = 6,
    EX_ACT_MACRO_FIGHT        = 7,
    EX_ACT_MACRO_FLEE         = 8,
    EX_ACT_MACRO_EQUIP        = 9,
    EX_ACT_MACRO_EXPLORE      = 10,
    EX_NUM_BASE_ACTIONS       = 11,
    /* Stat indices */
    EX_STAT_STR               = 0,
    EX_STAT_DEX               = 1,
    EX_STAT_VIT               = 2,
    EX_STAT_INT               = 3,
    EX_STAT_WIS               = 4,
    EX_STAT_CHA               = 5,
    EX_STAT_LUCK              = 6,
};

/* ─── Reward configuration ───────────────────────────────────────────── */

typedef struct {
    /* Combat */
    float attack;               /* per attack step (Python default: 0) */
    float attack_over_15;       /* extra per attack when beast_level > 15 (default: 0) */
    float attack_safe_sim;      /* bonus when sim says safe to attack (S049: 0.15) */
    float kill;                 /* base kill reward * beast_level (Python default: 1.0) */
    float kill_over_15;         /* kill reward for beast_level > 15 * beast_level (default: 0) */
    float kill_high_level;      /* bonus for beasts >= beast_high_level_threshold (default: 0) */
    float kill_high_relative;   /* bonus for beasts >= kill_high_relative_ratio * adv_level (default: 0) */
    float kill_high_relative_ratio;
    int32_t beast_high_level_threshold;
    float flee_gain;            /* base flee gain (REWARD_GAIN_FLEE, default: 0) */
    float flee_penalty;         /* base flee penalty (Python default: 0) */
    float flee_low_hp_threshold;/* FLEE_LOW_HP_THRESHOLD (default: 0.30) */
    float flee_hard_bonus;      /* bonus when fleeing hard beast by level ratio (default: 0) */
    float flee_hard_sim;        /* bonus when fleeing and sim says hard (S049: 0.10) */
    float flee_hard_level_ratio;
    float flee_easy_penalty;    /* penalty for fleeing easy beasts (default: 0) */
    float flee_safe_sim_penalty;/* penalty for fleeing when sim says safe (S049: -0.10) */

    /* Economy */
    float buy_potion;           /* per potion bought (S049: 0.90) */
    float potion_low_hp;        /* bonus when hp ratio < potion_low_hp_threshold */
    float potion_high_hp_penalty;
    float potion_low_hp_threshold;
    float potion_high_hp_threshold;
    float buy_item;             /* per item bought/equipped (S049: 0.50) */
    float buy_t1_item;          /* bonus for tier-1 items — one-time per slot (S049: 1.10) */
    float buy_weapon_t1;        /* extra bonus for tier-1 weapon — one-time (S049: 3.50) */
    float buy_quality_delta;    /* bonus scaled by quality improvement (S049: 0.06) */
    float buy_asset_bank_delta; /* bonus scaled by armor rating improvement (S049: 0.045) */
    int32_t buy_asset_bank_delta_max_level; /* level cap for asset bank delta (default: 22) */
    float explore;              /* per explore step (default: 0) */
    float stat_upgrade;         /* per stat upgrade (default: 0) */
    float market_no_buy_penalty;/* penalty for exploring without buying (default: 0) */
    int32_t market_no_buy_max_level;  /* MARKET_NO_BUY_MAX_LEVEL (default: 20) */
    int32_t market_no_buy_min_gold;   /* MARKET_NO_BUY_MIN_GOLD (default: 10) */
    int32_t market_no_buy_min_options;/* MARKET_NO_BUY_MIN_OPTIONS (default: 1) */
    float market_no_buy_min_hp_ratio; /* MARKET_NO_BUY_MIN_HP_RATIO (default: 0.65) */
    float switch_item;          /* base reward for equipping item from bag (default: 0) */
    float switch_survivability; /* bonus * survivability improvement (S049: 0.50) */
    float switch_surv_score_clip; /* clip composite score delta (Python default: 2.0) */
    int32_t switch_surv_min_level; /* min level for switch survivability (default: 8) */
    int32_t switch_surv_max_level; /* max level for switch survivability (default: 34) */
    float drop_item;            /* reward for dropping item (default: 0) */

    /* Progression */
    float level_up;             /* per level gained (default: 0) */
    float xp_shaping_scale;    /* multiplier for diff_xp shaping (default: 0) */
    float death_penalty;        /* penalty on death (Python default: 0, no death penalty) */

    /* Kill milestones (one-time) */
    float milestone_kill_1;
    float milestone_kill_3;
    float milestone_kill_10;
    float milestone_kill_20;
    float milestone_kill_30;

    /* Level milestones (one-time) */
    float milestone_level_8;
    float milestone_level_12;
    float milestone_level_16;
    float milestone_level_20;
    float milestone_level_25;
    float milestone_level_35;

    /* Phase scaling: table-based lookup matching Python EngineEnv exactly.
     * 5 entries: [very_early, early, mid, late, endgame]
     * Level boundaries: <6, <16, <35, <70, >=70 */
    float phase_table_economy[5];
    float phase_table_stat[5];
    float phase_table_attack[5];
    float phase_table_kill[5];
    float phase_table_high_kill[5];
    float phase_table_flee[5];
    float phase_table_potion[5];
    float phase_table_xp[5];
    /* Global scale multipliers (multiply the table values) */
    float phase_scale_economy;
    float phase_scale_stat;
    float phase_scale_attack;
    float phase_scale_kill;
    float phase_scale_high_kill;
    float phase_scale_flee;
    float phase_scale_potion;
    float phase_scale_xp;

    /* Macro shaping */
    float macro_fight_favorable;
    float macro_fight_kill_resolve;
    float macro_flee_favorable;
    float macro_flee_hard_resolve;

    /* Sim-risk thresholds for rewards (Python: 0.95 / 1.35) */
    float sim_risk_safe_threshold;  /* below this = safe (reward attack, penalize flee) */
    float sim_risk_hard_threshold;  /* above this = hard (reward flee) */

    /* Sim-risk mask gate */
    int32_t mask_block_flee_on_safe;
    float mask_block_flee_safe_threshold;
    float mask_block_flee_min_hp_ratio;
    float mask_block_flee_max_beast_level_ratio;
    int32_t mask_block_flee_require_asset_ready; /* ACTION_MASK_BLOCK_FLEE_REQUIRE_ASSET_READY */

    /* ─── Extended reward signals (matching train.py reward_dict) ─── */

    /* Attack context */
    float attack_favorable;             /* REWARD_GAIN_ATTACK_FAVORABLE */
    float attack_favorable_level_ratio; /* ATTACK_FAVORABLE_LEVEL_RATIO (default 1.15) */
    float kill_streak;                  /* REWARD_GAIN_KILL_STREAK (* beasts_killed) */

    /* Flee context */
    float flee_streak_penalty;          /* REWARD_PENALTY_FLEE_STREAK */
    int32_t flee_streak_grace;          /* FLEE_STREAK_GRACE (default 2) */
    float flee_easy_level_ratio;        /* FLEE_EASY_LEVEL_RATIO (default 1.0) */
    float flee_safe_sim_asset_ready_extra; /* REWARD_PENALTY_FLEE_SAFE_SIM_ASSET_READY_EXTRA */
    int32_t flee_safe_sim_asset_ready_min_level;
    int32_t flee_safe_sim_asset_ready_min_items;
    int32_t flee_safe_sim_asset_ready_min_armor_slots;
    float flee_safe_sim_asset_ready_min_hp_ratio;
    float flee_after_swap_penalty;      /* REWARD_PENALTY_FLEE_AFTER_IMPROVING_SWAP */
    float flee_after_swap_min_delta;    /* FLEE_AFTER_SWAP_MIN_SIM_SCORE_DELTA (default 0.15) */
    float flee_phase_target_penalty;    /* REWARD_PENALTY_FLEE_PHASE_TARGET */
    float kill_phase_target_bonus;      /* REWARD_GAIN_KILL_PHASE_TARGET */
    int32_t phase_target_min_resolutions; /* PHASE_TARGET_MIN_RESOLUTIONS (default 4) */

    /* Phase progress */
    float phase_progress;               /* REWARD_GAIN_PHASE_PROGRESS */

    /* Stagnation & level floor */
    float stagnation_pre16;             /* REWARD_PENALTY_STAGNATION_PRE16 */
    int32_t stagnation_pre16_level_thresh;
    int32_t stagnation_pre16_action_thresh;
    float stagnation_pre16_max_penalty;
    float level_floor_penalty;          /* REWARD_PENALTY_LEVEL_FLOOR */
    int32_t level_floor_budgets[4];     /* LEVEL_FLOOR_ACTION_BUDGETS_CSV */
    int32_t level_floor_min_levels[4];  /* LEVEL_FLOOR_MIN_LEVELS_CSV */
    int32_t level_floor_count;
    float level_floor_max_step_penalty;

    /* Early farming */
    float early_farm_penalty;           /* REWARD_PENALTY_EARLY_FARM_LOW_RISK */
    int32_t early_farm_max_level;
    float early_farm_risk_max;
    float early_farm_easy_level_ratio;
    int32_t early_farm_target_items;

    /* Stat choice shaping */
    float stat_match_early_cha;
    float stat_match_level2_cha;
    float stat_penalty_level2_non_cha;
    float stat_match_early_dex_strong;
    float stat_match_mid_cha;
    float stat_match_mid_dex;
    float stat_match_late_vit;
    float stat_match_end_str;
    float stat_match_end_dex;
    float stat_match_end_vit;
    float stat_penalty_low_value_early;

    /* Jewelry/weapon timing */
    float first_weapon_online;
    float first_weapon_early;
    int32_t first_weapon_early_max_action;
    int32_t first_weapon_early_max_level;
    float jewelry_before_weapon;
    float weapon_before_jewelry;
    float early_jewelry_penalty;
    int32_t early_jewelry_min_action;
    int32_t early_jewelry_min_level;
    float jewelry_timing_good;
    float early_armor_slot_fill;
    int32_t early_armor_fill_max_level;
    float new_combat_family;
    float all_combat_families_early;
    int32_t combat_family_diversity_max_level;

    /* Equipment completion */
    float full_equipment;               /* REWARD_ONCE_FULL_EQUIPMENT */
    float necklace_ring_equipped;       /* REWARD_GAIN_NECKLACE_AND_RING_EQUIPPED */

    /* Equipment armor shaping */
    float equipment_armor_delta;
    float equipment_armor_drop;
    float low_equipment_armor;
    int32_t low_equipment_armor_min_level;
    float low_equipment_armor_target_per_level;

    /* Switch quality */
    float switch_quality_delta;         /* REWARD_GAIN_SWITCH_ITEMS_QUALITY_DELTA */
    int32_t switch_quality_min_level;
    int32_t switch_quality_max_level;
    float switch_worse_combat_sim;      /* REWARD_PENALTY_SWITCH_ITEMS_WORSE_COMBAT_SIM */

    /* Item greatness */
    float item_to_level_15;             /* REWARD_ITEM_TO_LEVEL_15 */
    float weapon_t1_level_15;           /* REWARD_WEAPON_T1_LEVEL_15 */

    /* Potion economics */
    float potion_without_gear;
    int32_t potion_without_gear_min_level;
    int32_t potion_without_gear_min_items;
    int32_t potion_without_gear_min_armor;
    float reduce_potion_cost_early;
    float early_potion_cost_above_target;
    float early_potion_cost_state_above_target;
    float early_potion_cost_state_at_below_target;
    int32_t early_potion_level_threshold;
    float early_potion_cost_target;

    /* Buy early target items */
    float buy_early_target_item;
    int32_t buy_early_target_item_max_level;
    int32_t early_target_item_ids[25];
    int32_t early_target_item_count;

    /* Invalid action (configurable) */
    float invalid_action_penalty;       /* default -0.002 */

    /* Macro cap hit */
    float macro_repeat_cap_hit_favorable;

    /* Phase boundaries (configurable) */
    int32_t phase_very_early_end;       /* default 5 */
    int32_t phase_early_end;            /* default 15 */
    int32_t phase_mid_end;              /* default 34 */
    int32_t phase_late_end;             /* default 70 */

    /* XP Subphase system — per-XP-bin multipliers for fine-grained curriculum */
    int32_t xp_subphase_enable;         /* XP_SUBPHASE_ENABLE (0=off, 1=on) */
    int32_t xp_subphase_bounds[10];     /* XP_SUBPHASE_BOUNDS_CSV (up to 10 boundaries → 11 bins) */
    int32_t xp_subphase_count;          /* number of boundaries (default 0) */
    float xp_subphase_flee_targets[11]; /* XP_SUBPHASE_FLEE_TARGETS_CSV (per-bin targets) */
    float xp_subphase_attack_commit_mult[11]; /* XP_SUBPHASE_ATTACK_COMMIT_MULT_CSV */
    float xp_subphase_kill_target_mult[11];   /* XP_SUBPHASE_KILL_TARGET_MULT_CSV */
    float xp_subphase_flee_target_mult[11];   /* XP_SUBPHASE_FLEE_TARGET_MULT_CSV */
    float xp_subphase_flee_easy_penalty_mult[11]; /* XP_SUBPHASE_FLEE_EASY_PENALTY_MULT_CSV */
    float xp_subphase_market_no_buy_mult[11]; /* XP_SUBPHASE_MARKET_NO_BUY_MULT_CSV */
    float xp_subphase_buy_reward_mult[11];    /* XP_SUBPHASE_BUY_REWARD_MULT_CSV */
    float xp_subphase_potion_reward_mult[11]; /* XP_SUBPHASE_POTION_REWARD_MULT_CSV */

    /* Curriculum */
    float curriculum_t1_weapon_prob;         /* probability of starting with T1 weapon (default 0) */
    float curriculum_snapshot_prob;           /* probability of restoring from snapshot on reset (default 0) */
    int32_t curriculum_snapshot_min_xp;       /* min XP to capture snapshot (default 169) */
    int32_t curriculum_snapshot_max_xp;       /* max XP to capture snapshot (default 675) */
    float curriculum_snapshot_min_hp_ratio;   /* min HP ratio to capture snapshot (default 0.40) */
    float max_level;   /* truncate episode at this adventurer level (0=disabled) — armor curriculum */
} DMFastRewardConfig;

static void ex_default_reward_config(DMFastRewardConfig *rc) {
    /* S049 best PPO config: Python EngineEnv defaults + 10 S049 overrides.
     * Python defaults are 0 for most params. Only S049 overrides are non-zero. */
    memset(rc, 0, sizeof(*rc));

    /* S049 overrides (the only non-zero reward signals) */
    rc->attack_safe_sim = 0.15f;       /* REWARD_GAIN_ATTACK_SAFE_SIM */
    rc->kill = 1.0f;                   /* REWARD_GAIN_BEAST_DEAD (Python default=1) */
    rc->flee_safe_sim_penalty = -0.10f; /* REWARD_PENALTY_FLEE_SAFE_SIM */
    rc->flee_hard_sim = 0.10f;         /* REWARD_GAIN_FLEE_HARD_SIM */
    rc->buy_potion = 0.90f;           /* REWARD_GAIN_BUY_POTION (3x = S049 key change) */
    rc->buy_item = 0.50f;             /* REWARD_GAIN_BUY_ITEM_EQUIPMENT */
    rc->buy_t1_item = 1.10f;          /* REWARD_GAIN_BUY_T1_ITEM */
    rc->buy_weapon_t1 = 3.50f;        /* REWARD_GAIN_BUY_WEAPON_T1 */
    rc->buy_quality_delta = 0.06f;    /* REWARD_GAIN_BUY_ITEM_QUALITY_DELTA */
    rc->buy_asset_bank_delta = 0.045f; /* REWARD_GAIN_BUY_ASSET_BANK_DELTA */
    rc->buy_asset_bank_delta_max_level = 22;
    rc->switch_survivability = 0.50f;  /* REWARD_GAIN_SWITCH_ITEMS_IMPROVED_SURVIVABILITY */
    rc->switch_surv_score_clip = 2.0f; /* SWITCH_ITEMS_COMBAT_SIM_SCORE_CLIP */
    rc->switch_surv_min_level = 8;
    rc->switch_surv_max_level = 34;

    /* Non-overridden but used: level ratio thresholds */
    rc->kill_high_relative_ratio = 1.25f;
    rc->beast_high_level_threshold = 30;
    rc->flee_hard_level_ratio = 1.35f;
    rc->potion_low_hp_threshold = 0.65f;
    rc->potion_high_hp_threshold = 0.95f;

    /* Phase scaling: table-based lookup matching Python EngineEnv exactly.
     * 5 entries: [very_early(<6), early(<16), mid(<35), late(<70), endgame(>=70)] */
    float economy[5]   = {1.7f, 1.35f, 1.0f, 0.6f, 0.35f};
    float stat[5]      = {1.8f, 1.4f, 1.1f, 0.8f, 0.6f};
    float attack[5]    = {0.75f, 1.0f, 1.2f, 1.35f, 1.5f};
    float kill_t[5]    = {0.8f, 1.0f, 1.3f, 1.65f, 2.0f};
    float high_kill[5] = {0.5f, 0.8f, 1.2f, 1.7f, 2.2f};
    float flee[5]      = {0.7f, 1.0f, 1.25f, 1.5f, 1.8f};
    float potion[5]    = {0.8f, 1.0f, 1.2f, 1.35f, 1.5f};
    float xp[5]        = {0.8f, 1.0f, 1.1f, 1.25f, 1.4f};
    memcpy(rc->phase_table_economy, economy, sizeof(economy));
    memcpy(rc->phase_table_stat, stat, sizeof(stat));
    memcpy(rc->phase_table_attack, attack, sizeof(attack));
    memcpy(rc->phase_table_kill, kill_t, sizeof(kill_t));
    memcpy(rc->phase_table_high_kill, high_kill, sizeof(high_kill));
    memcpy(rc->phase_table_flee, flee, sizeof(flee));
    memcpy(rc->phase_table_potion, potion, sizeof(potion));
    memcpy(rc->phase_table_xp, xp, sizeof(xp));
    /* Global scale multipliers (all 1.0 = Python default) */
    rc->phase_scale_economy = 1.0f;
    rc->phase_scale_stat = 1.0f;
    rc->phase_scale_attack = 1.0f;
    rc->phase_scale_kill = 1.0f;
    rc->phase_scale_high_kill = 1.0f;
    rc->phase_scale_flee = 1.0f;
    rc->phase_scale_potion = 1.0f;
    rc->phase_scale_xp = 1.0f;

    /* Sim-risk thresholds matching Python EngineEnv exactly */
    rc->sim_risk_safe_threshold = 0.95f;   /* SIM_RISK_SAFE_THRESHOLD */
    rc->sim_risk_hard_threshold = 1.35f;   /* SIM_RISK_HARD_THRESHOLD */

    /* Sim-risk mask gate (enabled by default for S049) */
    rc->mask_block_flee_on_safe = 1;
    rc->mask_block_flee_safe_threshold = 1.20f;
    rc->mask_block_flee_min_hp_ratio = 0.50f;
    rc->mask_block_flee_max_beast_level_ratio = 1.10f;

    /* ─── Extended reward defaults (matching Python engine.py / train.py) ─── */

    /* Attack context */
    rc->attack_favorable_level_ratio = 1.15f;

    /* Flee context */
    rc->flee_streak_grace = 2;
    rc->flee_easy_level_ratio = 1.0f;  /* Python default; C was 0.8 */
    rc->flee_safe_sim_asset_ready_min_level = 10;
    rc->flee_safe_sim_asset_ready_min_items = 5;
    rc->flee_safe_sim_asset_ready_min_armor_slots = 2;
    rc->flee_safe_sim_asset_ready_min_hp_ratio = 0.55f;
    rc->flee_after_swap_min_delta = 0.15f;
    rc->phase_target_min_resolutions = 4;

    /* Stagnation & level floor */
    rc->stagnation_pre16_level_thresh = 16;
    rc->stagnation_pre16_action_thresh = 140;
    rc->stagnation_pre16_max_penalty = 1.5f;
    rc->level_floor_budgets[0] = 160; rc->level_floor_budgets[1] = 240; rc->level_floor_budgets[2] = 340;
    rc->level_floor_min_levels[0] = 14; rc->level_floor_min_levels[1] = 21; rc->level_floor_min_levels[2] = 30;
    rc->level_floor_count = 3;
    rc->level_floor_max_step_penalty = 1.0f;

    /* Early farming */
    rc->early_farm_max_level = 16;
    rc->early_farm_risk_max = 0.95f;
    rc->early_farm_easy_level_ratio = 0.85f;
    rc->early_farm_target_items = 4;

    /* Jewelry/weapon timing defaults */
    rc->first_weapon_early_max_action = 10;
    rc->first_weapon_early_max_level = 3;
    rc->early_jewelry_min_action = 25;
    rc->early_jewelry_min_level = 6;
    rc->early_armor_fill_max_level = 20;
    rc->combat_family_diversity_max_level = 22;

    /* Equipment armor */
    rc->low_equipment_armor_min_level = 8;
    rc->low_equipment_armor_target_per_level = 8.0f;

    /* Switch quality */
    rc->switch_quality_min_level = 8;
    rc->switch_quality_max_level = 36;

    /* Potion economics */
    rc->potion_without_gear_min_level = 8;
    rc->potion_without_gear_min_items = 6;
    rc->potion_without_gear_min_armor = 95;
    rc->early_potion_level_threshold = 20;
    rc->early_potion_cost_target = 1.5f;

    /* Buy target items */
    rc->buy_early_target_item_max_level = 12;
    rc->early_target_item_count = 0;  /* none by default, set via CSV */

    /* Invalid action */
    rc->invalid_action_penalty = -0.002f;

    /* Market no-buy conditions */
    rc->market_no_buy_max_level = 20;
    rc->market_no_buy_min_gold = 10;
    rc->market_no_buy_min_options = 1;
    rc->market_no_buy_min_hp_ratio = 0.65f;

    /* Flee thresholds */
    rc->flee_low_hp_threshold = 0.30f;

    /* Phase boundaries */
    rc->phase_very_early_end = 5;
    rc->phase_early_end = 15;
    rc->phase_mid_end = 34;
    rc->phase_late_end = 70;

    /* Curriculum */
    rc->curriculum_t1_weapon_prob = 0.0f;
    rc->curriculum_snapshot_prob = 0.0f;
    rc->curriculum_snapshot_min_xp = 169;
    rc->curriculum_snapshot_max_xp = 675;
    rc->curriculum_snapshot_min_hp_ratio = 0.40f;
    rc->max_level = 0.0f;

    /* XP subphase: disabled by default, all multipliers 1.0 */
    rc->xp_subphase_enable = 0;
    rc->xp_subphase_count = 0;
    for (int i = 0; i < 11; i++) {
        rc->xp_subphase_flee_targets[i] = 0.15f;  /* neutral default */
        rc->xp_subphase_attack_commit_mult[i] = 1.0f;
        rc->xp_subphase_kill_target_mult[i] = 1.0f;
        rc->xp_subphase_flee_target_mult[i] = 1.0f;
        rc->xp_subphase_flee_easy_penalty_mult[i] = 1.0f;
        rc->xp_subphase_market_no_buy_mult[i] = 1.0f;
        rc->xp_subphase_buy_reward_mult[i] = 1.0f;
        rc->xp_subphase_potion_reward_mult[i] = 1.0f;
    }
}

/* ─── Exact per-environment state ────────────────────────────────────── */

typedef struct {
    uint64_t rng;

    /* Core game state (int32_t to match exact kernels) */
    int32_t phase;
    int32_t adventurer_xp;
    int32_t health;
    int32_t gold;
    int32_t action_count;
    int32_t stat_upgrades_available;
    int32_t item_specials_seed;

    /* Stats: str, dex, vit, int, wis, cha, luck */
    int32_t stats[EX_NUM_STATS];
    int32_t special_stats[EX_NUM_STATS];

    /* Equipment (8 slots) */
    int32_t equipment_ids[EX_NUM_EQUIPMENT_SLOTS];
    int32_t equipment_xp[EX_NUM_EQUIPMENT_SLOTS];
    int32_t equipment_specials[EX_NUM_EQUIPMENT_SLOTS * EX_NUM_SPECIAL_FIELDS];

    /* Bag (15 slots) */
    int32_t bag_ids[EX_NUM_BAG_SLOTS];
    int32_t bag_xp[EX_NUM_BAG_SLOTS];
    int32_t bag_specials[EX_NUM_BAG_SLOTS * EX_NUM_SPECIAL_FIELDS];

    /* Market (25 slots) */
    int32_t market_ids[EX_NUM_MARKET_SLOTS];

    /* Beast */
    int32_t beast_id;
    int32_t beast_health;
    int32_t beast_starting_health;
    int32_t beast_tier;
    int32_t beast_type;
    int32_t beast_level;
    int32_t beast_specials[EX_NUM_SPECIAL_FIELDS];

    /* Seeds */
    uint64_t settings_id;
    uint64_t game_seed;
    uint64_t market_seed;

    /* Buffer for equip tracking (cleared every combat tick + explore) */
    int32_t switch_buffer_slots[EX_NUM_EQUIPMENT_SLOTS];
    /* Slot ids (1-8) of items bought this market session; masks repeat
     * same-slot purchases. Cleared on reset + explore. */
    int32_t buy_buffer_slots[EX_NUM_EQUIPMENT_SLOTS];

    /* Episode tracking */
    uint8_t done;
    int32_t episode_length;
    float episode_return;

    /* Reward tracking state */
    int32_t prev_xp;            /* XP at start of step (for level-up detection) */
    int32_t beasts_killed;      /* cumulative kills this episode */
    int32_t items_bought;       /* cumulative items bought this episode */
    int32_t potions_bought;     /* cumulative potions bought this episode */
    int32_t market_explores;    /* cumulative market explores this episode */
    int32_t prev_level;         /* level at start of step */
    uint8_t milestone_kill_flags; /* bit flags for kill milestones (1,3,10,20,30) */
    uint8_t milestone_level_flags; /* bit flags for level milestones */
    int32_t market_items_available; /* count of affordable items at explore time */

    /* One-time reward flags */
    uint8_t flag_buy_t1_slots;  /* bit per slot (0-7) for one-time T1 buy bonus */
    uint8_t flag_buy_weapon_t1; /* 1 if T1 weapon buy bonus already given */

    /* Cached sim result (from obs packing, used by mask gate + rewards) */
    float cached_sim_risk;      /* survivability_risk: beast_surv / adv_surv */

    /* ─── Extended tracking state ─── */

    /* Flee tracking */
    int32_t flee_streak;            /* consecutive non-hard flees */
    int32_t phase_flees;            /* flees in current reward-phase */
    int32_t phase_kills;            /* kills in current reward-phase */
    int32_t current_phase_idx;      /* 0=very_early..4=endgame */
    int32_t highest_phase_reached;  /* for phase_progress one-time bonus */

    /* Weapon/jewelry timing */
    int32_t first_weapon_action;    /* -1 = not acquired yet */
    int32_t first_weapon_level;
    int32_t first_jewelry_action;   /* -1 = not acquired yet */
    int32_t first_jewelry_level;
    uint8_t flag_early_armor_slots; /* bit per armor slot rewarded */
    uint8_t bought_combat_families; /* bit 0=magic/cloth, 1=blade/hide, 2=bludgeon/metal */
    uint8_t flag_all_combat_families;
    uint8_t flag_full_equipment;
    uint8_t flag_necklace_ring;

    /* Item greatness tracking */
    uint8_t items_reached_level_15; /* bit per equipment slot */

    /* Early target items */
    int32_t early_target_items_bought;
    uint32_t early_target_item_seen[4]; /* bit set for item IDs 0-127 */

    /* Armor tracking */
    float last_equipment_armor_rating;

    /* Swap tracking (for flee-after-swap) */
    float last_swap_sim_delta;
    uint8_t had_swap_this_combat;   /* 1 if equip action done in current combat */

    /* Potion cost tracking */
    float current_potion_cost;

    /* Curriculum tracking */
    uint8_t started_from_curriculum;  /* 1 if T1 weapon or snapshot start */
} DMFastExactEnv;

#define EX_SNAPSHOT_POOL_MAX 16

typedef struct {
    DMFastExactEnv state;  /* full copy of env state */
    int32_t xp;            /* for sorting / dedup */
    int32_t level;
    uint8_t valid;         /* 1 if this slot contains a snapshot */
} DMFastSnapshot;

struct DMFastExactBatch {
    int32_t batch_size;
    int32_t max_steps;
    uint64_t seed_counter;
    DMFastRewardConfig rc;
    DMFastExactEnv *envs;
    float *obs;
    uint8_t *action_mask;
    float *reward;
    uint8_t *terminated;
    uint8_t *truncated;
    int32_t *phase_buf;
    int32_t *step_count;
    float *episode_return;
    float *last_episode_return;
    int32_t *last_episode_length;
    float *last_episode_info; /* (batch_size, EX_INFO_DIM) — terminal metrics of last completed episode */
    float *last_terminal_obs; /* (batch_size, EX_OBS_DIM) — final observation before auto-reset */
    uint8_t *last_terminal_action_mask; /* (batch_size, EX_ACTION_DIM) — final mask before auto-reset */
    float *info;    /* (batch_size, EX_INFO_DIM) — per-env metrics for PPO logging */
    DMFastSnapshot snapshot_pool[EX_SNAPSHOT_POOL_MAX];
    int32_t snapshot_count;
};

/* ─── RNG helpers ────────────────────────────────────────────────────── */

static inline uint64_t ex_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline int32_t ex_rnd_u32(DMFastExactEnv *e) {
    return (int32_t)(uint32_t)(ex_splitmix64(&e->rng) >> 32);
}

static inline int32_t ex_rnd_u16(DMFastExactEnv *e) {
    return ex_rnd_u32(e) & 0xFFFF;
}

static inline int32_t ex_rnd_u8(DMFastExactEnv *e) {
    return ex_rnd_u32(e) & 0xFF;
}

static inline int32_t ex_rnd_range(DMFastExactEnv *e, int32_t limit) {
    if (limit <= 0) return 0;
    return ((uint32_t)ex_rnd_u32(e)) % (uint32_t)limit;
}

/* ─── Derived state helpers ──────────────────────────────────────────── */

static int32_t ex_level_from_xp(int32_t xp) {
    int level = 1;
    if (xp <= 0) return 1;
    while ((level + 1) * (level + 1) <= xp) ++level;
    return level;
}

static int32_t ex_greatness_from_xp(int32_t xp) {
    int g = 1;
    if (xp <= 0) return 1;
    while ((g + 1) * (g + 1) <= xp && g < 20) ++g;
    return g;
}

/* Luck is a pure function of jewelry greatness, but special_stats[] caches it
 * and only the inventory kernel (dmfast_inventory_recalculate_special_stats)
 * ever recomputes it. Two paths change the inputs without going through that
 * kernel: combat raises equipment_xp (and so greatness) on every hit, and
 * ex_step_buy_item writes jewelry straight into equipment_ids. Either way luck
 * went stale until the next equip/drop happened to refresh it. Recomputed once
 * per step instead; mirrors the kernel's formula exactly, silver ring included. */
static void ex_refresh_luck(DMFastExactEnv *e) {
    int32_t neck = 0, ring = 0, bag_jewelry = 0;
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        if (e->equipment_ids[i] == 0) continue;
        if (i == 6) neck = ex_greatness_from_xp(e->equipment_xp[i]);
        else if (i == 7) ring = ex_greatness_from_xp(e->equipment_xp[i]);
    }
    for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
        if (e->bag_ids[i] == 0) continue;
        int slot = dmfast_loot_slot(e->bag_ids[i]);
        if (slot == 7 || slot == 8)
            bag_jewelry += ex_greatness_from_xp(e->bag_xp[i]);
    }
    /* Mirrors DMFAST_SILVER_RING_ID; dmfast_internal.h is not in scope here. */
    e->special_stats[EX_STAT_LUCK] = neck + ring + bag_jewelry
        + (e->equipment_ids[7] == 4 ? ring : 0);
}

static int32_t ex_max_health(const DMFastExactEnv *e) {
    int mh = EX_STARTING_HEALTH
        + (e->stats[EX_STAT_VIT] + e->special_stats[EX_STAT_VIT]) * EX_HEALTH_PER_VITALITY;
    return mh > EX_MAX_HEALTH ? EX_MAX_HEALTH : mh;
}

static int32_t ex_potion_cost(const DMFastExactEnv *e) {
    int level = ex_level_from_xp(e->adventurer_xp);
    int total_cha = e->stats[EX_STAT_CHA] + e->special_stats[EX_STAT_CHA];
    int cost = level - total_cha * EX_CHARISMA_POTION_DISC;
    return cost < EX_MIN_POTION_PRICE ? EX_MIN_POTION_PRICE : cost;
}

static int32_t ex_item_discount(const DMFastExactEnv *e) {
    return EX_CHARISMA_ITEM_DISC
        * (e->stats[EX_STAT_CHA] + e->special_stats[EX_STAT_CHA]);
}

static int ex_beast_present(const DMFastExactEnv *e) {
    return e->beast_health > 0;
}

static void ex_compute_market_prices(const DMFastExactEnv *e, int32_t *prices) {
    int32_t disc = ex_item_discount(e);
    for (int i = 0; i < EX_NUM_MARKET_SLOTS; ++i) {
        int id = e->market_ids[i];
        int tier = id == 0 ? 0 : dmfast_loot_tier(id);
        int base = tier == 0 ? 0 : (6 - tier) * EX_MARKET_TIER_PRICE;
        int adj = base - disc;
        prices[i] = adj < EX_MIN_ITEM_PRICE ? EX_MIN_ITEM_PRICE : adj;
    }
}

/* ─── Reward helpers ─────────────────────────────────────────────────── */

/* Phase index from level: matches Python EngineEnv phase boundaries exactly */
static int ex_phase_index(int32_t level) {
    if (level < 6) return 0;   /* very_early */
    if (level < 16) return 1;  /* early */
    if (level < 35) return 2;  /* mid */
    if (level < 70) return 3;  /* late */
    return 4;                  /* endgame */
}

static int ex_phase_index_cfg(int32_t level, const DMFastRewardConfig *rc) {
    if (level <= rc->phase_very_early_end) return 0;
    if (level <= rc->phase_early_end) return 1;
    if (level <= rc->phase_mid_end) return 2;
    if (level <= rc->phase_late_end) return 3;
    return 4;
}

/* Phase-scaled reward: table[phase_index] * global_scale.
 * Matches Python _phase_scaled() exactly. */
static float ex_phase_scale(const float *table, float global_scale, int32_t level) {
    return table[ex_phase_index(level)] * global_scale;
}

/* Phase flee target by level — matches Python _phase_flee_target() */
static float ex_phase_flee_target(int32_t level) {
    if (level <= 2) return 0.00f;
    if (level <= 5) return 0.21f;
    if (level <= 7) return 0.19f;
    if (level <= 10) return 0.18f;
    if (level <= 13) return 0.17f;
    if (level <= 16) return 0.16f;
    if (level <= 25) return 0.15f;
    if (level <= 34) return 0.14f;
    if (level <= 70) return 0.12f;
    return 0.26f;
}

/* XP subphase bin index: find which bin the current XP falls into.
 * bounds[] has 'count' entries; XP below bounds[0] → bin 0, etc. */
static int ex_xp_subphase_bin(int32_t xp, const DMFastRewardConfig *rc) {
    for (int i = 0; i < rc->xp_subphase_count; i++) {
        if (xp < rc->xp_subphase_bounds[i]) return i;
    }
    return rc->xp_subphase_count;  /* last bin */
}

/* XP subphase multiplier: returns table[bin] if enabled, else 1.0 */
static float ex_xp_subphase_mult(int32_t xp, const float *table, const DMFastRewardConfig *rc) {
    if (!rc->xp_subphase_enable || rc->xp_subphase_count == 0) return 1.0f;
    int bin = ex_xp_subphase_bin(xp, rc);
    int n_bins = rc->xp_subphase_count + 1;
    if (bin >= n_bins) bin = n_bins - 1;
    return table[bin];
}

/* XP subphase flee target: if enabled and has per-bin targets, use those */
static float ex_xp_subphase_flee_target(int32_t xp, int32_t level, const DMFastRewardConfig *rc) {
    if (rc->xp_subphase_enable && rc->xp_subphase_count > 0) {
        int bin = ex_xp_subphase_bin(xp, rc);
        int n_bins = rc->xp_subphase_count + 1;
        if (bin >= n_bins) bin = n_bins - 1;
        return rc->xp_subphase_flee_targets[bin];
    }
    return ex_phase_flee_target(level);
}

/* Compute equipment armor rating (sum of armor-slot tiers) */
static float ex_equipment_armor_rating(const DMFastExactEnv *e) {
    float rating = 0.0f;
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        if (e->equipment_ids[i] == 0) continue;
        int slot = dmfast_loot_slot(e->equipment_ids[i]);
        /* Armor slots: chest(2), head(3), waist(4), feet(5), hands(6) */
        if (slot >= 2 && slot <= 6) {
            int tier = dmfast_loot_tier(e->equipment_ids[i]);
            int greatness = ex_greatness_from_xp(e->equipment_xp[i]);
            rating += (float)((6 - tier) * 5 + greatness);
        }
    }
    return rating;
}

/* Count filled armor slots (indices 1-5 = chest,head,waist,feet,hands) */
static int ex_count_armor_slots(const DMFastExactEnv *e) {
    int count = 0;
    for (int i = 1; i <= 5; ++i) {
        if (e->equipment_ids[i] != 0) count++;
    }
    return count;
}

/* Check if all equipment slots filled */
static int ex_all_equipment_filled(const DMFastExactEnv *e) {
    /* Slots 0-7: weapon, chest, head, waist, feet, hands, neck, ring */
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        if (e->equipment_ids[i] == 0) return 0;
    }
    return 1;
}

/* Check asset_ready condition for flee penalty */
static int ex_asset_ready(const DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    int level = ex_level_from_xp(e->adventurer_xp);
    if (level < rc->flee_safe_sim_asset_ready_min_level) return 0;
    if (e->items_bought < rc->flee_safe_sim_asset_ready_min_items) return 0;
    if (ex_count_armor_slots(e) < rc->flee_safe_sim_asset_ready_min_armor_slots) return 0;
    int mh = ex_max_health(e);
    float hp_ratio = mh > 0 ? (float)e->health / (float)mh : 0.0f;
    if (hp_ratio < rc->flee_safe_sim_asset_ready_min_hp_ratio) return 0;
    return 1;
}

/* Get item combat family: 0=magic/cloth, 1=blade/hide, 2=bludgeon/metal, -1=none */
static int ex_item_combat_family(int32_t item_id) {
    if (item_id == 0) return -1;
    int item_type = dmfast_loot_type(item_id);
    /* Types: 0=blade/hide, 1=bludgeon/metal, 2=magic/cloth (varies by implementation) */
    /* In Loot Survivor: type is (id-1)/3 % 3: 0=magic/cloth, 1=blade/hide, 2=bludgeon/metal */
    /* Actually: type from dmfast_loot_type. We'll use direct mapping */
    if (item_type == 1) return 0; /* magic/cloth */
    if (item_type == 2) return 1; /* blade/hide */
    if (item_type == 3) return 2; /* bludgeon/metal */
    return -1;
}

/* Forward declarations for functions used before definition */
static float ex_item_quality_score(int32_t item_id, int32_t item_xp);
static float ex_best_slot_asset_score(const DMFastExactEnv *e);

static float ex_kill_milestone(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    float bonus = 0.0f;
    int k = e->beasts_killed;
    if (k >= 1 && !(e->milestone_kill_flags & 0x01)) {
        e->milestone_kill_flags |= 0x01;
        bonus += rc->milestone_kill_1;
    }
    if (k >= 3 && !(e->milestone_kill_flags & 0x02)) {
        e->milestone_kill_flags |= 0x02;
        bonus += rc->milestone_kill_3;
    }
    if (k >= 10 && !(e->milestone_kill_flags & 0x04)) {
        e->milestone_kill_flags |= 0x04;
        bonus += rc->milestone_kill_10;
    }
    if (k >= 20 && !(e->milestone_kill_flags & 0x08)) {
        e->milestone_kill_flags |= 0x08;
        bonus += rc->milestone_kill_20;
    }
    if (k >= 30 && !(e->milestone_kill_flags & 0x10)) {
        e->milestone_kill_flags |= 0x10;
        bonus += rc->milestone_kill_30;
    }
    return bonus;
}

static float ex_level_milestone(DMFastExactEnv *e, const DMFastRewardConfig *rc, int32_t new_level) {
    float bonus = 0.0f;
    if (new_level >= 8 && !(e->milestone_level_flags & 0x01)) {
        e->milestone_level_flags |= 0x01;
        bonus += rc->milestone_level_8;
    }
    if (new_level >= 12 && !(e->milestone_level_flags & 0x02)) {
        e->milestone_level_flags |= 0x02;
        bonus += rc->milestone_level_12;
    }
    if (new_level >= 16 && !(e->milestone_level_flags & 0x04)) {
        e->milestone_level_flags |= 0x04;
        bonus += rc->milestone_level_16;
    }
    if (new_level >= 20 && !(e->milestone_level_flags & 0x08)) {
        e->milestone_level_flags |= 0x08;
        bonus += rc->milestone_level_20;
    }
    if (new_level >= 25 && !(e->milestone_level_flags & 0x10)) {
        e->milestone_level_flags |= 0x10;
        bonus += rc->milestone_level_25;
    }
    if (new_level >= 35 && !(e->milestone_level_flags & 0x20)) {
        e->milestone_level_flags |= 0x20;
        bonus += rc->milestone_level_35;
    }
    return bonus;
}

/* Get sim_risk: uses cached value from last obs packing (real simulate_battles).
   Falls back to heuristic if no sim has been run yet. */
static float ex_get_sim_risk(const DMFastExactEnv *e) {
    if (e->cached_sim_risk > 0.0f) return e->cached_sim_risk;
    /* Fallback: simple heuristic for first step before obs is packed */
    int32_t level = ex_level_from_xp(e->adventurer_xp);
    if (level <= 0 || e->beast_level <= 0) return 0.0f;
    return (float)e->beast_level / (float)level;
}

/* Run simulate_battles on current env state; fills sim_out[10].
   Returns 0 (and zeros sim_out) if no beast present. */
static void ex_compute_sim_stats(const DMFastExactEnv *e, float *sim_out) {
    memset(sim_out, 0, sizeof(float) * 10);
    if (!ex_beast_present(e) || e->health <= 0) return;

    int32_t level = ex_level_from_xp(e->adventurer_xp);
    int32_t w_id = e->equipment_ids[0];
    int32_t w_tier = w_id == 0 ? 0 : dmfast_loot_tier(w_id);
    int32_t w_type = w_id == 0 ? 0 : dmfast_loot_type(w_id);
    int32_t w_level = ex_greatness_from_xp(e->equipment_xp[0]);
    int32_t w_specials[3] = {
        e->equipment_specials[0], e->equipment_specials[1], e->equipment_specials[2],
    };
    int32_t neck_id = e->equipment_ids[6];
    int32_t neck_great = ex_greatness_from_xp(e->equipment_xp[6]);
    int32_t ring_id = e->equipment_ids[7];
    int32_t ring_great = ex_greatness_from_xp(e->equipment_xp[7]);
    int32_t total_str = e->stats[EX_STAT_STR] + e->special_stats[EX_STAT_STR];
    int32_t total_luck = e->stats[EX_STAT_LUCK] + e->special_stats[EX_STAT_LUCK];

    int32_t armor_ids[5], armor_tier[5], armor_type[5], armor_level[5], armor_specials[15];
    for (int a = 0; a < 5; ++a) {
        int eq_idx = a + 1;
        armor_ids[a] = e->equipment_ids[eq_idx];
        armor_tier[a] = armor_ids[a] == 0 ? 0 : dmfast_loot_tier(armor_ids[a]);
        armor_type[a] = armor_ids[a] == 0 ? 0 : dmfast_loot_type(armor_ids[a]);
        armor_level[a] = ex_greatness_from_xp(e->equipment_xp[eq_idx]);
        for (int s = 0; s < 3; ++s)
            armor_specials[a * 3 + s] = e->equipment_specials[eq_idx * 3 + s];
    }

    int32_t h = e->health, bh = e->beast_health;
    dmfast_simulate_battles(
        &h, &bh,
        &total_str, &total_luck, &level,
        &w_id, &w_tier, &w_type, &w_level, w_specials,
        &neck_id, &neck_great, &ring_id, &ring_great,
        armor_ids, armor_tier, armor_type, armor_level, armor_specials,
        &e->beast_tier, &e->beast_type, &e->beast_level, e->beast_specials,
        1, sim_out
    );
}

/* Composite score matching Python _combat_swap_sim_score():
   1.6 * survivability_adventurer - 1.0 * survivability_risk + 0.35 * avg_damage_to_beast */
static float ex_combat_swap_sim_score(const float *sim_out) {
    return 1.6f * sim_out[0] - 1.0f * sim_out[2] + 0.35f * sim_out[3];
}

/* ─── Observation packing (exact Python EngineEnv layout) ────────────── */

static void ex_pack_item_vector(float *dst, int32_t id, int32_t xp,
                                int32_t sp1, int32_t sp2, int32_t sp3) {
    if (id == 0) {
        memset(dst, 0, sizeof(float) * EX_NUM_ITEM_FIELDS);
        return;
    }
    int greatness = ex_greatness_from_xp(xp);
    dst[0] = (float)id;
    /* tier_monotone: tier 1→5, 2→4, 3→3, 4→2, 5→1 → (-tier+6)/5 */
    int tier = dmfast_loot_tier(id);
    dst[1] = tier == 0 ? 0.0f : (float)(-tier + 6) / 5.0f;
    dst[2] = (float)dmfast_loot_type(id);
    dst[3] = (float)dmfast_loot_slot(id);
    dst[4] = (float)sp1;
    dst[5] = (float)sp2;
    dst[6] = (float)sp3;
    dst[7] = (float)xp / 400.0f;
    dst[8] = (float)(greatness - 1) / 20.0f;
}

static void ex_pack_equipment_item(float *dst, const DMFastExactEnv *e, int slot) {
    int id = e->equipment_ids[slot];
    int xp = e->equipment_xp[slot];
    /* Equipment specials come from the specials array, but for obs we use
       the item_specials_seed to compute them (matching Python). */
    int sp1 = 0, sp2 = 0, sp3 = 0;
    if (id != 0 && e->item_specials_seed != 0) {
        dmfast_loot_specials_for_item(id, 20, e->item_specials_seed, &sp1, &sp2, &sp3);
    }
    ex_pack_item_vector(dst, id, xp, sp1, sp2, sp3);
}

static void ex_pack_bag_item(float *dst, const DMFastExactEnv *e, int idx) {
    int id = e->bag_ids[idx];
    int xp = e->bag_xp[idx];
    int sp1 = 0, sp2 = 0, sp3 = 0;
    if (id != 0 && e->item_specials_seed != 0) {
        dmfast_loot_specials_for_item(id, 20, e->item_specials_seed, &sp1, &sp2, &sp3);
    }
    ex_pack_item_vector(dst, id, xp, sp1, sp2, sp3);
}

static void ex_pack_market_item(float *dst, const DMFastExactEnv *e, int idx) {
    int id = e->market_ids[idx];
    /* Market items have no XP or specials */
    int sp1 = 0, sp2 = 0, sp3 = 0;
    if (id != 0 && e->item_specials_seed != 0) {
        dmfast_loot_specials_for_item(id, 20, e->item_specials_seed, &sp1, &sp2, &sp3);
    }
    ex_pack_item_vector(dst, id, 0, sp1, sp2, sp3);
}

static void ex_pack_obs(DMFastExactBatch *b, int idx) {
    DMFastExactEnv *e = &b->envs[idx];
    float *dst = b->obs + (size_t)idx * EX_OBS_DIM;
    int c = 0;

    /* game_phase [1] */
    dst[c++] = (float)e->phase;

    /* adventurer [13] */
    int32_t mh = ex_max_health(e);
    dst[c++] = (float)e->health / (float)EX_MAX_HEALTH;
    dst[c++] = (float)mh / (float)EX_MAX_HEALTH;
    dst[c++] = (float)e->adventurer_xp / (float)EX_MAX_XP;
    dst[c++] = (float)e->gold / (float)EX_MAX_GOLD;
    {
        int su = e->stat_upgrades_available;
        if (su > EX_MAX_STAT_UPGRADES) su = EX_MAX_STAT_UPGRADES;
        dst[c++] = (float)su / (float)EX_MAX_STAT_UPGRADES;
    }
    dst[c++] = (float)e->beast_health / (float)EX_MAX_BEAST_HEALTH;
    /* Combined stats (base + special) */
    dst[c++] = (float)(e->stats[EX_STAT_STR] + e->special_stats[EX_STAT_STR]) / 31.0f;
    dst[c++] = (float)(e->stats[EX_STAT_DEX] + e->special_stats[EX_STAT_DEX]) / 31.0f;
    dst[c++] = (float)(e->stats[EX_STAT_VIT] + e->special_stats[EX_STAT_VIT]) / 31.0f;
    dst[c++] = (float)(e->stats[EX_STAT_INT] + e->special_stats[EX_STAT_INT]) / 31.0f;
    dst[c++] = (float)(e->stats[EX_STAT_WIS] + e->special_stats[EX_STAT_WIS]) / 31.0f;
    dst[c++] = (float)(e->stats[EX_STAT_CHA] + e->special_stats[EX_STAT_CHA]) / 31.0f;
    dst[c++] = (float)(e->stats[EX_STAT_LUCK] + e->special_stats[EX_STAT_LUCK]) / 100.0f;

    /* equipment [72] */
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        ex_pack_equipment_item(dst + c, e, i);
        c += EX_NUM_ITEM_FIELDS;
    }

    /* bag [135] */
    for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
        ex_pack_bag_item(dst + c, e, i);
        c += EX_NUM_ITEM_FIELDS;
    }

    /* market [225] */
    for (int i = 0; i < EX_NUM_MARKET_SLOTS; ++i) {
        ex_pack_market_item(dst + c, e, i);
        c += EX_NUM_ITEM_FIELDS;
    }

    /* beast [7] */
    if (ex_beast_present(e)) {
        dst[c++] = (float)e->beast_starting_health / (float)EX_MAX_BEAST_HEALTH;
        /* tier inverted: (-tier+6)/5 */
        dst[c++] = e->beast_tier == 0 ? 0.0f : (float)(-e->beast_tier + 6) / 5.0f;
        /* level: log1p(level) / log1p(640) */
        dst[c++] = e->beast_level > 0
            ? (float)(log1p((double)e->beast_level) / log1p((double)EX_MAX_OBS_COMBAT_LEVEL))
            : 0.0f;
        dst[c++] = (float)e->beast_type;
        dst[c++] = (float)e->beast_specials[0];
        dst[c++] = (float)e->beast_specials[1];
        dst[c++] = (float)e->beast_specials[2];
    } else {
        for (int i = 0; i < 7; ++i) dst[c++] = 0.0f;
    }

    /* sim_stats [10] — reuse ex_compute_sim_stats to avoid duplication */
    {
        float sim_out[10];
        ex_compute_sim_stats(e, sim_out);
        for (int i = 0; i < 10; ++i) dst[c++] = sim_out[i];
        e->cached_sim_risk = sim_out[2];
    }

    /* Verify obs dimension */
    if (c != EX_OBS_DIM) abort();
}

/* ─── Mask generation ────────────────────────────────────────────────── */

static void ex_update_mask(DMFastExactBatch *b, int idx) {
    DMFastExactEnv *e = &b->envs[idx];
    const DMFastRewardConfig *rc = &b->rc;
    uint8_t *mask = b->action_mask + (size_t)idx * EX_ACTION_DIM;
    memset(mask, 0, (size_t)EX_ACTION_DIM);

    if (e->done) return;

    int32_t mh = ex_max_health(e);
    int32_t pcost = ex_potion_cost(e);
    int32_t market_prices[EX_NUM_MARKET_SLOTS];
    ex_compute_market_prices(e, market_prices);

    /* Use the exact mask kernel */
    dmfast_action_masks(
        &e->phase, &e->gold, &e->health, &mh, &pcost,
        e->stats,  /* 7 stats */
        e->equipment_ids, e->bag_ids,
        e->market_ids, market_prices,
        e->switch_buffer_slots, e->buy_buffer_slots,
        1, mask
    );

    /* Sim-risk flee gate: block flee when sim says safe */
    if (rc->mask_block_flee_on_safe && e->phase == EX_PHASE_COMBAT && ex_beast_present(e)) {
        float sim_risk = ex_get_sim_risk(e);
        float hp_ratio = mh > 0 ? (float)e->health / (float)mh : 0.0f;
        int32_t level = ex_level_from_xp(e->adventurer_xp);
        float beast_ratio = level > 0 ? (float)e->beast_level / (float)level : 999.0f;

        if (sim_risk <= rc->mask_block_flee_safe_threshold
            && hp_ratio >= rc->mask_block_flee_min_hp_ratio
            && beast_ratio <= rc->mask_block_flee_max_beast_level_ratio) {
            mask[EX_ACT_FLEE] = 0;
            mask[EX_ACT_MACRO_FLEE] = 0;
        }
    }
}

/* ─── Reset ──────────────────────────────────────────────────────────── */

static void ex_write_info(float *info, const DMFastExactEnv *e) {
    info[EX_INFO_ADVENTURER_LEVEL] = (float)ex_level_from_xp(e->adventurer_xp);
    info[EX_INFO_XP]               = (float)e->adventurer_xp;
    info[EX_INFO_HEALTH]           = (float)e->health;
    info[EX_INFO_GOLD]             = (float)e->gold;
    info[EX_INFO_BEAST_LEVEL]      = (float)e->beast_level;
    info[EX_INFO_BEASTS_KILLED]    = (float)e->beasts_killed;
    info[EX_INFO_ITEMS_BOUGHT]     = (float)e->items_bought;
    info[EX_INFO_POTIONS_BOUGHT]   = (float)e->potions_bought;
    info[EX_INFO_CURRICULUM_START]  = (float)e->started_from_curriculum;
}

static void ex_pack_info(DMFastExactBatch *b, int idx) {
    const DMFastExactEnv *e = &b->envs[idx];
    ex_write_info(&b->info[idx * EX_INFO_DIM], e);
}

static void ex_pack_step_outputs(DMFastExactBatch *b, int idx, int live_mask_on_done) {
    DMFastExactEnv *e = &b->envs[idx];
    uint8_t saved_done = e->done;
    if (live_mask_on_done) {
        e->done = 0;
    }
    ex_pack_obs(b, idx);
    ex_update_mask(b, idx);
    e->done = saved_done;
    ex_pack_info(b, idx);
}

static void ex_capture_terminal_buffers(DMFastExactBatch *b, int idx, int live_mask_on_done) {
    ex_pack_step_outputs(b, idx, live_mask_on_done);
    memcpy(
        &b->last_terminal_obs[(size_t)idx * EX_OBS_DIM],
        &b->obs[(size_t)idx * EX_OBS_DIM],
        sizeof(float) * EX_OBS_DIM
    );
    memcpy(
        &b->last_terminal_action_mask[(size_t)idx * EX_ACTION_DIM],
        &b->action_mask[(size_t)idx * EX_ACTION_DIM],
        sizeof(uint8_t) * EX_ACTION_DIM
    );
}

static void ex_finish_reset(DMFastExactBatch *b, int idx) {
    b->reward[idx] = 0.0f;
    b->terminated[idx] = 0;
    b->truncated[idx] = 0;
    b->last_episode_return[idx] = 0.0f;
    b->last_episode_length[idx] = 0;
    b->phase_buf[idx] = b->envs[idx].phase;
    b->step_count[idx] = b->envs[idx].episode_length;
    b->episode_return[idx] = b->envs[idx].episode_return;
    ex_pack_obs(b, idx);
    ex_update_mask(b, idx);
    ex_pack_info(b, idx);
}

static void ex_reset_env(DMFastExactBatch *b, int idx, uint64_t seed) {
    DMFastExactEnv *e = &b->envs[idx];
    memset(e, 0, sizeof(*e));
    e->rng = seed;
    e->settings_id = seed;
    e->game_seed = ex_splitmix64(&e->rng);
    e->market_seed = ex_splitmix64(&e->rng);

    /* Draw starting weapon and stat rolls */
    int32_t starter_weapons[4] = {12, 16, 76, 46};  /* Wand, Book, Club, ShortSword */
    int32_t starting_weapon = starter_weapons[ex_rnd_range(e, 4)];
    int32_t stat_rnds[13];
    for (int i = 0; i < 13; ++i) {
        stat_rnds[i] = ex_rnd_range(e, 6);
    }

    /* Use the exact start kernel */
    int32_t out_phase;
    uint64_t out_game_seed, out_market_seed;
    int32_t out_health, out_xp, out_gold;
    int32_t out_stats[EX_NUM_STATS];
    int32_t out_special_stats[EX_NUM_STATS];
    int32_t out_stat_upgrades, out_action_count, out_item_specials_seed;
    int32_t out_equipment_ids[EX_NUM_EQUIPMENT_SLOTS];
    int32_t out_equipment_xp[EX_NUM_EQUIPMENT_SLOTS];
    int32_t out_equipment_specials[EX_NUM_EQUIPMENT_SLOTS * EX_NUM_SPECIAL_FIELDS];
    int32_t out_bag_ids[EX_NUM_BAG_SLOTS];
    int32_t out_market_ids[EX_NUM_MARKET_SLOTS];

    dmfast_process_starts(
        &e->settings_id, &starting_weapon, stat_rnds,
        &e->game_seed, &e->market_seed,
        1,
        &out_phase, &out_game_seed, &out_market_seed,
        &out_health, &out_xp, &out_gold,
        out_stats, out_special_stats,
        &out_stat_upgrades, &out_action_count, &out_item_specials_seed,
        out_equipment_ids, out_equipment_xp, out_equipment_specials,
        out_bag_ids, out_market_ids
    );

    /* Write state */
    e->phase = out_phase;
    e->game_seed = out_game_seed;
    e->market_seed = out_market_seed;
    e->adventurer_xp = out_xp;
    e->health = out_health;
    e->gold = out_gold;
    e->stat_upgrades_available = out_stat_upgrades;
    e->action_count = out_action_count;
    e->item_specials_seed = out_item_specials_seed;
    memcpy(e->stats, out_stats, sizeof(out_stats));
    memcpy(e->special_stats, out_special_stats, sizeof(out_special_stats));
    ex_refresh_luck(e);
    memcpy(e->equipment_ids, out_equipment_ids, sizeof(out_equipment_ids));
    memcpy(e->equipment_xp, out_equipment_xp, sizeof(out_equipment_xp));
    memcpy(e->equipment_specials, out_equipment_specials, sizeof(out_equipment_specials));
    memcpy(e->bag_ids, out_bag_ids, sizeof(out_bag_ids));
    memcpy(e->market_ids, out_market_ids, sizeof(out_market_ids));
    memset(e->switch_buffer_slots, 0, sizeof(e->switch_buffer_slots));
    memset(e->buy_buffer_slots, 0, sizeof(e->buy_buffer_slots));

    e->done = 0;
    e->episode_length = 0;
    e->episode_return = 0.0f;

    /* Curriculum: T1 weapon start */
    if (b->rc.curriculum_t1_weapon_prob > 0.0f) {
        float roll = (float)(ex_splitmix64(&e->rng) >> 40) / (float)(1ULL << 24);
        if (roll < b->rc.curriculum_t1_weapon_prob) {
            /* Replace weapon (slot 0) with random T1 weapon */
            static const int32_t t1_weapons[5] = {9, 13, 42, 43, 72};
            int32_t widx = ex_rnd_range(e, 5);
            e->equipment_ids[0] = t1_weapons[widx];
            e->equipment_xp[0] = 0;
            /* Clear weapon specials */
            e->equipment_specials[0 * EX_NUM_SPECIAL_FIELDS + 0] = 0;
            e->equipment_specials[0 * EX_NUM_SPECIAL_FIELDS + 1] = 0;
            e->equipment_specials[0 * EX_NUM_SPECIAL_FIELDS + 2] = 0;
            e->started_from_curriculum = 1;
        }
    }

    /* Go-Explore: restore from snapshot pool */
    if (b->rc.curriculum_snapshot_prob > 0.0f && b->snapshot_count > 0) {
        float roll = (float)(ex_splitmix64(&e->rng) >> 40) / (float)(1ULL << 24);
        if (roll < b->rc.curriculum_snapshot_prob) {
            /* Pick random snapshot */
            int snap_idx = ex_rnd_range(e, b->snapshot_count);
            uint64_t saved_rng = e->rng;
            memcpy(e, &b->snapshot_pool[snap_idx].state, sizeof(DMFastExactEnv));
            e->rng = saved_rng;  /* Use fresh RNG, not the snapshot's */
            /* Reset episode tracking (not game state) */
            e->done = 0;
            e->episode_length = 0;
            e->episode_return = 0.0f;
            e->started_from_curriculum = 1;
            /* Keep all reward tracking state from snapshot — that's the point.
             * Re-derive consistency fields below (prev_xp, etc.) and skip
             * to ex_finish_reset which re-packs obs/mask for the restored state. */
            e->prev_level = ex_level_from_xp(e->adventurer_xp);
            e->prev_xp = e->adventurer_xp;
            e->last_equipment_armor_rating = ex_equipment_armor_rating(e);
            e->current_potion_cost = ex_potion_cost(e);
            ex_finish_reset(b, idx);
            return;
        }
    }

    /* Extended tracking state (memset zeroed most, set non-zero defaults) */
    e->first_weapon_action = -1;
    e->first_jewelry_action = -1;
    e->current_phase_idx = ex_phase_index_cfg(ex_level_from_xp(e->adventurer_xp), &b->rc);
    e->highest_phase_reached = e->current_phase_idx;
    e->prev_level = ex_level_from_xp(e->adventurer_xp);
    e->prev_xp = e->adventurer_xp;
    e->last_equipment_armor_rating = ex_equipment_armor_rating(e);
    e->current_potion_cost = ex_potion_cost(e);

    ex_finish_reset(b, idx);
}

/* ─── Step: stat upgrade phase ───────────────────────────────────────── */

static float ex_step_upgrade(DMFastExactEnv *e, int32_t stat_index,
                             const DMFastRewardConfig *rc) {
    uint8_t valid;
    int32_t out_phase, out_health;
    int32_t out_stats[EX_NUM_STATS], out_special_stats[EX_NUM_STATS];
    int32_t out_upgrades;

    dmfast_process_stat_upgrades(
        &e->health, e->stats, e->special_stats, &e->stat_upgrades_available,
        &stat_index, 1,
        &valid, &out_phase, &out_health, out_stats, out_special_stats, &out_upgrades
    );

    if (!valid) return rc->invalid_action_penalty;

    e->phase = out_phase;
    e->health = out_health;
    e->stat_upgrades_available = out_upgrades;
    memcpy(e->stats, out_stats, sizeof(e->stats));
    memcpy(e->special_stats, out_special_stats, sizeof(e->special_stats));

    int32_t level = ex_level_from_xp(e->adventurer_xp);
    float reward = rc->stat_upgrade * ex_phase_scale(rc->phase_table_stat, rc->phase_scale_stat, level);

    /* Stat choice shaping — matching Python _stat_choice_shaping() */
    float stat_bonus = 0.0f;
    if (level == 2) {
        if (stat_index == EX_STAT_CHA) {
            stat_bonus += rc->stat_match_early_cha;
            stat_bonus += rc->stat_match_level2_cha;
        } else {
            stat_bonus += rc->stat_penalty_level2_non_cha;
        }
    } else if (level >= 3 && level <= 7) {
        if (stat_index == EX_STAT_DEX) stat_bonus += rc->stat_match_early_dex_strong;
        else if (stat_index == EX_STAT_CHA) stat_bonus += rc->stat_match_mid_cha * 0.25f;
    } else if (level >= 8 && level <= 12) {
        if (stat_index == EX_STAT_CHA) stat_bonus += rc->stat_match_mid_cha;
        else if (stat_index == EX_STAT_DEX) stat_bonus += rc->stat_match_mid_dex;
    } else if (level >= 13 && level <= 15) {
        if (stat_index == EX_STAT_VIT) stat_bonus += rc->stat_match_late_vit;
        else if (stat_index == EX_STAT_DEX) stat_bonus += rc->stat_match_mid_dex * 0.5f;
        else if (stat_index == EX_STAT_CHA) stat_bonus += rc->stat_match_mid_cha * 0.5f;
    } else if (level >= 16 && level <= 34) {
        if (stat_index == EX_STAT_VIT) stat_bonus += rc->stat_match_late_vit;
    } else if (level >= 35) {
        if (stat_index == EX_STAT_STR) stat_bonus += rc->stat_match_end_str;
        else if (stat_index == EX_STAT_DEX) stat_bonus += rc->stat_match_end_dex;
        else if (stat_index == EX_STAT_VIT) stat_bonus += rc->stat_match_end_vit;
    }
    /* Low-value stat penalty (INT=3, WIS=4) */
    if (level <= 34 && (stat_index == EX_STAT_INT || stat_index == EX_STAT_WIS)) {
        stat_bonus += rc->stat_penalty_low_value_early;
    }
    reward += stat_bonus;

    /* Update potion cost after stat change */
    e->current_potion_cost = ex_potion_cost(e);

    return reward;
}

/* ─── Step: market explore ───────────────────────────────────────────── */

static float ex_step_explore(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    int32_t mh = ex_max_health(e);
    int32_t level = ex_level_from_xp(e->adventurer_xp);
    int32_t base_dmg_reduction = EX_BASE_DAMAGE_REDUCTION;

    /* Draw all needed random values */
    int32_t explore_result = ex_rnd_range(e, 3);  /* 0=beast, 1=obstacle, 2=discovery */
    int32_t beast_seed = ex_rnd_u32(e);
    int32_t beast_health_rnd = ex_rnd_u16(e);
    int32_t beast_level_rnd = ex_rnd_u16(e);
    int32_t beast_dmg_loc_rnd = ex_rnd_u8(e);
    int32_t beast_crit_rnd = ex_rnd_u8(e);
    int32_t beast_ambush_rnd = ex_rnd_u8(e);
    int32_t beast_sp2_rnd = ex_rnd_u8(e);
    int32_t beast_sp3_rnd = ex_rnd_u8(e);
    int32_t obstacle_seed = ex_rnd_u32(e);
    int32_t obstacle_level_rnd = ex_rnd_u16(e);
    int32_t obstacle_dmg_loc_rnd = ex_rnd_u8(e);
    int32_t obstacle_crit_rnd = ex_rnd_u8(e);
    int32_t obstacle_dodge_rnd = ex_rnd_u8(e);
    int32_t obstacle_specials_rnd = ex_rnd_u16(e);
    int32_t disc_type_rnd = ex_rnd_u8(e);
    int32_t disc_amt_rnd1 = ex_rnd_u8(e);
    int32_t disc_amt_rnd2 = ex_rnd_u8(e);

    /* Output buffers */
    int32_t out_action_count, out_phase, out_xp, out_health, out_gold;
    int32_t out_stat_upgrades, out_item_specials_seed;
    int32_t out_equip_ids[EX_NUM_EQUIPMENT_SLOTS];
    int32_t out_bag_ids[EX_NUM_BAG_SLOTS];
    int32_t out_equip_xp[EX_NUM_EQUIPMENT_SLOTS];
    int32_t out_equip_specials[EX_NUM_EQUIPMENT_SLOTS * EX_NUM_SPECIAL_FIELDS];
    int32_t out_special_stats[EX_NUM_STATS];
    /* Beast outputs */
    int32_t out_beast_id, out_beast_health, out_beast_tier, out_beast_type, out_beast_level;
    int32_t out_beast_specials[3];
    uint8_t out_beast_ambush;
    int32_t out_beast_atk_dmg, out_beast_atk_loc;
    uint8_t out_beast_atk_crit;
    /* Obstacle outputs */
    int32_t out_obstacle_id, out_obstacle_dmg, out_obstacle_loc, out_obstacle_xp_reward;
    uint8_t out_obstacle_dodged, out_obstacle_crit;
    /* Discovery outputs */
    int32_t out_disc_type, out_disc_loot_id;

    int32_t intelligence = e->stats[EX_STAT_INT];
    int32_t wisdom = e->stats[EX_STAT_WIS];
    int32_t base_vit = e->stats[EX_STAT_VIT];

    dmfast_process_explores(
        &e->action_count, &e->adventurer_xp, &mh, &e->health, &e->gold,
        &intelligence, &wisdom, &base_vit, &base_dmg_reduction,
        e->special_stats, &e->stat_upgrades_available, &e->item_specials_seed,
        e->equipment_ids, e->equipment_xp, e->equipment_specials,
        e->bag_ids,
        &explore_result,
        &beast_seed, &beast_health_rnd, &beast_level_rnd,
        &beast_dmg_loc_rnd, &beast_crit_rnd, &beast_ambush_rnd,
        &beast_sp2_rnd, &beast_sp3_rnd,
        &obstacle_seed, &obstacle_level_rnd, &obstacle_dmg_loc_rnd,
        &obstacle_crit_rnd, &obstacle_dodge_rnd, &obstacle_specials_rnd,
        &disc_type_rnd, &disc_amt_rnd1, &disc_amt_rnd2,
        1,
        &out_action_count, &out_phase, &out_xp, &out_health, &out_gold,
        &out_stat_upgrades, &out_item_specials_seed,
        out_equip_ids, out_bag_ids,
        out_equip_xp, out_equip_specials, out_special_stats,
        &out_beast_id, &out_beast_health, &out_beast_tier, &out_beast_type,
        &out_beast_level, out_beast_specials,
        &out_beast_ambush, &out_beast_atk_dmg, &out_beast_atk_loc, &out_beast_atk_crit,
        &out_obstacle_id, &out_obstacle_dodged, &out_obstacle_dmg,
        &out_obstacle_loc, &out_obstacle_crit, &out_obstacle_xp_reward,
        &out_disc_type, &out_disc_loot_id
    );

    /* Write back state */
    e->action_count = out_action_count;
    e->phase = out_phase;
    e->adventurer_xp = out_xp;
    e->health = out_health;
    e->gold = out_gold;
    e->stat_upgrades_available = out_stat_upgrades;
    e->item_specials_seed = out_item_specials_seed;
    memcpy(e->equipment_ids, out_equip_ids, sizeof(e->equipment_ids));
    memcpy(e->bag_ids, out_bag_ids, sizeof(e->bag_ids));
    memcpy(e->equipment_xp, out_equip_xp, sizeof(e->equipment_xp));
    memcpy(e->equipment_specials, out_equip_specials, sizeof(e->equipment_specials));
    memcpy(e->special_stats, out_special_stats, sizeof(e->special_stats));

    /* Update beast state if beast encountered */
    if (out_phase == EX_PHASE_COMBAT) {
        e->beast_id = out_beast_id;
        e->beast_health = out_beast_health;
        e->beast_starting_health = out_beast_health;
        /* If ambush dealt damage, beast_health is the starting health but
           adventurer health was already reduced by the kernel */
        e->beast_tier = out_beast_tier;
        e->beast_type = out_beast_type;
        e->beast_level = out_beast_level;
        memcpy(e->beast_specials, out_beast_specials, sizeof(e->beast_specials));
    }

    /* Clear switch buffer on explore; market session is over so the buy
     * buffer resets too. */
    memset(e->switch_buffer_slots, 0, sizeof(e->switch_buffer_slots));
    memset(e->buy_buffer_slots, 0, sizeof(e->buy_buffer_slots));

    /* Check for stat upgrades on level-up */
    int32_t new_level = ex_level_from_xp(e->adventurer_xp);
    if (new_level > level && e->stat_upgrades_available > 0 && out_phase != EX_PHASE_COMBAT) {
        e->phase = EX_PHASE_UPGRADE;
    }

    float reward = rc->explore;

    /* XP shaping: reward for XP gained */
    int32_t xp_gained = e->adventurer_xp - e->prev_xp;
    if (xp_gained > 0) {
        reward += rc->xp_shaping_scale * (float)xp_gained / (float)EX_MAX_XP;
    }

    /* Level-up bonus */
    if (new_level > e->prev_level) {
        reward += rc->level_up * (float)(new_level - e->prev_level);
        reward += ex_level_milestone(e, rc, new_level);
    }

    return reward;
}

/* ─── Step: buy potion ───────────────────────────────────────────────── */

static float ex_step_buy_potion(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    uint8_t bp = ex_beast_present(e) ? 1 : 0;
    uint8_t valid;
    int32_t out_phase, out_health, out_gold;
    int32_t out_stats[EX_NUM_STATS], out_special_stats[EX_NUM_STATS];
    int32_t out_upgrades, out_potions;

    int32_t old_health = e->health;
    int32_t mh = ex_max_health(e);

    dmfast_process_buy_potions(
        &e->adventurer_xp, &e->health, &e->gold,
        e->stats, e->special_stats, &e->stat_upgrades_available,
        &bp, 1,
        &valid, &out_phase, &out_health, &out_gold,
        out_stats, out_special_stats, &out_upgrades, &out_potions
    );

    if (!valid) return rc->invalid_action_penalty;

    e->health = out_health;
    e->gold = out_gold;
    e->potions_bought += out_potions;

    int32_t level = ex_level_from_xp(e->adventurer_xp);
    float potion_sub_mult = ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_potion_reward_mult, rc);
    float reward = rc->buy_potion * ex_phase_scale(rc->phase_table_potion, rc->phase_scale_potion, level)
                 * potion_sub_mult;

    /* HP-conditional shaping (xp_subphase: potion_reward_mult) */
    float hp_ratio = mh > 0 ? (float)old_health / (float)mh : 0.0f;
    if (hp_ratio < rc->potion_low_hp_threshold) {
        reward += rc->potion_low_hp * potion_sub_mult;
    } else if (hp_ratio > rc->potion_high_hp_threshold) {
        reward += rc->potion_high_hp_penalty * potion_sub_mult;
    }

    /* Potion without gear penalty */
    if (rc->potion_without_gear != 0.0f && level >= rc->potion_without_gear_min_level) {
        int item_deficit = rc->potion_without_gear_min_items - e->items_bought;
        if (item_deficit < 0) item_deficit = 0;
        float armor = ex_equipment_armor_rating(e);
        float armor_deficit = (float)rc->potion_without_gear_min_armor - armor;
        if (armor_deficit < 0.0f) armor_deficit = 0.0f;
        if (item_deficit > 0 || armor_deficit > 0.0f) {
            float deficit_score = (float)item_deficit + armor_deficit / 20.0f;
            reward += rc->potion_without_gear * deficit_score
                      * ex_phase_scale(rc->phase_table_economy, rc->phase_scale_economy, level);
        }
    }

    /* Potion cost shaping (early game charisma drive) */
    if (level <= rc->early_potion_level_threshold) {
        float old_cost = e->current_potion_cost;
        float new_cost = ex_potion_cost(e);
        /* Reduce potion cost bonus */
        if (rc->reduce_potion_cost_early != 0.0f) {
            float cost_drop = old_cost - new_cost;
            if (cost_drop > 0.0f) {
                reward += rc->reduce_potion_cost_early * cost_drop;
            }
        }
        /* Penalty if cost above target */
        if (rc->early_potion_cost_above_target != 0.0f && new_cost > rc->early_potion_cost_target) {
            reward += rc->early_potion_cost_above_target * (new_cost - rc->early_potion_cost_target);
        }
    }

    return reward;
}

/* ─── Step: combat attack ────────────────────────────────────────────── */

/* Pending-swap tax: if any gear was swapped since the last combat tick, the
 * beast takes ONE free swing for the whole batch when the player commits to
 * attack/flee. Matches the python engine — and the rational on-chain play of
 * batching all swaps into a single equip tx (one counter-attack total).
 * Returns 1 if a retaliation fired. */
static int ex_pending_switch_retaliation(DMFastExactEnv *e) {
    int has_pending = 0;
    for (int si = 0; si < EX_NUM_EQUIPMENT_SLOTS; ++si) {
        if (e->switch_buffer_slots[si] != 0) {
            has_pending = 1;
            break;
        }
    }
    if (!has_pending || !ex_beast_present(e)) {
        return 0;
    }
    int32_t level = ex_level_from_xp(e->adventurer_xp);
    int32_t beast_crit_rnd = ex_rnd_u8(e);
    int32_t atk_loc_rnd = ex_rnd_u8(e);
    int32_t out_ret_health;
    int32_t out_ret_dmg, out_ret_loc;
    uint8_t out_ret_crit;
    dmfast_beast_retaliation(
        0, &e->health,
        e->equipment_ids, e->equipment_xp, e->equipment_specials,
        &e->beast_tier, &e->beast_type, &e->beast_level, e->beast_specials,
        &beast_crit_rnd, &atk_loc_rnd, level,
        &out_ret_health, &out_ret_dmg, &out_ret_loc, &out_ret_crit
    );
    e->health = out_ret_health;
    return 1;
}

static float ex_step_attack(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    /* Pending swaps pay their one-time tax before the attack resolves. */
    ex_pending_switch_retaliation(e);
    /* Each combat tick refreshes the per-slot swap budget: the switch
     * buffer was previously only cleared on explore, so once a slot was
     * swapped it stayed locked for every attack/flee tick of the same
     * encounter. */
    memset(e->switch_buffer_slots, 0, sizeof(e->switch_buffer_slots));
    if (e->health <= 0) {
        /* Killed by the swap retaliation; outer loop handles death. */
        return 0.0f;
    }

    int32_t str = e->stats[EX_STAT_STR];
    int32_t luck = e->stats[EX_STAT_LUCK];
    int32_t base_vit = e->stats[EX_STAT_VIT];

    /* Draw random values */
    int32_t item_specials_rnd = ex_rnd_u16(e);
    int32_t adv_crit_rnd = ex_rnd_u8(e);
    int32_t beast_crit_rnd = ex_rnd_u8(e);
    int32_t atk_loc_rnd = ex_rnd_u8(e);

    /* Output buffers */
    int32_t out_atk_dmg, out_atk_loc;
    uint8_t out_atk_crit, out_beast_dead;
    int32_t out_beast_level_dead, out_gold_reward, out_xp_reward;
    int32_t out_xp, out_gold, out_health, out_beast_health;
    int32_t out_beast_atk_dmg, out_beast_atk_loc;
    uint8_t out_beast_atk_crit;
    int32_t out_equip_xp[EX_NUM_EQUIPMENT_SLOTS];
    int32_t out_equip_specials[EX_NUM_EQUIPMENT_SLOTS * EX_NUM_SPECIAL_FIELDS];
    int32_t out_special_stats[EX_NUM_STATS];
    int32_t out_upgrades, out_iss;

    dmfast_process_attacks(
        &e->adventurer_xp, &e->gold, &e->health,
        &str, &luck, &base_vit,
        e->special_stats, &e->stat_upgrades_available, &e->item_specials_seed,
        e->equipment_ids, e->equipment_xp, e->equipment_specials,
        &e->beast_health, &e->beast_tier, &e->beast_type, &e->beast_level, e->beast_specials,
        &item_specials_rnd, &adv_crit_rnd, &beast_crit_rnd, &atk_loc_rnd,
        1,
        &out_atk_dmg, &out_atk_loc, &out_atk_crit,
        &out_beast_dead, &out_beast_level_dead,
        &out_gold_reward, &out_xp_reward,
        &out_xp, &out_gold, &out_health, &out_beast_health,
        &out_beast_atk_dmg, &out_beast_atk_loc, &out_beast_atk_crit,
        out_equip_xp, out_equip_specials, out_special_stats,
        &out_upgrades, &out_iss
    );

    /* Capture beast info before state update */
    int32_t beast_level_pre = e->beast_level;

    /* Write state */
    e->adventurer_xp = out_xp;
    e->gold = out_gold;
    e->health = out_health;
    e->beast_health = out_beast_health;
    memcpy(e->equipment_xp, out_equip_xp, sizeof(e->equipment_xp));
    memcpy(e->equipment_specials, out_equip_specials, sizeof(e->equipment_specials));
    memcpy(e->special_stats, out_special_stats, sizeof(e->special_stats));
    e->stat_upgrades_available = out_upgrades;
    e->item_specials_seed = out_iss;

    int32_t level = ex_level_from_xp(e->adventurer_xp);
    float attack_scale = ex_phase_scale(rc->phase_table_attack, rc->phase_scale_attack, level);

    /* Base attack reward */
    float reward = rc->attack * attack_scale;
    if (beast_level_pre > 15) reward += rc->attack_over_15;

    /* Sim-risk based attack bonus */
    float sim_risk = ex_get_sim_risk(e);
    if (sim_risk <= rc->sim_risk_safe_threshold) {
        reward += rc->attack_safe_sim;
    }

    /* Attack favorable bonus (xp_subphase: attack_commit_mult) */
    if (rc->attack_favorable != 0.0f && level > 0 &&
        (float)beast_level_pre <= rc->attack_favorable_level_ratio * (float)level) {
        reward += rc->attack_favorable * attack_scale
                * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_attack_commit_mult, rc);
    }

    /* Reset flee streak on attack */
    e->flee_streak = 0;

    if (out_beast_dead) {
        e->beasts_killed += 1;
        e->phase_kills += 1;
        e->beast_id = 0;
        e->beast_health = 0;
        e->beast_starting_health = 0;
        e->beast_tier = 0;
        e->beast_type = 0;
        e->beast_level = 0;
        memset(e->beast_specials, 0, sizeof(e->beast_specials));
        e->had_swap_this_combat = 0;

        /* Check level-up for stat upgrades */
        if (e->stat_upgrades_available > 0) {
            e->phase = EX_PHASE_UPGRADE;
        } else {
            e->phase = EX_PHASE_MARKET;
        }

        /* Refresh market on level-up */
        int32_t new_level = ex_level_from_xp(e->adventurer_xp);
        int32_t old_level = ex_level_from_xp(e->adventurer_xp - out_xp_reward);
        if (new_level > old_level) {
            e->market_seed = ex_splitmix64(&e->rng);
            dmfast_market_items(
                &e->settings_id, &e->market_seed, 1, EX_NUM_MARKET_SLOTS, e->market_ids
            );
        }

        /* Kill reward: base * beast_level, scaled by phase */
        float kill_scale = ex_phase_scale(rc->phase_table_kill, rc->phase_scale_kill, level);
        if (beast_level_pre > 15) {
            reward += rc->kill_over_15 * (float)beast_level_pre * kill_scale;
        } else {
            reward += rc->kill * (float)beast_level_pre * kill_scale;
        }

        /* Kill streak bonus (xp_subphase: kill_target_mult) */
        if (rc->kill_streak != 0.0f) {
            reward += rc->kill_streak * (float)e->beasts_killed * kill_scale
                    * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_kill_target_mult, rc);
        }

        /* High-level kill bonus */
        if (beast_level_pre >= rc->beast_high_level_threshold) {
            float hk_scale = ex_phase_scale(rc->phase_table_high_kill, rc->phase_scale_high_kill, level);
            reward += rc->kill_high_level * hk_scale;
        }

        /* Relative-level kill bonus */
        if (level > 0 && (float)beast_level_pre >= rc->kill_high_relative_ratio * (float)level) {
            reward += rc->kill_high_relative;
        }

        /* Early farm penalty: killing easy beasts after gearing up */
        if (rc->early_farm_penalty != 0.0f && level <= rc->early_farm_max_level &&
            sim_risk <= rc->early_farm_risk_max &&
            level > 0 && (float)beast_level_pre <= rc->early_farm_easy_level_ratio * (float)level &&
            e->items_bought >= rc->early_farm_target_items) {
            reward += rc->early_farm_penalty * kill_scale;
        }

        /* Kill phase target bonus (xp_subphase: flee targets + kill_target_mult) */
        if (rc->kill_phase_target_bonus != 0.0f) {
            int total = e->phase_kills + e->phase_flees;
            if (total >= rc->phase_target_min_resolutions) {
                float flee_ratio = (float)e->phase_flees / (float)total;
                float target = ex_xp_subphase_flee_target(e->adventurer_xp, level, rc);
                float drift = flee_ratio - target;
                if (drift > 0.0f) {
                    reward += rc->kill_phase_target_bonus * drift * kill_scale
                            * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_kill_target_mult, rc);
                }
            }
        }

        /* Kill milestones */
        reward += ex_kill_milestone(e, rc);

        /* Level-up from kill */
        if (new_level > e->prev_level) {
            reward += rc->level_up * (float)(new_level - e->prev_level);
            reward += ex_level_milestone(e, rc, new_level);
        }

        /* XP shaping */
        if (out_xp_reward > 0) {
            reward += rc->xp_shaping_scale * (float)out_xp_reward / (float)EX_MAX_XP;
        }
    }

    return reward;
}

/* ─── Step: combat flee ──────────────────────────────────────────────── */

static float ex_step_flee(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    /* Pending swaps pay their one-time tax before the flee attempt. */
    ex_pending_switch_retaliation(e);
    /* See ex_step_attack: flee clears the switch buffer too, both on
     * success (beast gone → MARKET/UPGRADE) and on failed flee (still in
     * combat but a new tick has elapsed). */
    memset(e->switch_buffer_slots, 0, sizeof(e->switch_buffer_slots));
    if (e->health <= 0) {
        /* Killed by the swap retaliation; outer loop handles death. */
        return 0.0f;
    }

    int32_t dex = e->stats[EX_STAT_DEX];
    int32_t beast_level_pre = e->beast_level;

    /* Draw random values */
    int32_t flee_rnd = ex_rnd_u8(e);
    int32_t beast_crit_rnd = ex_rnd_u8(e);
    int32_t atk_loc_rnd = ex_rnd_u8(e);

    uint8_t out_fled;
    int32_t out_xp, out_health, out_beast_health, out_upgrades;
    int32_t out_beast_atk_dmg, out_beast_atk_loc;
    uint8_t out_beast_atk_crit;

    dmfast_process_flees(
        &e->adventurer_xp, &e->health, &dex,
        e->special_stats, &e->stat_upgrades_available,
        e->equipment_ids, e->equipment_xp, e->equipment_specials,
        &e->beast_health, &e->beast_tier, &e->beast_type, &e->beast_level, e->beast_specials,
        &flee_rnd, &beast_crit_rnd, &atk_loc_rnd,
        1,
        &out_fled, &out_xp, &out_health, &out_beast_health, &out_upgrades,
        &out_beast_atk_dmg, &out_beast_atk_loc, &out_beast_atk_crit
    );

    e->adventurer_xp = out_xp;
    e->health = out_health;
    e->beast_health = out_beast_health;
    e->stat_upgrades_available = out_upgrades;

    int32_t level = ex_level_from_xp(e->adventurer_xp);
    float flee_scale = ex_phase_scale(rc->phase_table_flee, rc->phase_scale_flee, level);

    if (out_fled) {
        /* Determine if this is a "hard" flee context */
        float sim_risk = ex_get_sim_risk(e);
        int hard_flee = (level > 0 && (float)beast_level_pre >= rc->flee_hard_level_ratio * (float)level)
                     || sim_risk >= rc->sim_risk_hard_threshold;

        e->phase_flees += 1;
        e->beast_id = 0;
        e->beast_starting_health = 0;
        e->beast_tier = 0;
        e->beast_type = 0;
        e->beast_level = 0;
        memset(e->beast_specials, 0, sizeof(e->beast_specials));
        e->had_swap_this_combat = 0;

        if (e->stat_upgrades_available > 0) {
            e->phase = EX_PHASE_UPGRADE;
        } else {
            e->phase = EX_PHASE_MARKET;
        }

        float reward = (rc->flee_gain + rc->flee_penalty) * flee_scale;

        /* Easy flee penalty (xp_subphase: flee_easy_penalty_mult) */
        if (level > 0 && (float)beast_level_pre < rc->flee_easy_level_ratio * (float)level) {
            reward += rc->flee_easy_penalty * flee_scale
                    * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_flee_easy_penalty_mult, rc);
        }

        /* Hard flee bonus */
        if (level > 0 && (float)beast_level_pre >= rc->flee_hard_level_ratio * (float)level) {
            reward += rc->flee_hard_bonus;
        }

        /* Sim-risk penalty for fleeing when safe */
        if (sim_risk <= rc->sim_risk_safe_threshold) {
            reward += rc->flee_safe_sim_penalty;
            /* Asset-ready extra penalty (xp_subphase: flee_target_mult) */
            if (rc->flee_safe_sim_asset_ready_extra != 0.0f && ex_asset_ready(e, rc)) {
                reward += rc->flee_safe_sim_asset_ready_extra * flee_scale
                        * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_flee_target_mult, rc);
            }
        }

        /* Sim-risk bonus for fleeing when hard */
        if (sim_risk >= rc->sim_risk_hard_threshold) {
            reward += rc->flee_hard_sim;
        }

        /* Flee streak penalty */
        if (hard_flee) {
            e->flee_streak = 0;
        } else {
            e->flee_streak += 1;
            if (rc->flee_streak_penalty != 0.0f && e->flee_streak > rc->flee_streak_grace) {
                reward += rc->flee_streak_penalty * (float)(e->flee_streak - rc->flee_streak_grace) * flee_scale;
            }
        }

        /* Flee phase target penalty (xp_subphase: flee targets + flee_target_mult) */
        if (rc->flee_phase_target_penalty != 0.0f) {
            int total = e->phase_kills + e->phase_flees;
            if (total >= rc->phase_target_min_resolutions) {
                float flee_ratio = (float)e->phase_flees / (float)total;
                float target = ex_xp_subphase_flee_target(e->adventurer_xp, level, rc);
                float drift = flee_ratio - target;
                if (drift > 0.0f) {
                    reward += rc->flee_phase_target_penalty * drift * flee_scale
                            * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_flee_target_mult, rc);
                }
            }
        }

        /* Flee after improving swap penalty (xp_subphase: flee_target_mult) */
        if (rc->flee_after_swap_penalty != 0.0f && !hard_flee &&
            sim_risk > 0.0f && sim_risk <= rc->sim_risk_safe_threshold &&
            e->had_swap_this_combat &&
            e->last_swap_sim_delta >= rc->flee_after_swap_min_delta) {
            reward += rc->flee_after_swap_penalty * flee_scale
                    * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_flee_target_mult, rc);
        }

        return reward;
    }

    /* Failed flee: took damage */
    return rc->flee_penalty * flee_scale - 0.01f * (float)out_beast_atk_dmg / (float)EX_MAX_HEALTH;
}

/* ─── Step: equip item from bag ──────────────────────────────────────── */

static float ex_step_equip(DMFastExactEnv *e, int32_t bag_index,
                           const DMFastRewardConfig *rc) {
    uint8_t bp = ex_beast_present(e) ? 1 : 0;
    uint8_t valid;
    int32_t out_phase, out_health;
    int32_t out_special_stats[EX_NUM_STATS];
    int32_t out_equip_ids[EX_NUM_EQUIPMENT_SLOTS];
    int32_t out_equip_xp[EX_NUM_EQUIPMENT_SLOTS];
    int32_t out_equip_specials[EX_NUM_EQUIPMENT_SLOTS * EX_NUM_SPECIAL_FIELDS];
    int32_t out_bag_ids[EX_NUM_BAG_SLOTS];
    int32_t out_bag_xp[EX_NUM_BAG_SLOTS];
    int32_t out_bag_specials[EX_NUM_BAG_SLOTS * EX_NUM_SPECIAL_FIELDS];
    int32_t out_switch[EX_NUM_EQUIPMENT_SLOTS];

    /* Capture pre-equip sim stats for composite survivability score */
    float pre_sim[10];
    memset(pre_sim, 0, sizeof(pre_sim));
    int32_t level = ex_level_from_xp(e->adventurer_xp);
    int do_switch_surv = (rc->switch_survivability != 0.0f && ex_beast_present(e) &&
                          level >= rc->switch_surv_min_level && level <= rc->switch_surv_max_level);
    if (do_switch_surv) {
        ex_compute_sim_stats(e, pre_sim);
    }

    int32_t base_vit = e->stats[EX_STAT_VIT];

    dmfast_process_equips(
        &e->health, &base_vit, e->special_stats, &e->stat_upgrades_available,
        e->equipment_ids, e->equipment_xp, e->equipment_specials,
        e->bag_ids, e->bag_xp, e->bag_specials,
        e->switch_buffer_slots, &bp, &bag_index,
        1,
        &valid, &out_phase, &out_health, out_special_stats,
        out_equip_ids, out_equip_xp, out_equip_specials,
        out_bag_ids, out_bag_xp, out_bag_specials,
        out_switch
    );

    if (!valid) {
        e->phase = EX_PHASE_MARKET;
        return rc->invalid_action_penalty;
    }

    e->phase = out_phase;
    e->health = out_health;
    memcpy(e->special_stats, out_special_stats, sizeof(e->special_stats));
    memcpy(e->equipment_ids, out_equip_ids, sizeof(e->equipment_ids));
    memcpy(e->equipment_xp, out_equip_xp, sizeof(e->equipment_xp));
    memcpy(e->equipment_specials, out_equip_specials, sizeof(e->equipment_specials));
    memcpy(e->bag_ids, out_bag_ids, sizeof(e->bag_ids));
    memcpy(e->bag_xp, out_bag_xp, sizeof(e->bag_xp));
    memcpy(e->bag_specials, out_bag_specials, sizeof(e->bag_specials));
    memcpy(e->switch_buffer_slots, out_switch, sizeof(e->switch_buffer_slots));

    /* No per-swap retaliation here. Matches the python engine (and the
     * rational on-chain play of batching all swaps into one equip tx):
     * swaps accumulate in the switch buffer and the beast takes ONE free
     * swing at the next attack/flee commit (ex_step_attack/ex_step_flee)
     * if any swaps are pending. */

    float reward = rc->switch_item;

    /* Switch quality delta: reward quality improvement on equip */
    if (rc->switch_quality_delta != 0.0f &&
        level >= rc->switch_quality_min_level && level <= rc->switch_quality_max_level) {
        /* Find which slot was swapped (compare old vs new equipment) */
        for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
            if (out_switch[i]) {
                /* Slot i was modified. Compute quality delta. */
                float new_q = ex_item_quality_score(e->equipment_ids[i], e->equipment_xp[i]);
                /* Old item is now in bag (find it) */
                float old_q = 0.0f;
                for (int j = 0; j < EX_NUM_BAG_SLOTS; ++j) {
                    if (e->bag_ids[j] != 0 && dmfast_loot_slot(e->bag_ids[j]) == i + 1) {
                        old_q = ex_item_quality_score(e->bag_ids[j], e->bag_xp[j]);
                        break;
                    }
                }
                float qdelta = new_q - old_q;
                if (qdelta > 0.0f) {
                    float econ_scale = ex_phase_scale(rc->phase_table_economy, rc->phase_scale_economy, level);
                    reward += rc->switch_quality_delta * qdelta * econ_scale
                            * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_buy_reward_mult, rc);
                }
                break;
            }
        }
    }

    /* Survivability delta: composite score matching Python _switch_items_combat_sim_bonus.
       score = 1.6 * surv_adv - 1.0 * surv_risk + 0.35 * avg_dmg_to_beast.
       Delta > 0 = improvement. Phase-scaled with "attack" family. Clipped. */
    if (do_switch_surv) {
        float post_sim[10];
        ex_compute_sim_stats(e, post_sim);
        float score_delta = ex_combat_swap_sim_score(post_sim) - ex_combat_swap_sim_score(pre_sim);
        if (rc->switch_surv_score_clip > 0.0f) {
            if (score_delta > rc->switch_surv_score_clip) score_delta = rc->switch_surv_score_clip;
            if (score_delta < -rc->switch_surv_score_clip) score_delta = -rc->switch_surv_score_clip;
        }
        float atk_scale = ex_phase_scale(rc->phase_table_attack, rc->phase_scale_attack, level);
        float surv_sub = ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_kill_target_mult, rc);
        if (score_delta > 1e-9f) {
            reward += rc->switch_survivability * score_delta * atk_scale * surv_sub;
        } else if (score_delta < -1e-9f && rc->switch_worse_combat_sim != 0.0f) {
            /* Penalty for worse combat sim */
            reward += rc->switch_worse_combat_sim * (-score_delta) * atk_scale * surv_sub;
        }
        /* Track for flee-after-swap */
        e->last_swap_sim_delta = score_delta;
        e->had_swap_this_combat = 1;
    }

    return reward;
}

/* ─── Step: drop item from bag ───────────────────────────────────────── */

static float ex_step_drop(DMFastExactEnv *e, int32_t bag_index,
                          const DMFastRewardConfig *rc) {
    uint8_t bp = ex_beast_present(e) ? 1 : 0;
    uint8_t valid;
    int32_t out_phase, out_health;
    int32_t out_special_stats[EX_NUM_STATS];
    int32_t out_bag_ids[EX_NUM_BAG_SLOTS];
    int32_t out_bag_xp[EX_NUM_BAG_SLOTS];
    int32_t out_bag_specials[EX_NUM_BAG_SLOTS * EX_NUM_SPECIAL_FIELDS];

    int32_t base_vit = e->stats[EX_STAT_VIT];

    dmfast_process_drops(
        &e->health, &base_vit, e->special_stats, &e->stat_upgrades_available,
        e->equipment_ids, e->equipment_xp, e->equipment_specials,
        e->bag_ids, e->bag_xp, e->bag_specials,
        &bp, &bag_index,
        1,
        &valid, &out_phase, &out_health, out_special_stats,
        out_bag_ids, out_bag_xp, out_bag_specials
    );

    if (!valid) {
        e->phase = EX_PHASE_MARKET;
        return rc->invalid_action_penalty;
    }

    e->phase = out_phase;
    e->health = out_health;
    memcpy(e->special_stats, out_special_stats, sizeof(e->special_stats));
    memcpy(e->bag_ids, out_bag_ids, sizeof(e->bag_ids));
    memcpy(e->bag_xp, out_bag_xp, sizeof(e->bag_xp));
    memcpy(e->bag_specials, out_bag_specials, sizeof(e->bag_specials));
    return rc->drop_item;
}

/* ─── Item quality helpers (match Python _item_quality_score exactly) ── */

/* Quality score: 10*(6-tier) + greatness + slot_bonus.
   slot_bonus: weapon=1.0, armor(chest/head/waist/feet/hands)=0.6, jewelry(neck/ring)=0.3 */
static float ex_item_quality_score(int32_t item_id, int32_t item_xp) {
    if (item_id == 0) return 0.0f;
    int tier = dmfast_loot_tier(item_id);
    int slot = dmfast_loot_slot(item_id);  /* 1=weapon, 2-6=armor, 7-8=jewelry */
    int greatness = ex_greatness_from_xp(item_xp);
    float quality = 10.0f * (float)(6 - tier) + (float)greatness;
    if (slot == 1) quality += 1.0f;
    else if (slot >= 2 && slot <= 6) quality += 0.6f;
    else if (slot == 7 || slot == 8) quality += 0.3f;
    return quality;
}

/* Sum of best quality item per equipment slot (across equipment + bag).
   Matches Python _best_slot_asset_score(). */
static float ex_best_slot_asset_score(const DMFastExactEnv *e) {
    float best[EX_NUM_EQUIPMENT_SLOTS];
    memset(best, 0, sizeof(best));
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        float q = ex_item_quality_score(e->equipment_ids[i], e->equipment_xp[i]);
        if (q > best[i]) best[i] = q;
    }
    for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
        if (e->bag_ids[i] == 0) continue;
        int slot = dmfast_loot_slot(e->bag_ids[i]);
        int eq_idx = slot - 1;
        if (eq_idx >= 0 && eq_idx < EX_NUM_EQUIPMENT_SLOTS) {
            float q = ex_item_quality_score(e->bag_ids[i], e->bag_xp[i]);
            if (q > best[eq_idx]) best[eq_idx] = q;
        }
    }
    float total = 0.0f;
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) total += best[i];
    return total;
}

/* ─── Step: buy item from market ─────────────────────────────────────── */

static float ex_step_buy_item(DMFastExactEnv *e, int32_t market_index,
                              const DMFastRewardConfig *rc) {
    if (market_index < 0 || market_index >= EX_NUM_MARKET_SLOTS) {
        e->phase = EX_PHASE_MARKET;
        return rc->invalid_action_penalty;
    }

    int32_t item_id = e->market_ids[market_index];
    if (item_id == 0) {
        e->phase = EX_PHASE_MARKET;
        return rc->invalid_action_penalty;
    }

    int32_t tier = dmfast_loot_tier(item_id);
    int32_t base_price = tier == 0 ? 0 : (6 - tier) * EX_MARKET_TIER_PRICE;
    int32_t disc = ex_item_discount(e);
    int32_t price = base_price - disc;
    if (price < EX_MIN_ITEM_PRICE) price = EX_MIN_ITEM_PRICE;

    if (price > e->gold) {
        e->phase = EX_PHASE_MARKET;
        return rc->invalid_action_penalty;
    }

    int32_t slot = dmfast_loot_slot(item_id);
    /* Map loot slot (1-8) to equipment index (0-7) */
    int32_t eq_idx = slot - 1;
    if (eq_idx < 0 || eq_idx >= EX_NUM_EQUIPMENT_SLOTS) {
        e->phase = EX_PHASE_MARKET;
        return rc->invalid_action_penalty;
    }

    /* Check if we can place existing equipment in bag */
    int existing_id = e->equipment_ids[eq_idx];
    int bag_slot = -1;
    if (existing_id != 0) {
        /* Need a free bag slot for the displaced item */
        for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
            if (e->bag_ids[i] == 0) { bag_slot = i; break; }
        }
        if (bag_slot < 0) {
            e->phase = EX_PHASE_MARKET;
            return rc->invalid_action_penalty;
        }
    }

    /* Check for duplicates */
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        if (e->equipment_ids[i] == item_id) {
            e->phase = EX_PHASE_MARKET;
            return rc->invalid_action_penalty;
        }
    }
    for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
        if (e->bag_ids[i] == item_id) {
            e->phase = EX_PHASE_MARKET;
            return rc->invalid_action_penalty;
        }
    }

    int32_t level = ex_level_from_xp(e->adventurer_xp);
    float econ_scale = ex_phase_scale(rc->phase_table_economy, rc->phase_scale_economy, level);

    /* Pre-buy quality scores (before state mutation) */
    float pre_quality = ex_item_quality_score(existing_id,
        existing_id != 0 ? e->equipment_xp[eq_idx] : 0);
    float pre_asset_score = 0.0f;
    if (rc->buy_asset_bank_delta > 0.0f && level <= rc->buy_asset_bank_delta_max_level) {
        pre_asset_score = ex_best_slot_asset_score(e);
    }

    /* Execute the buy */
    e->gold -= price;

    /* Move existing equipment to bag if needed */
    if (existing_id != 0 && bag_slot >= 0) {
        e->bag_ids[bag_slot] = e->equipment_ids[eq_idx];
        e->bag_xp[bag_slot] = e->equipment_xp[eq_idx];
        for (int s = 0; s < EX_NUM_SPECIAL_FIELDS; ++s) {
            e->bag_specials[bag_slot * EX_NUM_SPECIAL_FIELDS + s] =
                e->equipment_specials[eq_idx * EX_NUM_SPECIAL_FIELDS + s];
        }
    }

    /* Place new item in equipment */
    e->equipment_ids[eq_idx] = item_id;
    e->equipment_xp[eq_idx] = 0;
    memset(&e->equipment_specials[eq_idx * EX_NUM_SPECIAL_FIELDS], 0,
           sizeof(int32_t) * EX_NUM_SPECIAL_FIELDS);

    /* Remove from market */
    e->market_ids[market_index] = 0;

    /* Track the purchased slot for this market session (slot ids 1-8) */
    e->buy_buffer_slots[eq_idx] = eq_idx + 1;

    e->items_bought += 1;

    e->phase = EX_PHASE_MARKET;

    /* Base buy reward (phase-scaled) */
    float reward = rc->buy_item * econ_scale;

    /* Tier-1 item bonus — one-time per equipment slot */
    if (tier == 1 && !(e->flag_buy_t1_slots & (1u << eq_idx))) {
        e->flag_buy_t1_slots |= (1u << eq_idx);
        reward += rc->buy_t1_item * econ_scale;
        /* Extra bonus for T1 weapons — one-time per episode */
        if (slot == 1 && !e->flag_buy_weapon_t1) {  /* DMFAST_LOOT_SLOT_WEAPON */
            e->flag_buy_weapon_t1 = 1;
            reward += rc->buy_weapon_t1 * econ_scale;
        }
    }

    /* Quality delta: 10*(6-tier) + greatness + slot_bonus (matching Python exactly) */
    if (rc->buy_quality_delta > 0.0f) {
        float new_quality = ex_item_quality_score(item_id, 0);  /* new item: 0 XP = greatness 1 */
        float delta = new_quality - pre_quality;
        if (delta > 0.0f) {
            reward += rc->buy_quality_delta * delta * econ_scale;
        }
    }

    /* Asset bank delta: sum of best quality per slot, before vs after buy */
    if (rc->buy_asset_bank_delta > 0.0f && level <= rc->buy_asset_bank_delta_max_level) {
        float post_asset_score = ex_best_slot_asset_score(e);
        float delta = post_asset_score - pre_asset_score;
        if (delta > 0.0f) {
            reward += rc->buy_asset_bank_delta * delta * econ_scale;
        }
    }

    /* ─── Extended buy rewards ─── */

    /* First weapon online */
    if (slot == 1 && e->first_weapon_action < 0) {  /* weapon slot */
        e->first_weapon_action = e->action_count;
        e->first_weapon_level = level;
        if (rc->first_weapon_online != 0.0f) {
            reward += rc->first_weapon_online * econ_scale;
        }
        /* First weapon early bonus */
        if (rc->first_weapon_early != 0.0f &&
            e->first_weapon_action <= rc->first_weapon_early_max_action &&
            e->first_weapon_level <= rc->first_weapon_early_max_level) {
            reward += rc->first_weapon_early * econ_scale;
        }
    }

    /* Jewelry tracking */
    int is_jewelry = (slot == 7 || slot == 8);  /* neck or ring */
    if (is_jewelry && e->first_jewelry_action < 0) {
        e->first_jewelry_action = e->action_count;
        e->first_jewelry_level = level;
    }

    /* Jewelry timing rewards (only when first jewelry bought) */
    if (is_jewelry && e->first_jewelry_action == e->action_count) {
        /* Penalty for jewelry before weapon */
        if (e->first_weapon_action < 0 && rc->jewelry_before_weapon != 0.0f) {
            reward += rc->jewelry_before_weapon * econ_scale;
        }
        /* Bonus for weapon before jewelry */
        if (e->first_weapon_action >= 0 && e->first_weapon_action < e->first_jewelry_action &&
            rc->weapon_before_jewelry != 0.0f) {
            reward += rc->weapon_before_jewelry * econ_scale;
        }
        /* Early jewelry penalty/good timing */
        int action_deficit = rc->early_jewelry_min_action - e->first_jewelry_action;
        if (action_deficit < 0) action_deficit = 0;
        int level_deficit = rc->early_jewelry_min_level - e->first_jewelry_level;
        if (level_deficit < 0) level_deficit = 0;
        if (action_deficit > 0 || level_deficit > 0) {
            float action_scale = (float)action_deficit / (float)(rc->early_jewelry_min_action > 0 ? rc->early_jewelry_min_action : 1);
            float lscale = (float)level_deficit / (float)(rc->early_jewelry_min_level > 0 ? rc->early_jewelry_min_level : 1);
            float penalty_scale = action_scale + lscale;
            if (rc->early_jewelry_penalty != 0.0f)
                reward += rc->early_jewelry_penalty * penalty_scale * econ_scale;
        } else if (rc->jewelry_timing_good != 0.0f) {
            reward += rc->jewelry_timing_good * econ_scale;
        }
    }

    /* Early armor slot fill (slots 2-6 = armor) */
    if (slot >= 2 && slot <= 6 && rc->early_armor_slot_fill != 0.0f &&
        e->first_weapon_action >= 0 && level <= rc->early_armor_fill_max_level) {
        uint8_t slot_bit = (uint8_t)(1u << (slot - 2));
        if (!(e->flag_early_armor_slots & slot_bit)) {
            e->flag_early_armor_slots |= slot_bit;
            reward += rc->early_armor_slot_fill * econ_scale;
        }
    }

    /* Combat family diversity */
    if (rc->new_combat_family != 0.0f || rc->all_combat_families_early != 0.0f) {
        int family = ex_item_combat_family(item_id);
        if (family >= 0 && family < 3) {
            uint8_t fbit = (uint8_t)(1u << family);
            if (!(e->bought_combat_families & fbit)) {
                e->bought_combat_families |= fbit;
                if (rc->new_combat_family != 0.0f)
                    reward += rc->new_combat_family * econ_scale;
                /* All 3 families bonus */
                if (e->bought_combat_families == 0x07 && !e->flag_all_combat_families &&
                    level <= rc->combat_family_diversity_max_level &&
                    rc->all_combat_families_early != 0.0f) {
                    e->flag_all_combat_families = 1;
                    reward += rc->all_combat_families_early * econ_scale;
                }
            }
        }
    }

    /* Early target item bonus */
    if (rc->buy_early_target_item != 0.0f && level <= rc->buy_early_target_item_max_level) {
        for (int ti = 0; ti < rc->early_target_item_count; ++ti) {
            if (item_id == rc->early_target_item_ids[ti]) {
                /* Check if not already rewarded (bit set) */
                int word = item_id / 32;
                int bit = item_id % 32;
                if (word < 4 && !(e->early_target_item_seen[word] & (1u << bit))) {
                    e->early_target_item_seen[word] |= (1u << bit);
                    e->early_target_items_bought += 1;
                    reward += rc->buy_early_target_item * econ_scale;
                }
                break;
            }
        }
    }

    /* Necklace + ring equipped check */
    if (rc->necklace_ring_equipped != 0.0f && !e->flag_necklace_ring &&
        e->equipment_ids[6] != 0 && e->equipment_ids[7] != 0) {  /* neck=6, ring=7 */
        e->flag_necklace_ring = 1;
        reward += rc->necklace_ring_equipped * econ_scale;
    }

    /* Full equipment check */
    if (rc->full_equipment != 0.0f && !e->flag_full_equipment && ex_all_equipment_filled(e)) {
        e->flag_full_equipment = 1;
        reward += rc->full_equipment * econ_scale;
    }

    /* Apply XP subphase buy_reward_mult to total buy reward */
    return reward * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_buy_reward_mult, rc);
}

/* ─── Macro: fight until resolved ────────────────────────────────────── */

static float ex_macro_fight(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    float reward = 0.0f;
    for (int i = 0; i < 64; ++i) {
        if (e->phase != EX_PHASE_COMBAT || e->health <= 0) break;
        reward += ex_step_attack(e, rc);
        if (e->phase != EX_PHASE_COMBAT || e->health <= 0) break;
    }
    reward += rc->macro_fight_kill_resolve
            * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_kill_target_mult, rc);
    return reward;
}

/* ─── Macro: flee until resolved ─────────────────────────────────────── */

static float ex_macro_flee(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    float reward = 0.0f;
    for (int i = 0; i < 64; ++i) {
        if (e->phase != EX_PHASE_COMBAT || e->health <= 0) break;
        reward += ex_step_flee(e, rc);
        if (e->phase != EX_PHASE_COMBAT) break;
    }
    reward += rc->macro_flee_hard_resolve
            * ex_xp_subphase_mult(e->adventurer_xp, rc->xp_subphase_flee_target_mult, rc);
    return reward;
}

/* ─── Macro: equip best for fight ────────────────────────────────────── */

static float ex_macro_equip_best(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    /* Find best bag item that improves over current equipment */
    int best_idx = -1;
    float best_gain = 0.0f;
    for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
        if (e->bag_ids[i] == 0) continue;
        int slot = dmfast_loot_slot(e->bag_ids[i]) - 1;
        if (slot < 0 || slot >= EX_NUM_EQUIPMENT_SLOTS) continue;
        int old_tier = e->equipment_ids[slot] == 0 ? 0 : dmfast_loot_tier(e->equipment_ids[slot]);
        int new_tier = dmfast_loot_tier(e->bag_ids[i]);
        float gain = (float)(old_tier - new_tier);  /* lower tier = better */
        if (gain > best_gain) {
            best_gain = gain;
            best_idx = i;
        }
    }
    if (best_idx < 0) return rc->invalid_action_penalty;
    return ex_step_equip(e, best_idx, rc);
}

/* ─── Macro: explore until beast, then soften ────────────────────────── */

static float ex_macro_explore_soften(DMFastExactEnv *e, const DMFastRewardConfig *rc) {
    float reward = 0.0f;
    for (int i = 0; i < 24; ++i) {
        if (e->phase != EX_PHASE_MARKET) break;
        reward += ex_step_explore(e, rc);
    }
    /* If we found a beast, attack until one-hit range */
    if (e->phase == EX_PHASE_COMBAT) {
        for (int i = 0; i < 32 && e->phase == EX_PHASE_COMBAT; ++i) {
            reward += ex_step_attack(e, rc);
            if (e->phase != EX_PHASE_COMBAT || e->health <= 0) break;
        }
    }
    return reward;
}

/* ─── Action decode ──────────────────────────────────────────────────── */

static int32_t ex_decode_action(const DMFastExactEnv *e, int32_t action) {
    int bag_off = EX_NUM_BASE_ACTIONS;
    int market_off = bag_off + EX_NUM_BAG_SLOTS;
    int stats_off = market_off + EX_NUM_MARKET_SLOTS;

    if (e->phase == EX_PHASE_UPGRADE) {
        if (action >= stats_off && action < stats_off + 6) return action - stats_off;
        return action;
    }
    if (e->phase == EX_PHASE_DROP || e->phase == EX_PHASE_EQUIP) {
        if (action >= bag_off && action < bag_off + EX_NUM_BAG_SLOTS) return action - bag_off;
        return action;
    }
    if (e->phase == EX_PHASE_BUY) {
        if (action >= market_off && action < market_off + EX_NUM_MARKET_SLOTS)
            return action - market_off;
        return action;
    }
    return action;
}

/* ─── Main step function ─────────────────────────────────────────────── */

/* ─── Go-Explore: snapshot capture ───────────────────────────────────── */

static void ex_maybe_capture_snapshot(DMFastExactBatch *b, int idx) {
    DMFastExactEnv *e = &b->envs[idx];
    DMFastRewardConfig *rc = &b->rc;

    if (rc->curriculum_snapshot_prob <= 0.0f) return;
    if (e->done) return;
    if (e->adventurer_xp < rc->curriculum_snapshot_min_xp) return;
    if (e->adventurer_xp > rc->curriculum_snapshot_max_xp) return;

    int32_t max_health = EX_STARTING_HEALTH
        + (e->stats[EX_STAT_VIT] + e->special_stats[EX_STAT_VIT]) * EX_HEALTH_PER_VITALITY;
    if (max_health > EX_MAX_HEALTH) max_health = EX_MAX_HEALTH;
    float hp_ratio = (max_health > 0) ? (float)e->health / (float)max_health : 0.0f;
    if (hp_ratio < rc->curriculum_snapshot_min_hp_ratio) return;

    /* Check if we already have a similar XP snapshot (within 50 XP) */
    for (int i = 0; i < b->snapshot_count; i++) {
        int32_t diff = b->snapshot_pool[i].xp - e->adventurer_xp;
        if (diff < 0) diff = -diff;
        if (diff < 50) return;
    }

    /* Add to pool */
    if (b->snapshot_count < EX_SNAPSHOT_POOL_MAX) {
        DMFastSnapshot *s = &b->snapshot_pool[b->snapshot_count++];
        memcpy(&s->state, e, sizeof(DMFastExactEnv));
        s->xp = e->adventurer_xp;
        s->level = ex_level_from_xp(e->adventurer_xp);
        s->valid = 1;
    } else {
        /* Replace lowest-XP snapshot if current is higher */
        int min_idx = 0;
        for (int i = 1; i < EX_SNAPSHOT_POOL_MAX; i++) {
            if (b->snapshot_pool[i].xp < b->snapshot_pool[min_idx].xp)
                min_idx = i;
        }
        if (e->adventurer_xp > b->snapshot_pool[min_idx].xp) {
            DMFastSnapshot *s = &b->snapshot_pool[min_idx];
            memcpy(&s->state, e, sizeof(DMFastExactEnv));
            s->xp = e->adventurer_xp;
            s->level = ex_level_from_xp(e->adventurer_xp);
            s->valid = 1;
        }
    }
}

static void ex_step_one(DMFastExactBatch *b, int idx, int32_t action, int auto_reset) {
    DMFastExactEnv *e = &b->envs[idx];
    const DMFastRewardConfig *rc = &b->rc;
    float reward = rc->invalid_action_penalty;
    uint8_t terminated = 0;
    uint8_t truncated = 0;

    if (!e->done) {
        e->episode_length += 1;

        /* Track pre-step state for reward computation */
        e->prev_xp = e->adventurer_xp;
        e->prev_level = ex_level_from_xp(e->adventurer_xp);

        action = ex_decode_action(e, action);

        switch (e->phase) {
            case EX_PHASE_UPGRADE:
                reward = ex_step_upgrade(e, action, rc);
                break;

            case EX_PHASE_MARKET:
                switch (action) {
                    case EX_ACT_EXPLORE: {
                        /* Market no-buy penalty: penalize exploring without buying
                         * when conditions are met (have gold, options, HP ratio ok, level ok) */
                        float no_buy_penalty = 0.0f;
                        if (rc->market_no_buy_penalty != 0.0f) {
                            int32_t mkt_level = ex_level_from_xp(e->adventurer_xp);
                            if (mkt_level <= rc->market_no_buy_max_level &&
                                e->gold >= rc->market_no_buy_min_gold) {
                                /* Count affordable buyable options */
                                int mkt_options = 0;
                                for (int mi = 0; mi < EX_NUM_MARKET_SLOTS; mi++) {
                                    if (e->market_ids[mi] != 0) {
                                        int32_t t = dmfast_loot_tier(e->market_ids[mi]);
                                        int32_t bp = t == 0 ? 0 : (6 - t) * EX_MARKET_TIER_PRICE;
                                        int32_t d = ex_item_discount(e);
                                        int32_t p = bp - d;
                                        if (p < EX_MIN_ITEM_PRICE) p = EX_MIN_ITEM_PRICE;
                                        if (p <= e->gold) mkt_options++;
                                    }
                                }
                                if (mkt_options >= rc->market_no_buy_min_options) {
                                    int32_t mh_check = ex_max_health(e);
                                    float hp_r = mh_check > 0 ? (float)e->health / (float)mh_check : 1.0f;
                                    if (hp_r >= rc->market_no_buy_min_hp_ratio) {
                                        float econ_sc = ex_phase_scale(rc->phase_table_economy,
                                                                        rc->phase_scale_economy, mkt_level);
                                        no_buy_penalty = rc->market_no_buy_penalty * econ_sc
                                            * ex_xp_subphase_mult(e->adventurer_xp,
                                                rc->xp_subphase_market_no_buy_mult, rc);
                                    }
                                }
                            }
                        }
                        e->market_explores += 1;
                        reward = ex_step_explore(e, rc) + no_buy_penalty;
                        break;
                    }
                    case EX_ACT_BUY_ITEM:
                        e->phase = EX_PHASE_BUY;
                        reward = 0.0f;
                        break;
                    case EX_ACT_BUY_POTION:
                        reward = ex_step_buy_potion(e, rc);
                        break;
                    case EX_ACT_EQUIP:
                        e->phase = EX_PHASE_EQUIP;
                        reward = 0.0f;
                        break;
                    case EX_ACT_DROP:
                        e->phase = EX_PHASE_DROP;
                        reward = 0.0f;
                        break;
                    case EX_ACT_MACRO_EQUIP:
                        reward = ex_macro_equip_best(e, rc);
                        break;
                    case EX_ACT_MACRO_EXPLORE:
                        reward = ex_macro_explore_soften(e, rc);
                        break;
                    default:
                        break;
                }
                break;

            case EX_PHASE_BUY:
                reward = ex_step_buy_item(e, action, rc);
                break;

            case EX_PHASE_EQUIP:
                reward = ex_step_equip(e, action, rc);
                break;

            case EX_PHASE_DROP:
                reward = ex_step_drop(e, action, rc);
                break;

            case EX_PHASE_COMBAT:
                switch (action) {
                    case EX_ACT_ATTACK:
                        reward = ex_step_attack(e, rc);
                        break;
                    case EX_ACT_FLEE:
                        reward = ex_step_flee(e, rc);
                        break;
                    case EX_ACT_EQUIP:
                        /* Enter EQUIP phase; the actual swap happens next
                         * tick when the agent picks a bag index, and the
                         * beast takes a free swing on the way back to
                         * COMBAT (see ex_step_equip). */
                        e->phase = EX_PHASE_EQUIP;
                        reward = 0.0f;
                        break;
                    case EX_ACT_MACRO_FIGHT:
                        reward = ex_macro_fight(e, rc);
                        break;
                    case EX_ACT_MACRO_FLEE:
                        reward = ex_macro_flee(e, rc);
                        break;
                    default:
                        break;
                }
                break;

            default:
                break;
        }

        /* Every action above can change jewelry greatness or the jewelry
         * itself; refresh once here rather than at each mutation site. */
        ex_refresh_luck(e);

        /* ─── Per-step global rewards ─── */
        int32_t cur_level = ex_level_from_xp(e->adventurer_xp);

        /* Phase progress: one-time bonus per phase boundary crossing */
        if (rc->phase_progress != 0.0f) {
            int32_t new_phase = ex_phase_index_cfg(cur_level, rc);
            if (new_phase > e->highest_phase_reached) {
                e->highest_phase_reached = new_phase;
                reward += rc->phase_progress * ex_phase_scale(rc->phase_table_xp, rc->phase_scale_xp, cur_level);
            }
            /* Reset phase kills/flees on phase transition */
            if (new_phase != e->current_phase_idx) {
                e->phase_kills = 0;
                e->phase_flees = 0;
                e->current_phase_idx = new_phase;
            }
        }

        /* Stagnation penalty: level < threshold AND actions > budget */
        if (rc->stagnation_pre16 != 0.0f &&
            cur_level < rc->stagnation_pre16_level_thresh &&
            e->action_count > rc->stagnation_pre16_action_thresh) {
            int excess = e->action_count - rc->stagnation_pre16_action_thresh;
            float severity = (float)excess / (float)(rc->stagnation_pre16_action_thresh > 0 ? rc->stagnation_pre16_action_thresh : 1);
            if (severity > rc->stagnation_pre16_max_penalty) severity = rc->stagnation_pre16_max_penalty;
            reward += rc->stagnation_pre16 * severity
                      * ex_phase_scale(rc->phase_table_xp, rc->phase_scale_xp, cur_level);
        }

        /* Level floor penalty: action budgets vs level targets */
        if (rc->level_floor_penalty != 0.0f) {
            for (int lf = 0; lf < rc->level_floor_count; ++lf) {
                int budget = rc->level_floor_budgets[lf];
                int min_lvl = rc->level_floor_min_levels[lf];
                if (cur_level < min_lvl && e->action_count > budget) {
                    float level_gap_ratio = (float)(min_lvl - cur_level) / (float)(min_lvl > 0 ? min_lvl : 1);
                    float action_excess_ratio = (float)(e->action_count - budget) / (float)(budget > 0 ? budget : 1);
                    float severity = level_gap_ratio + 0.5f * action_excess_ratio;
                    if (severity > rc->level_floor_max_step_penalty) severity = rc->level_floor_max_step_penalty;
                    reward += rc->level_floor_penalty * severity;
                }
            }
        }

        /* Equipment armor delta tracking */
        if (rc->equipment_armor_delta != 0.0f || rc->equipment_armor_drop != 0.0f ||
            rc->low_equipment_armor != 0.0f) {
            float cur_armor = ex_equipment_armor_rating(e);
            float diff = cur_armor - e->last_equipment_armor_rating;
            float econ_s = ex_phase_scale(rc->phase_table_economy, rc->phase_scale_economy, cur_level);
            if (diff > 0.0f && rc->equipment_armor_delta != 0.0f) {
                reward += rc->equipment_armor_delta * diff * econ_s;
            } else if (diff < 0.0f && rc->equipment_armor_drop != 0.0f) {
                reward += rc->equipment_armor_drop * (-diff) * econ_s;
            }
            /* Low armor floor penalty */
            if (rc->low_equipment_armor != 0.0f && cur_level >= rc->low_equipment_armor_min_level) {
                float target_armor = rc->low_equipment_armor_target_per_level * (float)cur_level;
                float deficit = target_armor - cur_armor;
                if (deficit > 0.0f) {
                    reward += rc->low_equipment_armor * deficit * econ_s;
                }
            }
            e->last_equipment_armor_rating = cur_armor;
        }

        /* Item greatness-15 tracking */
        if (rc->item_to_level_15 != 0.0f || rc->weapon_t1_level_15 != 0.0f) {
            for (int gi = 0; gi < EX_NUM_EQUIPMENT_SLOTS; ++gi) {
                if (e->equipment_ids[gi] == 0) continue;
                if (e->items_reached_level_15 & (1u << gi)) continue;
                int greatness = ex_greatness_from_xp(e->equipment_xp[gi]);
                if (greatness >= 15) {
                    e->items_reached_level_15 |= (uint8_t)(1u << gi);
                    if (rc->item_to_level_15 != 0.0f) {
                        reward += rc->item_to_level_15 * ex_phase_scale(rc->phase_table_kill, rc->phase_scale_kill, cur_level);
                    }
                    if (rc->weapon_t1_level_15 != 0.0f && gi == 0 &&
                        dmfast_loot_tier(e->equipment_ids[gi]) == 1) {
                        reward += rc->weapon_t1_level_15 * ex_phase_scale(rc->phase_table_high_kill, rc->phase_scale_high_kill, cur_level);
                    }
                }
            }
        }

        /* Potion cost state shaping */
        if (rc->early_potion_cost_state_above_target != 0.0f && cur_level <= rc->early_potion_level_threshold) {
            float pcost = ex_potion_cost(e);
            if (pcost > rc->early_potion_cost_target) {
                reward += rc->early_potion_cost_state_above_target * (pcost - rc->early_potion_cost_target);
            } else if (rc->early_potion_cost_state_at_below_target != 0.0f) {
                reward += rc->early_potion_cost_state_at_below_target;
            }
            e->current_potion_cost = pcost;
        }

        /* Death check */
        if (e->health <= 0) {
            e->health = 0;
            e->done = 1;
            terminated = 1;
            reward += rc->death_penalty;
        } else if (e->episode_length >= b->max_steps) {
            e->done = 1;
            truncated = 1;
        } else if (b->rc.max_level > 0.0f &&
                   (float)ex_level_from_xp(e->adventurer_xp) >= b->rc.max_level) {
            e->done = 1;
            truncated = 1;  /* armor-curriculum: reached crossover+n level */
        }
    }

    /* Reward is now derived outside the native engine from observable state diffs.
     * Keep native reward/return buffers neutral so game logic remains the only
     * responsibility of the C step path. */
    reward = 0.0f;
    e->episode_return = 0.0f;

    /* Go-Explore: capture snapshot before auto-reset (skips done envs) */
    ex_maybe_capture_snapshot(b, idx);

    b->reward[idx] = reward;
    b->terminated[idx] = terminated;
    b->truncated[idx] = truncated;
    b->phase_buf[idx] = e->phase;
    b->step_count[idx] = e->episode_length;
    b->episode_return[idx] = e->episode_return;

    if (terminated || truncated) {
        ex_write_info(&b->last_episode_info[idx * EX_INFO_DIM], e);
    }

    if ((terminated || truncated) && auto_reset) {
        float terminal_return = e->episode_return;
        int32_t terminal_length = e->episode_length;
        ex_capture_terminal_buffers(b, idx, truncated && !terminated);
        ex_reset_env(b, idx, ++b->seed_counter);
        b->terminated[idx] = terminated;
        b->truncated[idx] = truncated;
        b->reward[idx] = reward;
        b->last_episode_return[idx] = terminal_return;
        b->last_episode_length[idx] = terminal_length;
        return;
    }

    if (terminated || truncated) {
        b->last_episode_return[idx] = e->episode_return;
        b->last_episode_length[idx] = e->episode_length;
    }

    ex_pack_step_outputs(b, idx, truncated && !terminated);
}

/* ─── Public API ─────────────────────────────────────────────────────── */

DMFastExactBatch *dmfast_exact_create(int32_t batch_size, uint64_t seed, int32_t max_steps) {
    if (batch_size <= 0 || max_steps <= 0) return NULL;

    DMFastExactBatch *b = (DMFastExactBatch *)calloc(1, sizeof(*b));
    if (!b) return NULL;

    b->batch_size = batch_size;
    b->max_steps = max_steps;
    b->seed_counter = seed;
    ex_default_reward_config(&b->rc);
    b->envs = (DMFastExactEnv *)calloc((size_t)batch_size, sizeof(*b->envs));
    b->obs = (float *)calloc((size_t)batch_size * EX_OBS_DIM, sizeof(float));
    b->action_mask = (uint8_t *)calloc((size_t)batch_size * EX_ACTION_DIM, sizeof(uint8_t));
    b->reward = (float *)calloc((size_t)batch_size, sizeof(float));
    b->terminated = (uint8_t *)calloc((size_t)batch_size, sizeof(uint8_t));
    b->truncated = (uint8_t *)calloc((size_t)batch_size, sizeof(uint8_t));
    b->phase_buf = (int32_t *)calloc((size_t)batch_size, sizeof(int32_t));
    b->step_count = (int32_t *)calloc((size_t)batch_size, sizeof(int32_t));
    b->episode_return = (float *)calloc((size_t)batch_size, sizeof(float));
    b->last_episode_return = (float *)calloc((size_t)batch_size, sizeof(float));
    b->last_episode_length = (int32_t *)calloc((size_t)batch_size, sizeof(int32_t));
    b->last_episode_info = (float *)calloc((size_t)batch_size * EX_INFO_DIM, sizeof(float));
    b->last_terminal_obs = (float *)calloc((size_t)batch_size * EX_OBS_DIM, sizeof(float));
    b->last_terminal_action_mask = (uint8_t *)calloc((size_t)batch_size * EX_ACTION_DIM, sizeof(uint8_t));
    b->info = (float *)calloc((size_t)batch_size * EX_INFO_DIM, sizeof(float));

    if (!b->envs || !b->obs || !b->action_mask || !b->reward ||
        !b->terminated || !b->truncated || !b->phase_buf ||
        !b->step_count || !b->episode_return || !b->last_episode_return ||
        !b->last_episode_length || !b->last_episode_info ||
        !b->last_terminal_obs || !b->last_terminal_action_mask || !b->info) {
        dmfast_exact_destroy(b);
        return NULL;
    }

    memset(b->snapshot_pool, 0, sizeof(b->snapshot_pool));
    b->snapshot_count = 0;

    dmfast_exact_reset_all(b, seed);
    return b;
}

void dmfast_exact_destroy(DMFastExactBatch *b) {
    if (!b) return;
    free(b->envs);
    free(b->obs);
    free(b->action_mask);
    free(b->reward);
    free(b->terminated);
    free(b->truncated);
    free(b->phase_buf);
    free(b->step_count);
    free(b->episode_return);
    free(b->last_episode_return);
    free(b->last_episode_length);
    free(b->last_episode_info);
    free(b->last_terminal_obs);
    free(b->last_terminal_action_mask);
    free(b->info);
    free(b);
}

void dmfast_exact_reset_all(DMFastExactBatch *b, uint64_t seed) {
    if (!b) return;
    b->seed_counter = seed;
    for (int i = 0; i < b->batch_size; ++i) {
        ex_reset_env(b, i, seed + (uint64_t)i * 1315423911ULL);
    }
    b->seed_counter = seed + (uint64_t)b->batch_size * 1315423911ULL;
}

void dmfast_exact_step(DMFastExactBatch *b, const int32_t *actions, int32_t auto_reset) {
    if (!b || !actions) return;
    for (int i = 0; i < b->batch_size; ++i) {
        ex_step_one(b, i, actions[i], auto_reset != 0);
    }
}

float *dmfast_exact_obs_ptr(DMFastExactBatch *b) { return b->obs; }
uint8_t *dmfast_exact_action_mask_ptr(DMFastExactBatch *b) { return b->action_mask; }
float *dmfast_exact_reward_ptr(DMFastExactBatch *b) { return b->reward; }
uint8_t *dmfast_exact_terminated_ptr(DMFastExactBatch *b) { return b->terminated; }
uint8_t *dmfast_exact_truncated_ptr(DMFastExactBatch *b) { return b->truncated; }
int32_t *dmfast_exact_phase_ptr(DMFastExactBatch *b) { return b->phase_buf; }
int32_t *dmfast_exact_step_count_ptr(DMFastExactBatch *b) { return b->step_count; }
float *dmfast_exact_episode_return_ptr(DMFastExactBatch *b) { return b->episode_return; }
float *dmfast_exact_last_episode_return_ptr(DMFastExactBatch *b) { return b->last_episode_return; }
int32_t *dmfast_exact_last_episode_length_ptr(DMFastExactBatch *b) { return b->last_episode_length; }
float *dmfast_exact_last_episode_info_ptr(DMFastExactBatch *b) { return b->last_episode_info; }
float *dmfast_exact_last_terminal_obs_ptr(DMFastExactBatch *b) { return b->last_terminal_obs; }
uint8_t *dmfast_exact_last_terminal_action_mask_ptr(DMFastExactBatch *b) { return b->last_terminal_action_mask; }
float *dmfast_exact_info_ptr(DMFastExactBatch *b) { return b->info; }
int32_t dmfast_exact_info_dim(void) { return EX_INFO_DIM; }

int32_t dmfast_exact_set_reward_param(DMFastExactBatch *b, const char *name, float value) {
    if (!b || !name) return 0;
    DMFastRewardConfig *rc = &b->rc;
    /* Combat */
    if      (!strcmp(name, "attack"))               rc->attack = value;
    else if (!strcmp(name, "attack_over_15"))        rc->attack_over_15 = value;
    else if (!strcmp(name, "attack_safe_sim"))       rc->attack_safe_sim = value;
    else if (!strcmp(name, "kill"))                  rc->kill = value;
    else if (!strcmp(name, "kill_over_15"))          rc->kill_over_15 = value;
    else if (!strcmp(name, "kill_high_level"))       rc->kill_high_level = value;
    else if (!strcmp(name, "kill_high_relative"))    rc->kill_high_relative = value;
    else if (!strcmp(name, "kill_high_relative_ratio")) rc->kill_high_relative_ratio = value;
    else if (!strcmp(name, "beast_high_level_threshold")) rc->beast_high_level_threshold = (int32_t)value;
    else if (!strcmp(name, "flee_penalty"))          rc->flee_penalty = value;
    else if (!strcmp(name, "flee_hard_bonus"))       rc->flee_hard_bonus = value;
    else if (!strcmp(name, "flee_hard_level_ratio")) rc->flee_hard_level_ratio = value;
    else if (!strcmp(name, "flee_easy_penalty"))     rc->flee_easy_penalty = value;
    else if (!strcmp(name, "flee_hard_sim"))          rc->flee_hard_sim = value;
    else if (!strcmp(name, "flee_safe_sim_penalty")) rc->flee_safe_sim_penalty = value;
    /* Economy */
    else if (!strcmp(name, "buy_potion"))            rc->buy_potion = value;
    else if (!strcmp(name, "potion_low_hp"))         rc->potion_low_hp = value;
    else if (!strcmp(name, "potion_high_hp_penalty")) rc->potion_high_hp_penalty = value;
    else if (!strcmp(name, "potion_low_hp_threshold")) rc->potion_low_hp_threshold = value;
    else if (!strcmp(name, "potion_high_hp_threshold")) rc->potion_high_hp_threshold = value;
    else if (!strcmp(name, "buy_item"))              rc->buy_item = value;
    else if (!strcmp(name, "buy_t1_item"))           rc->buy_t1_item = value;
    else if (!strcmp(name, "buy_weapon_t1"))         rc->buy_weapon_t1 = value;
    else if (!strcmp(name, "buy_quality_delta"))     rc->buy_quality_delta = value;
    else if (!strcmp(name, "buy_asset_bank_delta"))  rc->buy_asset_bank_delta = value;
    else if (!strcmp(name, "buy_asset_bank_delta_max_level")) rc->buy_asset_bank_delta_max_level = (int32_t)value;
    else if (!strcmp(name, "explore"))               rc->explore = value;
    else if (!strcmp(name, "stat_upgrade"))          rc->stat_upgrade = value;
    else if (!strcmp(name, "market_no_buy_penalty")) rc->market_no_buy_penalty = value;
    else if (!strcmp(name, "switch_item"))           rc->switch_item = value;
    else if (!strcmp(name, "switch_survivability"))  rc->switch_survivability = value;
    else if (!strcmp(name, "switch_surv_score_clip")) rc->switch_surv_score_clip = value;
    else if (!strcmp(name, "switch_surv_min_level")) rc->switch_surv_min_level = (int32_t)value;
    else if (!strcmp(name, "switch_surv_max_level")) rc->switch_surv_max_level = (int32_t)value;
    else if (!strcmp(name, "drop_item"))             rc->drop_item = value;
    /* Progression */
    else if (!strcmp(name, "level_up"))              rc->level_up = value;
    else if (!strcmp(name, "xp_shaping_scale"))      rc->xp_shaping_scale = value;
    else if (!strcmp(name, "death_penalty"))         rc->death_penalty = value;
    /* Milestones */
    else if (!strcmp(name, "milestone_kill_1"))      rc->milestone_kill_1 = value;
    else if (!strcmp(name, "milestone_kill_3"))      rc->milestone_kill_3 = value;
    else if (!strcmp(name, "milestone_kill_10"))     rc->milestone_kill_10 = value;
    else if (!strcmp(name, "milestone_kill_20"))     rc->milestone_kill_20 = value;
    else if (!strcmp(name, "milestone_kill_30"))     rc->milestone_kill_30 = value;
    else if (!strcmp(name, "milestone_level_8"))     rc->milestone_level_8 = value;
    else if (!strcmp(name, "milestone_level_12"))    rc->milestone_level_12 = value;
    else if (!strcmp(name, "milestone_level_16"))    rc->milestone_level_16 = value;
    else if (!strcmp(name, "milestone_level_20"))    rc->milestone_level_20 = value;
    else if (!strcmp(name, "milestone_level_25"))    rc->milestone_level_25 = value;
    else if (!strcmp(name, "milestone_level_35"))    rc->milestone_level_35 = value;
    /* Phase scaling (global multipliers) */
    else if (!strcmp(name, "phase_scale_economy"))   rc->phase_scale_economy = value;
    else if (!strcmp(name, "phase_scale_stat"))      rc->phase_scale_stat = value;
    else if (!strcmp(name, "phase_scale_attack"))    rc->phase_scale_attack = value;
    else if (!strcmp(name, "phase_scale_kill"))      rc->phase_scale_kill = value;
    else if (!strcmp(name, "phase_scale_high_kill")) rc->phase_scale_high_kill = value;
    else if (!strcmp(name, "phase_scale_flee"))      rc->phase_scale_flee = value;
    else if (!strcmp(name, "phase_scale_potion"))    rc->phase_scale_potion = value;
    else if (!strcmp(name, "phase_scale_xp"))       rc->phase_scale_xp = value;
    /* Macros */
    else if (!strcmp(name, "macro_fight_favorable")) rc->macro_fight_favorable = value;
    else if (!strcmp(name, "macro_fight_kill_resolve")) rc->macro_fight_kill_resolve = value;
    else if (!strcmp(name, "macro_flee_favorable"))  rc->macro_flee_favorable = value;
    else if (!strcmp(name, "macro_flee_hard_resolve")) rc->macro_flee_hard_resolve = value;
    /* Sim risk thresholds */
    else if (!strcmp(name, "sim_risk_safe_threshold")) rc->sim_risk_safe_threshold = value;
    else if (!strcmp(name, "sim_risk_hard_threshold")) rc->sim_risk_hard_threshold = value;
    /* Flee gate */
    else if (!strcmp(name, "mask_block_flee_on_safe")) rc->mask_block_flee_on_safe = (int32_t)value;
    else if (!strcmp(name, "mask_block_flee_safe_threshold")) rc->mask_block_flee_safe_threshold = value;
    else if (!strcmp(name, "mask_block_flee_min_hp_ratio")) rc->mask_block_flee_min_hp_ratio = value;
    else if (!strcmp(name, "mask_block_flee_max_beast_level_ratio")) rc->mask_block_flee_max_beast_level_ratio = value;
    /* ─── Extended reward params ─── */
    /* Attack context */
    else if (!strcmp(name, "attack_favorable"))         rc->attack_favorable = value;
    else if (!strcmp(name, "attack_favorable_level_ratio")) rc->attack_favorable_level_ratio = value;
    else if (!strcmp(name, "kill_streak"))              rc->kill_streak = value;
    /* Flee context */
    else if (!strcmp(name, "flee_streak_penalty"))      rc->flee_streak_penalty = value;
    else if (!strcmp(name, "flee_streak_grace"))        rc->flee_streak_grace = (int32_t)value;
    else if (!strcmp(name, "flee_easy_level_ratio"))    rc->flee_easy_level_ratio = value;
    else if (!strcmp(name, "flee_safe_sim_asset_ready_extra")) rc->flee_safe_sim_asset_ready_extra = value;
    else if (!strcmp(name, "flee_safe_sim_asset_ready_min_level")) rc->flee_safe_sim_asset_ready_min_level = (int32_t)value;
    else if (!strcmp(name, "flee_safe_sim_asset_ready_min_items")) rc->flee_safe_sim_asset_ready_min_items = (int32_t)value;
    else if (!strcmp(name, "flee_safe_sim_asset_ready_min_armor_slots")) rc->flee_safe_sim_asset_ready_min_armor_slots = (int32_t)value;
    else if (!strcmp(name, "flee_safe_sim_asset_ready_min_hp_ratio")) rc->flee_safe_sim_asset_ready_min_hp_ratio = value;
    else if (!strcmp(name, "flee_after_swap_penalty"))  rc->flee_after_swap_penalty = value;
    else if (!strcmp(name, "flee_after_swap_min_delta")) rc->flee_after_swap_min_delta = value;
    else if (!strcmp(name, "flee_phase_target_penalty")) rc->flee_phase_target_penalty = value;
    else if (!strcmp(name, "kill_phase_target_bonus"))  rc->kill_phase_target_bonus = value;
    else if (!strcmp(name, "phase_target_min_resolutions")) rc->phase_target_min_resolutions = (int32_t)value;
    /* Phase progress */
    else if (!strcmp(name, "phase_progress"))           rc->phase_progress = value;
    /* Stagnation & level floor */
    else if (!strcmp(name, "stagnation_pre16"))         rc->stagnation_pre16 = value;
    else if (!strcmp(name, "stagnation_pre16_level_thresh")) rc->stagnation_pre16_level_thresh = (int32_t)value;
    else if (!strcmp(name, "stagnation_pre16_action_thresh")) rc->stagnation_pre16_action_thresh = (int32_t)value;
    else if (!strcmp(name, "stagnation_pre16_max_penalty")) rc->stagnation_pre16_max_penalty = value;
    else if (!strcmp(name, "level_floor_penalty"))      rc->level_floor_penalty = value;
    else if (!strcmp(name, "level_floor_max_step_penalty")) rc->level_floor_max_step_penalty = value;
    /* Early farming */
    else if (!strcmp(name, "early_farm_penalty"))       rc->early_farm_penalty = value;
    else if (!strcmp(name, "early_farm_max_level"))     rc->early_farm_max_level = (int32_t)value;
    else if (!strcmp(name, "early_farm_risk_max"))      rc->early_farm_risk_max = value;
    else if (!strcmp(name, "early_farm_easy_level_ratio")) rc->early_farm_easy_level_ratio = value;
    else if (!strcmp(name, "early_farm_target_items"))  rc->early_farm_target_items = (int32_t)value;
    /* Stat shaping */
    else if (!strcmp(name, "stat_match_early_cha"))     rc->stat_match_early_cha = value;
    else if (!strcmp(name, "stat_match_level2_cha"))    rc->stat_match_level2_cha = value;
    else if (!strcmp(name, "stat_penalty_level2_non_cha")) rc->stat_penalty_level2_non_cha = value;
    else if (!strcmp(name, "stat_match_early_dex_strong")) rc->stat_match_early_dex_strong = value;
    else if (!strcmp(name, "stat_match_mid_cha"))       rc->stat_match_mid_cha = value;
    else if (!strcmp(name, "stat_match_mid_dex"))       rc->stat_match_mid_dex = value;
    else if (!strcmp(name, "stat_match_late_vit"))      rc->stat_match_late_vit = value;
    else if (!strcmp(name, "stat_match_end_str"))       rc->stat_match_end_str = value;
    else if (!strcmp(name, "stat_match_end_dex"))       rc->stat_match_end_dex = value;
    else if (!strcmp(name, "stat_match_end_vit"))       rc->stat_match_end_vit = value;
    else if (!strcmp(name, "stat_penalty_low_value_early")) rc->stat_penalty_low_value_early = value;
    /* Jewelry/weapon timing */
    else if (!strcmp(name, "first_weapon_online"))      rc->first_weapon_online = value;
    else if (!strcmp(name, "first_weapon_early"))       rc->first_weapon_early = value;
    else if (!strcmp(name, "first_weapon_early_max_action")) rc->first_weapon_early_max_action = (int32_t)value;
    else if (!strcmp(name, "first_weapon_early_max_level")) rc->first_weapon_early_max_level = (int32_t)value;
    else if (!strcmp(name, "jewelry_before_weapon"))    rc->jewelry_before_weapon = value;
    else if (!strcmp(name, "weapon_before_jewelry"))    rc->weapon_before_jewelry = value;
    else if (!strcmp(name, "early_jewelry_penalty"))    rc->early_jewelry_penalty = value;
    else if (!strcmp(name, "early_jewelry_min_action")) rc->early_jewelry_min_action = (int32_t)value;
    else if (!strcmp(name, "early_jewelry_min_level"))  rc->early_jewelry_min_level = (int32_t)value;
    else if (!strcmp(name, "jewelry_timing_good"))      rc->jewelry_timing_good = value;
    else if (!strcmp(name, "early_armor_slot_fill"))    rc->early_armor_slot_fill = value;
    else if (!strcmp(name, "early_armor_fill_max_level")) rc->early_armor_fill_max_level = (int32_t)value;
    else if (!strcmp(name, "new_combat_family"))        rc->new_combat_family = value;
    else if (!strcmp(name, "all_combat_families_early")) rc->all_combat_families_early = value;
    else if (!strcmp(name, "combat_family_diversity_max_level")) rc->combat_family_diversity_max_level = (int32_t)value;
    /* Equipment completion */
    else if (!strcmp(name, "full_equipment"))           rc->full_equipment = value;
    else if (!strcmp(name, "necklace_ring_equipped"))   rc->necklace_ring_equipped = value;
    /* Equipment armor */
    else if (!strcmp(name, "equipment_armor_delta"))    rc->equipment_armor_delta = value;
    else if (!strcmp(name, "equipment_armor_drop"))     rc->equipment_armor_drop = value;
    else if (!strcmp(name, "low_equipment_armor"))      rc->low_equipment_armor = value;
    else if (!strcmp(name, "low_equipment_armor_min_level")) rc->low_equipment_armor_min_level = (int32_t)value;
    else if (!strcmp(name, "low_equipment_armor_target_per_level")) rc->low_equipment_armor_target_per_level = value;
    /* Switch quality */
    else if (!strcmp(name, "switch_quality_delta"))     rc->switch_quality_delta = value;
    else if (!strcmp(name, "switch_quality_min_level")) rc->switch_quality_min_level = (int32_t)value;
    else if (!strcmp(name, "switch_quality_max_level")) rc->switch_quality_max_level = (int32_t)value;
    else if (!strcmp(name, "switch_worse_combat_sim"))  rc->switch_worse_combat_sim = value;
    /* Item greatness */
    else if (!strcmp(name, "item_to_level_15"))         rc->item_to_level_15 = value;
    else if (!strcmp(name, "weapon_t1_level_15"))       rc->weapon_t1_level_15 = value;
    /* Potion economics */
    else if (!strcmp(name, "potion_without_gear"))      rc->potion_without_gear = value;
    else if (!strcmp(name, "potion_without_gear_min_level")) rc->potion_without_gear_min_level = (int32_t)value;
    else if (!strcmp(name, "potion_without_gear_min_items")) rc->potion_without_gear_min_items = (int32_t)value;
    else if (!strcmp(name, "potion_without_gear_min_armor")) rc->potion_without_gear_min_armor = (int32_t)value;
    else if (!strcmp(name, "reduce_potion_cost_early")) rc->reduce_potion_cost_early = value;
    else if (!strcmp(name, "early_potion_cost_above_target")) rc->early_potion_cost_above_target = value;
    else if (!strcmp(name, "early_potion_cost_state_above_target")) rc->early_potion_cost_state_above_target = value;
    else if (!strcmp(name, "early_potion_cost_state_at_below_target")) rc->early_potion_cost_state_at_below_target = value;
    else if (!strcmp(name, "early_potion_level_threshold")) rc->early_potion_level_threshold = (int32_t)value;
    else if (!strcmp(name, "early_potion_cost_target")) rc->early_potion_cost_target = value;
    /* Buy targets */
    else if (!strcmp(name, "buy_early_target_item"))    rc->buy_early_target_item = value;
    else if (!strcmp(name, "buy_early_target_item_max_level")) rc->buy_early_target_item_max_level = (int32_t)value;
    /* Invalid action */
    else if (!strcmp(name, "invalid_action_penalty"))   rc->invalid_action_penalty = value;
    /* Macro cap hit */
    else if (!strcmp(name, "macro_repeat_cap_hit_favorable")) rc->macro_repeat_cap_hit_favorable = value;
    /* Phase boundaries */
    else if (!strcmp(name, "phase_very_early_end"))     rc->phase_very_early_end = (int32_t)value;
    else if (!strcmp(name, "phase_early_end"))          rc->phase_early_end = (int32_t)value;
    else if (!strcmp(name, "phase_mid_end"))            rc->phase_mid_end = (int32_t)value;
    else if (!strcmp(name, "phase_late_end"))           rc->phase_late_end = (int32_t)value;
    /* Level floor indexed params */
    else if (!strcmp(name, "level_floor_count"))        rc->level_floor_count = (int32_t)value;
    else if (!strncmp(name, "level_floor_budget_", 19)) {
        int idx = name[19] - '0';
        if (idx >= 0 && idx < 4) rc->level_floor_budgets[idx] = (int32_t)value;
    }
    else if (!strncmp(name, "level_floor_min_level_", 21)) {
        int idx = name[21] - '0';
        if (idx >= 0 && idx < 4) rc->level_floor_min_levels[idx] = (int32_t)value;
    }
    /* Early target item indexed params */
    else if (!strcmp(name, "early_target_item_count"))  rc->early_target_item_count = (int32_t)value;
    else if (!strncmp(name, "early_target_item_id_", 20)) {
        int idx = atoi(name + 20);
        if (idx >= 0 && idx < 25) rc->early_target_item_ids[idx] = (int32_t)value;
    }
    /* Market no-buy conditions */
    else if (!strcmp(name, "market_no_buy_max_level"))    rc->market_no_buy_max_level = (int32_t)value;
    else if (!strcmp(name, "market_no_buy_min_gold"))     rc->market_no_buy_min_gold = (int32_t)value;
    else if (!strcmp(name, "market_no_buy_min_options"))  rc->market_no_buy_min_options = (int32_t)value;
    else if (!strcmp(name, "market_no_buy_min_hp_ratio")) rc->market_no_buy_min_hp_ratio = value;
    /* Flee gain/threshold */
    else if (!strcmp(name, "flee_gain"))                  rc->flee_gain = value;
    else if (!strcmp(name, "flee_low_hp_threshold"))      rc->flee_low_hp_threshold = value;
    /* Flee mask gate */
    else if (!strcmp(name, "mask_block_flee_require_asset_ready")) rc->mask_block_flee_require_asset_ready = (int32_t)value;
    /* XP Subphase system */
    else if (!strcmp(name, "xp_subphase_enable"))         rc->xp_subphase_enable = (int32_t)value;
    else if (!strcmp(name, "xp_subphase_count"))          rc->xp_subphase_count = (int32_t)value;
    else if (!strncmp(name, "xp_subphase_bound_", 18)) {
        int idx = atoi(name + 18);
        if (idx >= 0 && idx < 10) rc->xp_subphase_bounds[idx] = (int32_t)value;
    }
    else if (!strncmp(name, "xp_subphase_flee_target_", 24)) {
        int idx = atoi(name + 24);
        if (idx >= 0 && idx < 11) rc->xp_subphase_flee_targets[idx] = value;
    }
    else if (!strncmp(name, "xp_subphase_attack_commit_mult_", 31)) {
        int idx = atoi(name + 31);
        if (idx >= 0 && idx < 11) rc->xp_subphase_attack_commit_mult[idx] = value;
    }
    else if (!strncmp(name, "xp_subphase_kill_target_mult_", 29)) {
        int idx = atoi(name + 29);
        if (idx >= 0 && idx < 11) rc->xp_subphase_kill_target_mult[idx] = value;
    }
    else if (!strncmp(name, "xp_subphase_flee_target_mult_", 29)) {
        int idx = atoi(name + 29);
        if (idx >= 0 && idx < 11) rc->xp_subphase_flee_target_mult[idx] = value;
    }
    else if (!strncmp(name, "xp_subphase_flee_easy_penalty_mult_", 35)) {
        int idx = atoi(name + 35);
        if (idx >= 0 && idx < 11) rc->xp_subphase_flee_easy_penalty_mult[idx] = value;
    }
    else if (!strncmp(name, "xp_subphase_market_no_buy_mult_", 31)) {
        int idx = atoi(name + 31);
        if (idx >= 0 && idx < 11) rc->xp_subphase_market_no_buy_mult[idx] = value;
    }
    else if (!strncmp(name, "xp_subphase_buy_reward_mult_", 28)) {
        int idx = atoi(name + 28);
        if (idx >= 0 && idx < 11) rc->xp_subphase_buy_reward_mult[idx] = value;
    }
    else if (!strncmp(name, "xp_subphase_potion_reward_mult_", 31)) {
        int idx = atoi(name + 31);
        if (idx >= 0 && idx < 11) rc->xp_subphase_potion_reward_mult[idx] = value;
    }
    /* Curriculum */
    else if (!strcmp(name, "curriculum_t1_weapon_prob"))     rc->curriculum_t1_weapon_prob = value;
    else if (!strcmp(name, "curriculum_snapshot_prob"))       rc->curriculum_snapshot_prob = value;
    else if (!strcmp(name, "curriculum_snapshot_min_xp"))     rc->curriculum_snapshot_min_xp = (int32_t)value;
    else if (!strcmp(name, "curriculum_snapshot_max_xp"))     rc->curriculum_snapshot_max_xp = (int32_t)value;
    else if (!strcmp(name, "curriculum_snapshot_min_hp_ratio")) rc->curriculum_snapshot_min_hp_ratio = value;
    else if (!strcmp(name, "max_level"))                     rc->max_level = value;
    else return 0;
    return 1;
}

/* ─── Vectorized masked action sampling ──────────────────────────────── */

void dmfast_exact_sample_masked_actions(DMFastExactBatch *b, int32_t *out_actions, uint64_t *rng_state) {
    const int32_t n = b->batch_size;
    const uint8_t *masks = b->action_mask;
    uint64_t state = *rng_state;

    for (int32_t i = 0; i < n; i++) {
        const uint8_t *m = masks + i * EX_ACTION_DIM;

        /* Count valid actions */
        int32_t count = 0;
        for (int32_t j = 0; j < EX_ACTION_DIM; j++) {
            count += (m[j] != 0);
        }
        if (count == 0) {
            out_actions[i] = 0;
            continue;
        }

        /* Pick a random valid action */
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);

        int32_t pick = (int32_t)(((uint32_t)(z >> 32)) % (uint32_t)count);
        int32_t seen = 0;
        for (int32_t j = 0; j < EX_ACTION_DIM; j++) {
            if (m[j] != 0) {
                if (seen == pick) {
                    out_actions[i] = j;
                    break;
                }
                seen++;
            }
        }
    }
    *rng_state = state;
}

/* ─── State injection API for serving ─────────────────────────────────
   Inject on-chain game state into env[0], repack obs + mask.
   Call dmfast_exact_obs_ptr / dmfast_exact_action_mask_ptr to read results.
   ──────────────────────────────────────────────────────────────────── */

void dmfast_exact_inject_state(
    DMFastExactBatch *b,
    int32_t phase,
    int32_t xp, int32_t health, int32_t gold,
    int32_t stat_upgrades_available, int32_t item_specials_seed,
    const int32_t *stats,            /* 7: str dex vit int wis cha luck */
    const int32_t *equipment_ids,    /* 8 */
    const int32_t *equipment_xp,     /* 8 */
    const int32_t *equipment_specials, /* 24 (8 slots × 3) */
    const int32_t *bag_ids,          /* 15 */
    const int32_t *bag_xp,           /* 15 */
    const int32_t *market_ids,       /* 25 */
    int32_t beast_id, int32_t beast_health, int32_t beast_starting_health,
    int32_t beast_level,
    const int32_t *beast_specials    /* 3 */
) {
    if (!b || b->batch_size < 1) return;
    DMFastExactEnv *e = &b->envs[0];

    /* Zero the env to clear stale state */
    memset(e, 0, sizeof(DMFastExactEnv));

    /* Core state */
    e->phase = phase;
    e->adventurer_xp = xp;
    e->health = health;
    e->gold = gold;
    e->stat_upgrades_available = stat_upgrades_available;
    e->item_specials_seed = item_specials_seed;

    /* Stats */
    for (int i = 0; i < EX_NUM_STATS && i < 7; ++i)
        e->stats[i] = stats[i];

    /* Equipment */
    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        e->equipment_ids[i] = equipment_ids[i];
        e->equipment_xp[i] = equipment_xp[i];
        for (int s = 0; s < EX_NUM_SPECIAL_FIELDS; ++s)
            e->equipment_specials[i * EX_NUM_SPECIAL_FIELDS + s] = equipment_specials[i * 3 + s];
    }

    /* TODO: compute special_stats from equipment specials (suffix bonuses).
       For now, caller can pass them via the stats array if known. */

    /* Bag */
    for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
        e->bag_ids[i] = bag_ids[i];
        e->bag_xp[i] = bag_xp[i];
    }

    /* Market */
    for (int i = 0; i < EX_NUM_MARKET_SLOTS; ++i)
        e->market_ids[i] = market_ids[i];

    /* Beast */
    e->beast_id = beast_id;
    e->beast_health = beast_health;
    e->beast_starting_health = beast_starting_health > 0 ? beast_starting_health : beast_health;
    if (beast_id > 0 && beast_health > 0) {
        /* Inline beast tier: T1=[1-5,26-30,51-55], T2=[6-10,31-35,56-60], etc */
        int bid = beast_id;
        if ((bid>=1&&bid<=5)||(bid>=26&&bid<31)||(bid>=51&&bid<56)) e->beast_tier = 1;
        else if ((bid>=6&&bid<11)||(bid>=31&&bid<36)||(bid>=56&&bid<61)) e->beast_tier = 2;
        else if ((bid>=11&&bid<16)||(bid>=36&&bid<41)||(bid>=61&&bid<66)) e->beast_tier = 3;
        else if ((bid>=16&&bid<21)||(bid>=41&&bid<46)||(bid>=66&&bid<71)) e->beast_tier = 4;
        else e->beast_tier = 5;
        /* Beast type: 1-25=magic(1), 26-50=blade(2), 51-75=bludgeon(3) */
        if (bid < 26) e->beast_type = 1;
        else if (bid < 51) e->beast_type = 2;
        else e->beast_type = 3;
    }
    e->beast_level = beast_level;
    for (int i = 0; i < EX_NUM_SPECIAL_FIELDS; ++i)
        e->beast_specials[i] = beast_specials[i];

    /* Pack obs and mask */
    ex_pack_obs(b, 0);
    ex_update_mask(b, 0);
}

/* ─── State export API (inverse of inject_state, for parity testing) ───
   Exports env[idx]'s fields into the caller-provided buffers. Layout matches
   the inject_state signature exactly so a roundtrip export→inject is a
   well-defined operation.
   ──────────────────────────────────────────────────────────────────── */

void dmfast_exact_export_state(
    DMFastExactBatch *b,
    int32_t idx,
    int32_t *out_phase,
    int32_t *out_xp, int32_t *out_health, int32_t *out_gold,
    int32_t *out_stat_upgrades_available, int32_t *out_item_specials_seed,
    int32_t *out_stats,            /* 7 */
    int32_t *out_special_stats,    /* 7 */
    int32_t *out_equipment_ids,    /* 8 */
    int32_t *out_equipment_xp,     /* 8 */
    int32_t *out_equipment_specials, /* 24 */
    int32_t *out_bag_ids,          /* 15 */
    int32_t *out_bag_xp,           /* 15 */
    int32_t *out_market_ids,       /* 25 */
    int32_t *out_beast_id, int32_t *out_beast_health,
    int32_t *out_beast_starting_health,
    int32_t *out_beast_level,
    int32_t *out_beast_specials    /* 3 */
) {
    if (!b || idx < 0 || idx >= b->batch_size) return;
    const DMFastExactEnv *e = &b->envs[idx];

    *out_phase = e->phase;
    *out_xp = e->adventurer_xp;
    *out_health = e->health;
    *out_gold = e->gold;
    *out_stat_upgrades_available = e->stat_upgrades_available;
    *out_item_specials_seed = e->item_specials_seed;

    for (int i = 0; i < 7; ++i) out_stats[i] = e->stats[i];
    for (int i = 0; i < 7; ++i) out_special_stats[i] = e->special_stats[i];

    for (int i = 0; i < EX_NUM_EQUIPMENT_SLOTS; ++i) {
        out_equipment_ids[i] = e->equipment_ids[i];
        out_equipment_xp[i] = e->equipment_xp[i];
        for (int s = 0; s < EX_NUM_SPECIAL_FIELDS; ++s)
            out_equipment_specials[i * 3 + s] = e->equipment_specials[i * EX_NUM_SPECIAL_FIELDS + s];
    }
    for (int i = 0; i < EX_NUM_BAG_SLOTS; ++i) {
        out_bag_ids[i] = e->bag_ids[i];
        out_bag_xp[i] = e->bag_xp[i];
    }
    for (int i = 0; i < EX_NUM_MARKET_SLOTS; ++i)
        out_market_ids[i] = e->market_ids[i];

    *out_beast_id = e->beast_id;
    *out_beast_health = e->beast_health;
    *out_beast_starting_health = e->beast_starting_health;
    *out_beast_level = e->beast_level;
    for (int i = 0; i < EX_NUM_SPECIAL_FIELDS; ++i)
        out_beast_specials[i] = e->beast_specials[i];
}

/* ─── State snapshot / restore (search & expert-iteration support) ────── */

int32_t dmfast_exact_state_size(void) {
    return (int32_t)sizeof(DMFastExactEnv);
}

/* Copy env idx's full state (POD struct incl. its rng) into out. */
void dmfast_exact_save_state(DMFastExactBatch *b, int32_t idx, uint8_t *out) {
    if (!b || !out || idx < 0 || idx >= b->batch_size) return;
    memcpy(out, &b->envs[idx], sizeof(DMFastExactEnv));
}

/* Restore env idx from a saved state and refresh its obs/mask buffers.
   done flags are cleared: a restored state is mid-episode by definition. */
void dmfast_exact_load_state(DMFastExactBatch *b, int32_t idx, const uint8_t *in) {
    if (!b || !in || idx < 0 || idx >= b->batch_size) return;
    memcpy(&b->envs[idx], in, sizeof(DMFastExactEnv));
    b->terminated[idx] = 0;
    b->truncated[idx] = 0;
    ex_pack_obs(b, idx);
    ex_update_mask(b, idx);
}

/* Broadcast ONE saved state into envs [0, count): the search primitive.
   When reseed != 0, each clone's rng is decorrelated (splitmix-style hash of
   clone index) so rollouts sample DIFFERENT futures — the correct semantics
   for planning under unknown upcoming randomness. reseed == 0 keeps the
   exact rng for bit-identical replay/debugging. */
void dmfast_exact_broadcast_state(DMFastExactBatch *b, const uint8_t *in,
                                  int32_t count, uint64_t reseed) {
    if (!b || !in) return;
    if (count < 0 || count > b->batch_size) count = b->batch_size;
    for (int32_t i = 0; i < count; ++i) {
        memcpy(&b->envs[i], in, sizeof(DMFastExactEnv));
        if (reseed != 0) {
            uint64_t z = b->envs[i].rng ^ (reseed + 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1));
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            b->envs[i].rng = z ^ (z >> 31);
        }
        b->terminated[i] = 0;
        b->truncated[i] = 0;
        ex_pack_obs(b, i);
        ex_update_mask(b, i);
    }
}
