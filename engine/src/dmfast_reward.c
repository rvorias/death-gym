#include "dmfast_internal.h"

enum {
    DMFAST_REWARD_CFG_ATTACK = 0,
    DMFAST_REWARD_CFG_EXPLORE = 1,
    DMFAST_REWARD_CFG_FLEE_PENALTY = 2,
    DMFAST_REWARD_CFG_BUY_ITEM = 3,
    DMFAST_REWARD_CFG_BUY_POTION = 4,
    DMFAST_REWARD_CFG_BUY_T1_ITEM = 5,
    DMFAST_REWARD_CFG_BUY_WEAPON_T1 = 6,
    DMFAST_REWARD_CFG_BUY_QUALITY_DELTA = 7,
    DMFAST_REWARD_CFG_BUY_ASSET_BANK_DELTA = 8,
    DMFAST_REWARD_CFG_BUY_ASSET_BANK_DELTA_MAX_LEVEL = 9,
    DMFAST_REWARD_CFG_SWITCH_ITEM = 10,
    DMFAST_REWARD_CFG_SWITCH_SURVIVABILITY = 11,
    DMFAST_REWARD_CFG_SWITCH_SURV_SCORE_CLIP = 12,
    DMFAST_REWARD_CFG_SWITCH_SURV_MIN_LEVEL = 13,
    DMFAST_REWARD_CFG_SWITCH_SURV_MAX_LEVEL = 14,
    DMFAST_REWARD_CFG_DROP_ITEM = 15,
    DMFAST_REWARD_CFG_STAT_UPGRADE = 16,
    DMFAST_REWARD_CFG_KILL = 17,
    DMFAST_REWARD_CFG_LEVEL_UP = 18,
    DMFAST_REWARD_CFG_XP_SHAPING_SCALE = 19,
    DMFAST_REWARD_CFG_DEATH_PENALTY = 20,
    DMFAST_REWARD_CFG_ATTACK_SAFE_SIM = 21,
    DMFAST_REWARD_CFG_FLEE_SAFE_SIM_PENALTY = 22,
    DMFAST_REWARD_CFG_FLEE_HARD_SIM = 23,
    DMFAST_REWARD_CFG_SIM_RISK_SAFE_THRESHOLD = 24,
    DMFAST_REWARD_CFG_SIM_RISK_HARD_THRESHOLD = 25,
    DMFAST_REWARD_CFG_PHASE_VERY_EARLY_END = 26,
    DMFAST_REWARD_CFG_PHASE_EARLY_END = 27,
    DMFAST_REWARD_CFG_PHASE_MID_END = 28,
    DMFAST_REWARD_CFG_PHASE_LATE_END = 29,
    DMFAST_REWARD_CFG_PHASE_SCALE_ECONOMY = 30,
    DMFAST_REWARD_CFG_PHASE_SCALE_STAT = 31,
    DMFAST_REWARD_CFG_PHASE_SCALE_ATTACK = 32,
    DMFAST_REWARD_CFG_PHASE_SCALE_KILL = 33,
    DMFAST_REWARD_CFG_PHASE_SCALE_HIGH_KILL = 34,
    DMFAST_REWARD_CFG_PHASE_SCALE_FLEE = 35,
    DMFAST_REWARD_CFG_PHASE_SCALE_POTION = 36,
    DMFAST_REWARD_CFG_PHASE_SCALE_XP = 37,
    DMFAST_REWARD_CFG_ITEM_XP_GAIN = 38,
    DMFAST_REWARD_CFG_INVALID_ACTION_PENALTY = 39,
    /* Death-audit (2026-06-11): 55% of deaths were explore-at-low-HP, 38%
     * failed flees downstream of the same mistake. Penalty scales linearly
     * with the HP deficit below the threshold (ratio of max health). */
    DMFAST_REWARD_CFG_EXPLORE_LOW_HP = 40,
    DMFAST_REWARD_CFG_EXPLORE_LOW_HP_THRESHOLD = 41,
    /* Death-audit v2 (2026-06-11): deaths happen at ~FULL hp because the
     * policy under-invests vitality (max_health ~150 at level 13). Extra
     * bonus when the chosen stat upgrade is vitality. */
    DMFAST_REWARD_CFG_STAT_UPGRADE_VIT = 42,
    /* Gear-compounding package (2026-06-12): suffixes at greatness 15 are
     * the "forgiving regime" threshold (humans max armor g=20 in ALL
     * quartiles; the agent dies at armor g~13). Bonus per held item
     * crossing g15. */
    DMFAST_REWARD_CFG_SUFFIX_UNLOCK = 43,
    /* Kill-gated item XP (2026-06-12): the rules pay items 2x on KILLS but
     * 1x on obstacles; the flat item_xp_gain term was arbitraged via the
     * riskless obstacle channel (gear package -15.4). This term prices only
     * item XP earned on kill steps — the leak-free channel. */
    DMFAST_REWARD_CFG_ITEM_XP_KILL_GAIN = 44,
    /* Gold-hoard penalty (2026-06-16, user): early game, penalize holding gold
     * while healthy -> forces gold->gear conversion (potions are mask-gated on
     * missing HP, so a healthy agent can only spend gold on items). Active when
     * post_level < LEVEL and hp_ratio > HP; penalty = coef * gold_norm. */
    DMFAST_REWARD_CFG_GOLD_HOARD = 45,
    DMFAST_REWARD_CFG_GOLD_HOARD_LEVEL = 46,
    DMFAST_REWARD_CFG_GOLD_HOARD_HP = 47,
    /* Armor-score objective (2026-06-16, user 2-stage curriculum): reward the
     * delta of total armor score = sum over equipped+bag armor of
     * greatness*(6-tier). Stage 1 truncates at level 20 -> maximize armor by
     * the human gear-completion crossover. obs packs (6-tier)=field1*5,
     * greatness=field8*20+1. */
    DMFAST_REWARD_CFG_ARMOR_SCORE = 48,
    DMFAST_REWARD_CFG_COUNT = 49,
};

enum {
    DMFAST_REWARD_INFO_DIM = 9,
    DMFAST_REWARD_INFO_LEVEL = 0,
    DMFAST_REWARD_INFO_XP = 1,
    DMFAST_REWARD_INFO_HEALTH = 2,
    DMFAST_REWARD_INFO_GOLD = 3,
    DMFAST_REWARD_INFO_BEAST_LEVEL = 4,
    DMFAST_REWARD_INFO_BEASTS_KILLED = 5,
    DMFAST_REWARD_INFO_ITEMS_BOUGHT = 6,
    DMFAST_REWARD_INFO_POTIONS_BOUGHT = 7,
};

enum {
    DMFAST_REWARD_OBS_EQUIPMENT_START = 14,
    DMFAST_REWARD_OBS_BAG_START = 86,
    DMFAST_REWARD_OBS_SIM_START = 453,
};

enum {
    DMFAST_REWARD_ACTION_EXPLORE = 0,
    DMFAST_REWARD_ACTION_ATTACK = 1,
    DMFAST_REWARD_ACTION_FLEE = 2,
    DMFAST_REWARD_ACTION_STAT_VIT = 53,  /* stat block 51-56: str,dex,VIT,int,wis,cha */
};

enum {
    DMFAST_REWARD_ITEM_ID = 0,
    DMFAST_REWARD_ITEM_TIER = 1,
    DMFAST_REWARD_ITEM_SLOT = 3,
    DMFAST_REWARD_ITEM_XP = 7,
    DMFAST_REWARD_ITEM_GREATNESS = 8,
};

static const float DMFAST_REWARD_PHASE_TABLE_ECONOMY[5] = {1.7f, 1.35f, 1.0f, 0.6f, 0.35f};
static const float DMFAST_REWARD_PHASE_TABLE_STAT[5] = {1.8f, 1.4f, 1.1f, 0.8f, 0.6f};
static const float DMFAST_REWARD_PHASE_TABLE_ATTACK[5] = {0.75f, 1.0f, 1.2f, 1.35f, 1.5f};
static const float DMFAST_REWARD_PHASE_TABLE_KILL[5] = {0.8f, 1.0f, 1.3f, 1.65f, 2.0f};
static const float DMFAST_REWARD_PHASE_TABLE_FLEE[5] = {0.7f, 1.0f, 1.25f, 1.5f, 1.8f};
static const float DMFAST_REWARD_PHASE_TABLE_POTION[5] = {0.8f, 1.0f, 1.2f, 1.35f, 1.5f};
static const float DMFAST_REWARD_PHASE_TABLE_XP[5] = {0.8f, 1.0f, 1.1f, 1.25f, 1.4f};

static int32_t dmfast_reward_phase_index(const float *cfg, float level) {
    if (level <= cfg[DMFAST_REWARD_CFG_PHASE_VERY_EARLY_END]) return 0;
    if (level <= cfg[DMFAST_REWARD_CFG_PHASE_EARLY_END]) return 1;
    if (level <= cfg[DMFAST_REWARD_CFG_PHASE_MID_END]) return 2;
    if (level <= cfg[DMFAST_REWARD_CFG_PHASE_LATE_END]) return 3;
    return 4;
}

static float dmfast_reward_phase_scale(
    const float *cfg,
    const float *table,
    float level,
    int32_t scale_idx
) {
    int32_t phase_idx = dmfast_reward_phase_index(cfg, level);
    return table[phase_idx] * cfg[scale_idx];
}

static float dmfast_reward_item_quality_score(const float *item_row) {
    int32_t item_id = (int32_t)(item_row[DMFAST_REWARD_ITEM_ID] + 0.5f);
    if (item_id <= 0) return 0.0f;

    int32_t slot = (int32_t)(item_row[DMFAST_REWARD_ITEM_SLOT] + 0.5f);
    float slot_bonus = 0.0f;
    if (slot == DMFAST_SLOT_WEAPON) {
        slot_bonus = 1.0f;
    } else if (
        slot == DMFAST_SLOT_CHEST || slot == DMFAST_SLOT_HEAD || slot == DMFAST_SLOT_WAIST ||
        slot == DMFAST_SLOT_FOOT || slot == DMFAST_SLOT_HAND
    ) {
        slot_bonus = 0.6f;
    } else if (slot == DMFAST_SLOT_NECK || slot == DMFAST_SLOT_RING) {
        slot_bonus = 0.3f;
    }

    return 10.0f * (item_row[DMFAST_REWARD_ITEM_TIER] * 5.0f)
         + (1.0f + item_row[DMFAST_REWARD_ITEM_GREATNESS] * 20.0f)
         + slot_bonus;
}

static const float *dmfast_reward_equipment_item_for_slot(const float *obs_row, int32_t slot) {
    const float *equipment = obs_row + DMFAST_REWARD_OBS_EQUIPMENT_START;
    for (int32_t idx = 0; idx < DMFAST_NUM_EQUIPMENT_SLOTS; ++idx) {
        const float *item_row = equipment + idx * DMFAST_NUM_ITEM_FIELDS;
        int32_t item_slot = (int32_t)(item_row[DMFAST_REWARD_ITEM_SLOT] + 0.5f);
        if (item_slot == slot) return item_row;
    }
    return NULL;
}

static const float *dmfast_reward_find_new_item(const float *pre_obs_row, const float *post_obs_row) {
    const float *pre_equipment = pre_obs_row + DMFAST_REWARD_OBS_EQUIPMENT_START;
    const float *pre_bag = pre_obs_row + DMFAST_REWARD_OBS_BAG_START;
    const float *post_equipment = post_obs_row + DMFAST_REWARD_OBS_EQUIPMENT_START;
    const float *post_bag = post_obs_row + DMFAST_REWARD_OBS_BAG_START;

    for (int32_t post_idx = 0; post_idx < DMFAST_NUM_EQUIPMENT_SLOTS + DMFAST_NUM_BAG; ++post_idx) {
        const float *post_item = (post_idx < DMFAST_NUM_EQUIPMENT_SLOTS)
            ? post_equipment + post_idx * DMFAST_NUM_ITEM_FIELDS
            : post_bag + (post_idx - DMFAST_NUM_EQUIPMENT_SLOTS) * DMFAST_NUM_ITEM_FIELDS;
        int32_t post_id = (int32_t)(post_item[DMFAST_REWARD_ITEM_ID] + 0.5f);
        if (post_id <= 0) continue;

        int found = 0;
        for (int32_t pre_idx = 0; pre_idx < DMFAST_NUM_EQUIPMENT_SLOTS + DMFAST_NUM_BAG; ++pre_idx) {
            const float *pre_item = (pre_idx < DMFAST_NUM_EQUIPMENT_SLOTS)
                ? pre_equipment + pre_idx * DMFAST_NUM_ITEM_FIELDS
                : pre_bag + (pre_idx - DMFAST_NUM_EQUIPMENT_SLOTS) * DMFAST_NUM_ITEM_FIELDS;
            int32_t pre_id = (int32_t)(pre_item[DMFAST_REWARD_ITEM_ID] + 0.5f);
            if (pre_id == post_id) {
                found = 1;
                break;
            }
        }
        if (!found) return post_item;
    }
    return NULL;
}

static float dmfast_reward_best_slot_asset_score(const float *obs_row) {
    float best_by_slot[9] = {0};
    const float *equipment = obs_row + DMFAST_REWARD_OBS_EQUIPMENT_START;
    const float *bag = obs_row + DMFAST_REWARD_OBS_BAG_START;

    for (int32_t idx = 0; idx < DMFAST_NUM_EQUIPMENT_SLOTS + DMFAST_NUM_BAG; ++idx) {
        const float *item_row = (idx < DMFAST_NUM_EQUIPMENT_SLOTS)
            ? equipment + idx * DMFAST_NUM_ITEM_FIELDS
            : bag + (idx - DMFAST_NUM_EQUIPMENT_SLOTS) * DMFAST_NUM_ITEM_FIELDS;
        int32_t item_id = (int32_t)(item_row[DMFAST_REWARD_ITEM_ID] + 0.5f);
        if (item_id <= 0) continue;
        int32_t slot = (int32_t)(item_row[DMFAST_REWARD_ITEM_SLOT] + 0.5f);
        if (slot <= DMFAST_SLOT_NONE || slot > DMFAST_SLOT_RING) continue;
        float score = dmfast_reward_item_quality_score(item_row);
        if (score > best_by_slot[slot]) best_by_slot[slot] = score;
    }

    float total = 0.0f;
    for (int32_t slot = 0; slot <= DMFAST_SLOT_RING; ++slot) total += best_by_slot[slot];
    return total;
}

static float dmfast_reward_held_item_xp_total(const float *obs_row) {
    float total = 0.0f;
    const float *equipment = obs_row + DMFAST_REWARD_OBS_EQUIPMENT_START;
    const float *bag = obs_row + DMFAST_REWARD_OBS_BAG_START;

    for (int32_t idx = 0; idx < DMFAST_NUM_EQUIPMENT_SLOTS + DMFAST_NUM_BAG; ++idx) {
        const float *item_row = (idx < DMFAST_NUM_EQUIPMENT_SLOTS)
            ? equipment + idx * DMFAST_NUM_ITEM_FIELDS
            : bag + (idx - DMFAST_NUM_EQUIPMENT_SLOTS) * DMFAST_NUM_ITEM_FIELDS;
        int32_t item_id = (int32_t)(item_row[DMFAST_REWARD_ITEM_ID] + 0.5f);
        if (item_id <= 0) continue;
        total += item_row[DMFAST_REWARD_ITEM_XP] * 400.0f;
    }

    return total;
}

/* Held items (equipment + bag) at suffix greatness: obs item field 8 packs
 * (greatness - 1) / 20, so g >= 15 <=> field >= 0.7 (use 0.699 for float). */
static int dmfast_reward_suffix_count(const float *obs_row) {
    int count = 0;
    int base, i;
    /* equipment: 8 items from obs[14]; bag: 15 items from obs[86] */
    for (i = 0; i < 8; ++i) {
        base = 14 + i * 9;
        if (obs_row[base] > 0.5f && obs_row[base + 8] >= 0.699f) ++count;
    }
    for (i = 0; i < 15; ++i) {
        base = 86 + i * 9;
        if (obs_row[base] > 0.5f && obs_row[base + 8] >= 0.699f) ++count;
    }
    return count;
}

static float dmfast_reward_armor_score(const float *obs_row) {
    /* sum over equipped (8) + bag (15) of armor pieces (slot chest..hand):
     * greatness*(6-tier). greatness = field8*20+1; (6-tier) = field1*5. */
    float total = 0.0f;
    int k;
    for (k = 0; k < DMFAST_NUM_EQUIPMENT_SLOTS + DMFAST_NUM_BAG; ++k) {
        int base = (k < DMFAST_NUM_EQUIPMENT_SLOTS)
            ? DMFAST_REWARD_OBS_EQUIPMENT_START + k * DMFAST_NUM_ITEM_FIELDS
            : DMFAST_REWARD_OBS_BAG_START + (k - DMFAST_NUM_EQUIPMENT_SLOTS) * DMFAST_NUM_ITEM_FIELDS;
        const float *it = obs_row + base;
        if (it[DMFAST_REWARD_ITEM_ID] < 0.5f) continue;
        int slot = (int)(it[DMFAST_REWARD_ITEM_SLOT] + 0.5f);
        if (slot >= DMFAST_SLOT_CHEST && slot <= DMFAST_SLOT_HAND) {
            float greatness = it[DMFAST_REWARD_ITEM_GREATNESS] * 20.0f + 1.0f;
            float tierw = it[DMFAST_REWARD_ITEM_TIER] * 5.0f;  /* (6-tier) */
            total += greatness * tierw;
        }
    }
    return total;
}

static float dmfast_reward_combat_swap_score(const float *obs_row) {
    const float *sim = obs_row + DMFAST_REWARD_OBS_SIM_START;
    return 1.6f * sim[0] - 1.0f * sim[2] + 0.35f * sim[3];
}

void dmfast_reward_compute_batch(
    const float *config,
    const float *prev_obs,
    const float *prev_info,
    const uint8_t *prev_mask,
    const int32_t *actions,
    const float *post_obs,
    const float *post_info,
    const uint8_t *terminated,
    const uint8_t *truncated,
    int32_t count,
    float *out_rewards
) {
    (void)truncated;
    if (!config || !prev_obs || !prev_info || !prev_mask || !actions ||
        !post_obs || !post_info || !terminated || !out_rewards) {
        return;
    }

    for (int32_t idx = 0; idx < count; ++idx) {
        const float *pre_obs_row = prev_obs + idx * DMFAST_OBS_DIM;
        const float *pre_info_row = prev_info + idx * DMFAST_REWARD_INFO_DIM;
        const uint8_t *pre_mask_row = prev_mask + idx * DMFAST_ACTION_DIM;
        const float *post_obs_row = post_obs + idx * DMFAST_OBS_DIM;
        const float *post_info_row = post_info + idx * DMFAST_REWARD_INFO_DIM;
        int32_t action = actions[idx];
        int valid = action >= 0 && action < DMFAST_ACTION_DIM && pre_mask_row[action] != 0;

        float reward = valid ? 0.0f : config[DMFAST_REWARD_CFG_INVALID_ACTION_PENALTY];
        float prev_level = pre_info_row[DMFAST_REWARD_INFO_LEVEL];
        float post_level = post_info_row[DMFAST_REWARD_INFO_LEVEL];
        float prev_beast_level = pre_info_row[DMFAST_REWARD_INFO_BEAST_LEVEL];
        float xp_delta = post_info_row[DMFAST_REWARD_INFO_XP] - pre_info_row[DMFAST_REWARD_INFO_XP];
        float kill_delta = post_info_row[DMFAST_REWARD_INFO_BEASTS_KILLED] - pre_info_row[DMFAST_REWARD_INFO_BEASTS_KILLED];
        float item_delta = post_info_row[DMFAST_REWARD_INFO_ITEMS_BOUGHT] - pre_info_row[DMFAST_REWARD_INFO_ITEMS_BOUGHT];
        float potion_delta = post_info_row[DMFAST_REWARD_INFO_POTIONS_BOUGHT] - pre_info_row[DMFAST_REWARD_INFO_POTIONS_BOUGHT];
        float level_delta = post_level - prev_level;
        int32_t prev_phase = (int32_t)(pre_obs_row[0] + 0.5f);
        float prev_sim_risk = pre_obs_row[DMFAST_REWARD_OBS_SIM_START + 2];

        if (xp_delta > 0.0f) {
            reward += xp_delta * config[DMFAST_REWARD_CFG_XP_SHAPING_SCALE] / (float)DMFAST_MAX_ADVENTURER_XP
                * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_XP, post_level, DMFAST_REWARD_CFG_PHASE_SCALE_XP);
        }
        if (config[DMFAST_REWARD_CFG_ITEM_XP_GAIN] != 0.0f) {
            float item_xp_delta = dmfast_reward_held_item_xp_total(post_obs_row) - dmfast_reward_held_item_xp_total(pre_obs_row);
            if (item_xp_delta > 0.0f) {
                reward += item_xp_delta * config[DMFAST_REWARD_CFG_ITEM_XP_GAIN];
            }
        }
        if (config[DMFAST_REWARD_CFG_ITEM_XP_KILL_GAIN] != 0.0f && kill_delta > 0.0f) {
            float kxp = dmfast_reward_held_item_xp_total(post_obs_row) - dmfast_reward_held_item_xp_total(pre_obs_row);
            if (kxp > 0.0f) {
                reward += kxp * config[DMFAST_REWARD_CFG_ITEM_XP_KILL_GAIN];
            }
        }
        if (config[DMFAST_REWARD_CFG_GOLD_HOARD] != 0.0f) {
            float mh = post_obs_row[2];
            float hpr = mh > 0.0f ? post_obs_row[1] / mh : 0.0f;
            if (post_level < config[DMFAST_REWARD_CFG_GOLD_HOARD_LEVEL] &&
                hpr > config[DMFAST_REWARD_CFG_GOLD_HOARD_HP]) {
                reward += config[DMFAST_REWARD_CFG_GOLD_HOARD] * post_obs_row[4];  /* gold/511 */
            }
        }
        if (config[DMFAST_REWARD_CFG_ARMOR_SCORE] != 0.0f) {
            float d = dmfast_reward_armor_score(post_obs_row) - dmfast_reward_armor_score(pre_obs_row);
            reward += config[DMFAST_REWARD_CFG_ARMOR_SCORE] * d;
        }
        if (config[DMFAST_REWARD_CFG_SUFFIX_UNLOCK] != 0.0f) {
            int sd = dmfast_reward_suffix_count(post_obs_row) - dmfast_reward_suffix_count(pre_obs_row);
            if (sd > 0) {
                reward += (float)sd * config[DMFAST_REWARD_CFG_SUFFIX_UNLOCK];
            }
        }
        if (level_delta > 0.0f) {
            reward += level_delta * config[DMFAST_REWARD_CFG_LEVEL_UP];
        }
        if (kill_delta > 0.0f) {
            reward += kill_delta * prev_beast_level * config[DMFAST_REWARD_CFG_KILL]
                * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_KILL, prev_level, DMFAST_REWARD_CFG_PHASE_SCALE_KILL);
        }
        if (item_delta > 0.0f) {
            reward += item_delta * config[DMFAST_REWARD_CFG_BUY_ITEM]
                * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_ECONOMY, post_level, DMFAST_REWARD_CFG_PHASE_SCALE_ECONOMY);
        }
        if (potion_delta > 0.0f) {
            reward += potion_delta * config[DMFAST_REWARD_CFG_BUY_POTION]
                * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_POTION, post_level, DMFAST_REWARD_CFG_PHASE_SCALE_POTION);
        }

        if (valid && action == DMFAST_REWARD_ACTION_EXPLORE) {
            reward += config[DMFAST_REWARD_CFG_EXPLORE]
                * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_ECONOMY, post_level, DMFAST_REWARD_CFG_PHASE_SCALE_ECONOMY);
            /* obs[1] = health/1023, obs[2] = max_health/1023 -> ratio */
            if (config[DMFAST_REWARD_CFG_EXPLORE_LOW_HP] != 0.0f) {
                float thr = config[DMFAST_REWARD_CFG_EXPLORE_LOW_HP_THRESHOLD];
                float hpr = pre_obs_row[2] > 0.0f ? pre_obs_row[1] / pre_obs_row[2] : 1.0f;
                if (thr > 0.0f && hpr < thr) {
                    reward += config[DMFAST_REWARD_CFG_EXPLORE_LOW_HP] * (thr - hpr) / thr;
                }
            }
        }
        if (valid && action == DMFAST_REWARD_ACTION_ATTACK) {
            float attack_scale = dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_ATTACK, prev_level, DMFAST_REWARD_CFG_PHASE_SCALE_ATTACK);
            reward += config[DMFAST_REWARD_CFG_ATTACK] * attack_scale;
            if (prev_sim_risk > 0.0f && prev_sim_risk <= config[DMFAST_REWARD_CFG_SIM_RISK_SAFE_THRESHOLD]) {
                reward += config[DMFAST_REWARD_CFG_ATTACK_SAFE_SIM] * attack_scale;
            }
        }
        if (valid && action == DMFAST_REWARD_ACTION_FLEE) {
            float flee_scale = dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_FLEE, prev_level, DMFAST_REWARD_CFG_PHASE_SCALE_FLEE);
            reward += config[DMFAST_REWARD_CFG_FLEE_PENALTY] * flee_scale;
            if (prev_sim_risk > 0.0f && prev_sim_risk <= config[DMFAST_REWARD_CFG_SIM_RISK_SAFE_THRESHOLD]) {
                reward += config[DMFAST_REWARD_CFG_FLEE_SAFE_SIM_PENALTY] * flee_scale;
            }
            if (prev_sim_risk >= config[DMFAST_REWARD_CFG_SIM_RISK_HARD_THRESHOLD]) {
                reward += config[DMFAST_REWARD_CFG_FLEE_HARD_SIM] * flee_scale;
            }
        }
        if (valid && prev_phase == DMFAST_PHASE_UPGRADE) {
            float stat_scale = dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_STAT, prev_level, DMFAST_REWARD_CFG_PHASE_SCALE_STAT);
            reward += config[DMFAST_REWARD_CFG_STAT_UPGRADE] * stat_scale;
            if (action == DMFAST_REWARD_ACTION_STAT_VIT) {
                reward += config[DMFAST_REWARD_CFG_STAT_UPGRADE_VIT] * stat_scale;
            }
        }
        if (valid && prev_phase == DMFAST_PHASE_DROP) {
            reward += config[DMFAST_REWARD_CFG_DROP_ITEM]
                * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_ECONOMY, post_level, DMFAST_REWARD_CFG_PHASE_SCALE_ECONOMY);
        }
        if (valid && prev_phase == DMFAST_PHASE_EQUIP) {
            reward += config[DMFAST_REWARD_CFG_SWITCH_ITEM]
                * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_ECONOMY, post_level, DMFAST_REWARD_CFG_PHASE_SCALE_ECONOMY);
        }
        if (terminated[idx] != 0) {
            reward += config[DMFAST_REWARD_CFG_DEATH_PENALTY];
        }

        if (item_delta > 0.0f) {
            const float *new_item = dmfast_reward_find_new_item(pre_obs_row, post_obs_row);
            float economy_scale = dmfast_reward_phase_scale(
                config, DMFAST_REWARD_PHASE_TABLE_ECONOMY, post_level, DMFAST_REWARD_CFG_PHASE_SCALE_ECONOMY
            );
            if (new_item) {
                int32_t slot = (int32_t)(new_item[DMFAST_REWARD_ITEM_SLOT] + 0.5f);
                float tier_norm = new_item[DMFAST_REWARD_ITEM_TIER];
                if (tier_norm >= 0.999f) {
                    reward += config[DMFAST_REWARD_CFG_BUY_T1_ITEM] * economy_scale;
                    if (slot == DMFAST_SLOT_WEAPON) {
                        reward += config[DMFAST_REWARD_CFG_BUY_WEAPON_T1] * economy_scale;
                    }
                }
                if (config[DMFAST_REWARD_CFG_BUY_QUALITY_DELTA] != 0.0f && slot != DMFAST_SLOT_NONE) {
                    const float *equipped_pre = dmfast_reward_equipment_item_for_slot(pre_obs_row, slot);
                    float pre_quality = equipped_pre ? dmfast_reward_item_quality_score(equipped_pre) : 0.0f;
                    float delta = dmfast_reward_item_quality_score(new_item) - pre_quality;
                    if (delta > 0.0f) {
                        reward += config[DMFAST_REWARD_CFG_BUY_QUALITY_DELTA] * delta * economy_scale;
                    }
                }
            }
            if (config[DMFAST_REWARD_CFG_BUY_ASSET_BANK_DELTA] != 0.0f &&
                post_level <= config[DMFAST_REWARD_CFG_BUY_ASSET_BANK_DELTA_MAX_LEVEL]) {
                float delta = dmfast_reward_best_slot_asset_score(post_obs_row) - dmfast_reward_best_slot_asset_score(pre_obs_row);
                if (delta > 0.0f) {
                    reward += config[DMFAST_REWARD_CFG_BUY_ASSET_BANK_DELTA] * delta * economy_scale;
                }
            }
        }

        if (valid && prev_phase == DMFAST_PHASE_EQUIP &&
            config[DMFAST_REWARD_CFG_SWITCH_SURVIVABILITY] != 0.0f &&
            post_level >= config[DMFAST_REWARD_CFG_SWITCH_SURV_MIN_LEVEL] &&
            post_level <= config[DMFAST_REWARD_CFG_SWITCH_SURV_MAX_LEVEL]) {
            float delta = dmfast_reward_combat_swap_score(post_obs_row) - dmfast_reward_combat_swap_score(pre_obs_row);
            float clip = config[DMFAST_REWARD_CFG_SWITCH_SURV_SCORE_CLIP];
            if (clip > 0.0f) {
                if (delta > clip) delta = clip;
                if (delta < -clip) delta = -clip;
            }
            if (delta > 0.0f) {
                reward += config[DMFAST_REWARD_CFG_SWITCH_SURVIVABILITY] * delta
                    * dmfast_reward_phase_scale(config, DMFAST_REWARD_PHASE_TABLE_ATTACK, prev_level, DMFAST_REWARD_CFG_PHASE_SCALE_ATTACK);
            }
        }

        out_rewards[idx] = reward;
    }
}
