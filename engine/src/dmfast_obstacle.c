#include "dmfast_internal.h"

static int dmfast_obstacle_id(int seed) {
    /* Cast to unsigned: obstacle_seed comes from ex_rnd_u32() which is a u32
       RNG reinterpreted as int32_t — negative half the time. Signed % with a
       negative dividend produces a negative id, which collapses tier lookups
       to T5 and type lookups to MAGIC_OR_CLOTH. Same bug pattern as the beast
       id path — both had to be unsigned to match Cairo (which uses u32). */
    return (int)(((uint32_t)seed % (uint32_t)DMFAST_OBSTACLE_MAX_ID) + 1);
}

static int dmfast_obstacle_tier(int obstacle_id) {
    if ((1 <= obstacle_id && obstacle_id < 6) || (26 <= obstacle_id && obstacle_id < 31) || (51 <= obstacle_id && obstacle_id < 56)) {
        return DMFAST_TIER_T1;
    }
    if ((6 <= obstacle_id && obstacle_id < 11) || (31 <= obstacle_id && obstacle_id < 36) || (56 <= obstacle_id && obstacle_id < 61)) {
        return DMFAST_TIER_T2;
    }
    if ((11 <= obstacle_id && obstacle_id < 16) || (36 <= obstacle_id && obstacle_id < 41) || (61 <= obstacle_id && obstacle_id < 66)) {
        return DMFAST_TIER_T3;
    }
    if ((16 <= obstacle_id && obstacle_id < 21) || (41 <= obstacle_id && obstacle_id < 46) || (66 <= obstacle_id && obstacle_id < 71)) {
        return DMFAST_TIER_T4;
    }
    return DMFAST_TIER_T5;
}

static int dmfast_obstacle_type(int obstacle_id) {
    if (obstacle_id < 26) {
        return DMFAST_TYPE_MAGIC_OR_CLOTH;
    }
    if (obstacle_id < 51) {
        return DMFAST_TYPE_BLADE_OR_HIDE;
    }
    return DMFAST_TYPE_BLUDGEON_OR_METAL;
}

static int dmfast_obstacle_xp_reward(int obstacle_tier, int obstacle_level, int adventurer_level) {
    int level_decay_percentage = adventurer_level * 2;
    int reward_amount;
    int xp_reward;

    if (level_decay_percentage > 95) {
        level_decay_percentage = 95;
    }
    reward_amount = ((6 - obstacle_tier) * obstacle_level) / 2;
    xp_reward = (reward_amount * (100 - level_decay_percentage)) / 100;
    if (xp_reward < DMFAST_MIN_OBSTACLE_XP_REWARD) {
        return DMFAST_MIN_OBSTACLE_XP_REWARD;
    }
    return xp_reward;
}

static int dmfast_necklace_armor_bonus(int neck_id, int neck_greatness, int armor_type, int base_armor) {
    if (armor_type == DMFAST_TYPE_MAGIC_OR_CLOTH) {
        if (neck_id != 3) {
            return 0;
        }
    } else if (armor_type == DMFAST_TYPE_BLADE_OR_HIDE) {
        if (neck_id != 1) {
            return 0;
        }
    } else if (armor_type == DMFAST_TYPE_BLUDGEON_OR_METAL) {
        if (neck_id != 2) {
            return 0;
        }
    } else {
        return 0;
    }
    return (base_armor * DMFAST_NECKLACE_ARMOR_BONUS * neck_greatness) / 100;
}

void dmfast_process_obstacles(
    const int32_t *adventurer_xp,
    const int32_t *health,
    const int32_t *intelligence,
    const int32_t *base_vitality,
    const int32_t *base_damage_reduction,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *item_specials_seed,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *obstacle_seed,
    const int32_t *level_rnd,
    const int32_t *dmg_location_rnd,
    const int32_t *crit_hit_rnd,
    const int32_t *dodge_rnd,
    const int32_t *items_specials_rnd,
    int32_t count,
    int32_t *out_obstacle_id,
    uint8_t *out_dodged,
    int32_t *out_damage,
    int32_t *out_location,
    uint8_t *out_critical_hit,
    int32_t *out_xp_reward,
    int32_t *out_adventurer_xp,
    int32_t *out_health,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available,
    int32_t *out_item_specials_seed
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int current_level = dmfast_level_from_xp(adventurer_xp[i]);
        int obstacle_id = dmfast_obstacle_id(obstacle_seed[i]);
        int obstacle_tier = dmfast_obstacle_tier(obstacle_id);
        int obstacle_type = dmfast_obstacle_type(obstacle_id);
        int obstacle_level = dmfast_random_level(current_level, level_rnd[i]);
        int damage_slot = DMFAST_SLOT_CHEST + (dmg_location_rnd[i] % 5);
        int equipment_index = damage_slot - 1;
        int armor_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT + equipment_index];
        int armor_tier = armor_id == 0 ? DMFAST_TIER_NONE : dmfast_loot_tier(armor_id);
        int armor_type = armor_id == 0 ? DMFAST_TYPE_NONE : dmfast_loot_type(armor_id);
        int armor_level = dmfast_greatness_from_xp(
            equipment_xp[i * DMFAST_NUM_EQUIPMENT + equipment_index]
        );
        int neck_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT + 6];
        int neck_greatness = dmfast_greatness_from_xp(
            equipment_xp[i * DMFAST_NUM_EQUIPMENT + 6]
        );
        int32_t zero_specials[3] = {0, 0, 0};
        int32_t min_damage = DMFAST_MIN_DAMAGE_FROM_OBSTACLES;
        int32_t zero_value = 0;
        int32_t crit_chance = current_level > 100 ? 100 : current_level;
        int32_t damage_result[7];
        int32_t dodged_arr[1];
        uint8_t dodged_flag[1];
        int32_t reduced_damage_arr[1];
        int jewelry_bonus;
        int raw_damage;
        int reduced_damage;
        int reward_xp;
        int new_adv_xp;
        int upgrades_after_level;
        int new_level;

        dmfast_calculate_damage(
            &obstacle_tier,
            &obstacle_type,
            &obstacle_level,
            zero_specials,
            &armor_tier,
            &armor_type,
            &armor_level,
            zero_specials,
            &min_damage,
            &zero_value,
            &zero_value,
            &crit_chance,
            crit_hit_rnd + i,
            1,
            damage_result
        );
        jewelry_bonus = dmfast_necklace_armor_bonus(neck_id, neck_greatness, armor_type, damage_result[1]);
        raw_damage = damage_result[6];
        if (raw_damage > jewelry_bonus + DMFAST_MIN_DAMAGE_FROM_OBSTACLES) {
            raw_damage -= jewelry_bonus;
        } else {
            raw_damage = DMFAST_MIN_DAMAGE_FROM_OBSTACLES;
        }
        dmfast_apply_damage_reductions(
            &raw_damage,
            base_damage_reduction + i,
            1,
            reduced_damage_arr
        );
        reduced_damage = reduced_damage_arr[0];

        dodged_arr[0] = intelligence[i] + special_stats[i * DMFAST_NUM_STATS + DMFAST_STAT_INTELLIGENCE];
        dmfast_avoid_threat(&current_level, dodged_arr, dodge_rnd + i, 1, dodged_flag);

        out_obstacle_id[i] = obstacle_id;
        out_dodged[i] = dodged_flag[0];
        out_damage[i] = reduced_damage;
        out_location[i] = damage_slot;
        out_critical_hit[i] = damage_result[4] > 0 ? 1 : 0;
        out_health[i] = dodged_flag[0] || reduced_damage <= 0 ? health[i] : health[i] - reduced_damage;
        if (out_health[i] < 0) {
            out_health[i] = 0;
        }

        reward_xp = dmfast_obstacle_xp_reward(obstacle_tier, obstacle_level, current_level);
        out_xp_reward[i] = reward_xp;
        if (out_health[i] == 0) {
            out_adventurer_xp[i] = adventurer_xp[i];
            out_stat_upgrades_available[i] = stat_upgrades_available[i];
            out_item_specials_seed[i] = item_specials_seed[i];
            {
                int j;
                for (j = 0; j < DMFAST_NUM_EQUIPMENT; ++j) {
                    out_equipment_xp[i * DMFAST_NUM_EQUIPMENT + j] =
                        equipment_xp[i * DMFAST_NUM_EQUIPMENT + j];
                }
                for (j = 0; j < DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS; ++j) {
                    out_equipment_specials[
                        i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS + j
                    ] = equipment_specials[
                        i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS + j
                    ];
                }
                for (j = 0; j < DMFAST_NUM_STATS; ++j) {
                    out_special_stats[i * DMFAST_NUM_STATS + j] =
                        special_stats[i * DMFAST_NUM_STATS + j];
                }
            }
            continue;
        }

        new_adv_xp = adventurer_xp[i] + reward_xp;
        if (new_adv_xp > DMFAST_MAX_ADVENTURER_XP) {
            new_adv_xp = DMFAST_MAX_ADVENTURER_XP;
        }
        new_level = dmfast_level_from_xp(new_adv_xp);
        upgrades_after_level = stat_upgrades_available[i] + (new_level - current_level);
        out_adventurer_xp[i] = new_adv_xp;

        dmfast_grant_item_xp(
            equipment_ids + (i * DMFAST_NUM_EQUIPMENT),
            equipment_xp + (i * DMFAST_NUM_EQUIPMENT),
            equipment_specials + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
            out_health + i,
            base_vitality + i,
            special_stats + (i * DMFAST_NUM_STATS),
            &upgrades_after_level,
            item_specials_seed + i,
            &reward_xp,
            items_specials_rnd + i,
            1,
            out_equipment_xp + (i * DMFAST_NUM_EQUIPMENT),
            out_equipment_specials + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
            out_health + i,
            out_special_stats + (i * DMFAST_NUM_STATS),
            out_stat_upgrades_available + i,
            out_item_specials_seed + i
        );
    }
}
