#include "dmfast_internal.h"

static int dmfast_beast_tier(int beast_id) {
    if ((1 <= beast_id && beast_id <= 5) || (26 <= beast_id && beast_id < 31) || (51 <= beast_id && beast_id < 56)) {
        return 1;
    }
    if ((6 <= beast_id && beast_id < 11) || (31 <= beast_id && beast_id < 36) || (56 <= beast_id && beast_id < 61)) {
        return 2;
    }
    if ((11 <= beast_id && beast_id < 16) || (36 <= beast_id && beast_id < 41) || (61 <= beast_id && beast_id < 66)) {
        return 3;
    }
    if ((16 <= beast_id && beast_id < 21) || (41 <= beast_id && beast_id < 46) || (66 <= beast_id && beast_id < 71)) {
        return 4;
    }
    return 5;
}

static int dmfast_beast_type(int beast_id) {
    if (beast_id < 26) {
        return DMFAST_TYPE_MAGIC_OR_CLOTH;
    }
    if (beast_id < 51) {
        return DMFAST_TYPE_BLADE_OR_HIDE;
    }
    return DMFAST_TYPE_BLUDGEON_OR_METAL;
}

static int dmfast_starter_beast_id(int weapon_type) {
    if (weapon_type == DMFAST_TYPE_MAGIC_OR_CLOTH) {
        return 71;
    }
    if (weapon_type == DMFAST_TYPE_BLADE_OR_HIDE) {
        return 21;
    }
    if (weapon_type == DMFAST_TYPE_BLUDGEON_OR_METAL) {
        return 46;
    }
    return 0;
}

void dmfast_process_beast_encounters(
    const int32_t *adventurer_xp,
    const int32_t *health,
    const int32_t *wisdom,
    const int32_t *base_damage_reduction,
    const int32_t *special_stats,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *beast_seed,
    const int32_t *health_rnd,
    const int32_t *level_rnd,
    const int32_t *dmg_location_rnd,
    const int32_t *crit_hit_rnd,
    const int32_t *ambush_rnd,
    const int32_t *special2_rnd,
    const int32_t *special3_rnd,
    int32_t count,
    int32_t *out_beast_id,
    int32_t *out_beast_health,
    int32_t *out_beast_tier,
    int32_t *out_beast_type,
    int32_t *out_beast_level,
    int32_t *out_beast_specials,
    uint8_t *out_is_ambush,
    int32_t *out_attack_damage,
    int32_t *out_attack_location,
    uint8_t *out_attack_critical_hit,
    int32_t *out_health
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int current_level = dmfast_level_from_xp(adventurer_xp[i]);
        int weapon_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT];
        int weapon_type = dmfast_loot_type(weapon_id);
        int beast_id;
        int beast_health;
        int beast_tier;
        int beast_type;
        int beast_level;
        int beast_specials[3] = {0, 0, 0};
        int wisdom_total = wisdom[i] + special_stats[i * DMFAST_NUM_STATS + DMFAST_STAT_WISDOM];
        int is_ambush;

        if (current_level == 1) {
            beast_id = dmfast_starter_beast_id(weapon_type);
            beast_health = DMFAST_BEAST_STARTER_HEALTH;
            beast_tier = dmfast_beast_tier(beast_id);
            beast_type = dmfast_beast_type(beast_id);
            beast_level = 1;
        } else {
            /* Cast to unsigned: beast_seed comes from a u32 RNG reinterpreted as
               int32, so half the time it's negative. With signed %, -X % 75
               yields a negative result → beast_id < 0 → all downstream tier/
               type lookups fall through to tier=5 / type=magic, skewing the
               training distribution. Cairo uses u32 so never hits this.  */
            beast_id = ((uint32_t)beast_seed[i] % (uint32_t)DMFAST_BEAST_MAX_ID) + 1;
            beast_health = dmfast_random_beast_health(current_level, health_rnd[i]);
            beast_tier = dmfast_beast_tier(beast_id);
            beast_type = dmfast_beast_type(beast_id);
            beast_level = dmfast_random_level(current_level, level_rnd[i]);
            if (beast_level >= DMFAST_BEAST_SPECIAL_UNLOCK) {
                beast_specials[0] = 0;
                beast_specials[1] = 1 + (special2_rnd[i] % DMFAST_BEAST_MAX_SPECIAL2);
                beast_specials[2] = 1 + (special3_rnd[i] % DMFAST_BEAST_MAX_SPECIAL3);
            }
        }

        out_beast_id[i] = beast_id;
        out_beast_health[i] = beast_health;
        out_beast_tier[i] = beast_tier;
        out_beast_type[i] = beast_type;
        out_beast_level[i] = beast_level;
        out_beast_specials[i * 3] = beast_specials[0];
        out_beast_specials[i * 3 + 1] = beast_specials[1];
        out_beast_specials[i * 3 + 2] = beast_specials[2];

        is_ambush = dmfast_avoid_threat_one(current_level, wisdom_total, ambush_rnd[i]) ? 0 : 1;
        out_is_ambush[i] = (uint8_t)is_ambush;

        if (!is_ambush) {
            out_attack_damage[i] = 0;
            out_attack_location[i] = 0;
            out_attack_critical_hit[i] = 0;
            out_health[i] = health[i];
            continue;
        }

        {
            int damage_slot = DMFAST_SLOT_CHEST + (dmg_location_rnd[i] % 5);
            int equipment_index = damage_slot - 1;
            int armor_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT + equipment_index];
            int armor_tier = armor_id == 0 ? 0 : dmfast_loot_tier(armor_id);
            int armor_type = armor_id == 0 ? DMFAST_TYPE_NONE : dmfast_loot_type(armor_id);
            int armor_level = dmfast_greatness_from_xp(
                equipment_xp[i * DMFAST_NUM_EQUIPMENT + equipment_index]
            );
            int neck_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT + 6];
            int neck_greatness = dmfast_greatness_from_xp(
                equipment_xp[i * DMFAST_NUM_EQUIPMENT + 6]
            );
            int crit_chance = current_level > 100 ? 100 : current_level;
            int32_t defend_result[9];
            int32_t reduced_damage_arr[1];
            int raw_damage;

            dmfast_adventurer_defend_totals(
                &beast_tier,
                &beast_type,
                &beast_level,
                beast_specials,
                &armor_tier,
                &armor_type,
                &armor_level,
                equipment_specials
                    + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS)
                    + (equipment_index * DMFAST_NUM_SPECIALS),
                &neck_id,
                &neck_greatness,
                &crit_chance,
                crit_hit_rnd + i,
                1,
                defend_result
            );
            raw_damage = defend_result[8];
            dmfast_apply_damage_reductions(&raw_damage, base_damage_reduction + i, 1, reduced_damage_arr);

            out_attack_damage[i] = reduced_damage_arr[0];
            out_attack_location[i] = damage_slot;
            out_attack_critical_hit[i] = defend_result[4] > 0 ? 1 : 0;
            out_health[i] = health[i] - reduced_damage_arr[0];
            if (out_health[i] < 0) {
                out_health[i] = 0;
            }
        }
    }
}
