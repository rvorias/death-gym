#include "dmfast_internal.h"

static void dmfast_copy_state_unchanged(
    int index,
    const int32_t *health,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *item_specials_seed,
    int32_t *out_health,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available,
    int32_t *out_item_specials_seed
) {
    out_health[index] = health[index];
    out_stat_upgrades_available[index] = stat_upgrades_available[index];
    out_item_specials_seed[index] = item_specials_seed[index];
    dmfast_copy_ints(
        equipment_xp + (index * DMFAST_NUM_EQUIPMENT),
        DMFAST_NUM_EQUIPMENT,
        out_equipment_xp + (index * DMFAST_NUM_EQUIPMENT)
    );
    dmfast_copy_ints(
        equipment_specials + (index * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
        DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS,
        out_equipment_specials + (index * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS)
    );
    dmfast_copy_ints(
        special_stats + (index * DMFAST_NUM_STATS),
        DMFAST_NUM_STATS,
        out_special_stats + (index * DMFAST_NUM_STATS)
    );
}

void dmfast_beast_retaliation(
    int index,
    const int32_t *health,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *beast_tier,
    const int32_t *beast_type,
    const int32_t *beast_level,
    const int32_t *beast_specials,
    const int32_t *beast_crit_hit_rnd,
    const int32_t *attack_location_rnd,
    int current_level,
    int32_t *out_health,
    int32_t *out_beast_attack_damage,
    int32_t *out_beast_attack_location,
    uint8_t *out_beast_attack_critical_hit
) {
    int damage_slot = DMFAST_SLOT_CHEST + (attack_location_rnd[index] % 5);
    int equipment_index = damage_slot - 1;
    int armor_id = equipment_ids[index * DMFAST_NUM_EQUIPMENT + equipment_index];
    int armor_tier = armor_id == 0 ? 0 : dmfast_loot_tier(armor_id);
    int armor_type = armor_id == 0 ? DMFAST_TYPE_NONE : dmfast_loot_type(armor_id);
    int armor_level = dmfast_greatness_from_xp(
        equipment_xp[index * DMFAST_NUM_EQUIPMENT + equipment_index]
    );
    int neck_id = equipment_ids[index * DMFAST_NUM_EQUIPMENT + 6];
    int neck_greatness = dmfast_greatness_from_xp(
        equipment_xp[index * DMFAST_NUM_EQUIPMENT + 6]
    );
    int32_t crit_chance = current_level > 100 ? 100 : current_level;
    int32_t defend_result[9];
    int raw_damage;

    dmfast_adventurer_defend_totals(
        beast_tier + index,
        beast_type + index,
        beast_level + index,
        beast_specials + (index * 3),
        &armor_tier,
        &armor_type,
        &armor_level,
        equipment_specials
            + (index * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS)
            + (equipment_index * DMFAST_NUM_SPECIALS),
        &neck_id,
        &neck_greatness,
        &crit_chance,
        beast_crit_hit_rnd + index,
        1,
        defend_result
    );
    raw_damage = defend_result[8];
    out_beast_attack_damage[index] = raw_damage;
    out_beast_attack_location[index] = damage_slot;
    out_beast_attack_critical_hit[index] = defend_result[4] > 0 ? 1 : 0;
    out_health[index] = health[index] - raw_damage;
    if (out_health[index] < 0) {
        out_health[index] = 0;
    }
}

void dmfast_process_attacks(
    const int32_t *adventurer_xp,
    const int32_t *gold,
    const int32_t *health,
    const int32_t *strength,
    const int32_t *luck,
    const int32_t *base_vitality,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *item_specials_seed,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *beast_health,
    const int32_t *beast_tier,
    const int32_t *beast_type,
    const int32_t *beast_level,
    const int32_t *beast_specials,
    const int32_t *item_specials_rnd,
    const int32_t *adventurer_crit_hit_rnd,
    const int32_t *beast_crit_hit_rnd,
    const int32_t *attack_location_rnd,
    int32_t count,
    int32_t *out_attack_damage,
    int32_t *out_attack_location,
    uint8_t *out_attack_critical_hit,
    uint8_t *out_beast_dead,
    int32_t *out_beast_level_if_dead,
    int32_t *out_gold_reward,
    int32_t *out_xp_reward,
    int32_t *out_adventurer_xp,
    int32_t *out_gold,
    int32_t *out_health,
    int32_t *out_beast_health,
    int32_t *out_beast_attack_damage,
    int32_t *out_beast_attack_location,
    uint8_t *out_beast_attack_critical_hit,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available,
    int32_t *out_item_specials_seed
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int current_level = dmfast_level_from_xp(adventurer_xp[i]);
        int weapon_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT];
        int weapon_tier = weapon_id == 0 ? 0 : dmfast_loot_tier(weapon_id);
        int weapon_type = weapon_id == 0 ? DMFAST_TYPE_NONE : dmfast_loot_type(weapon_id);
        int weapon_greatness = dmfast_greatness_from_xp(
            equipment_xp[i * DMFAST_NUM_EQUIPMENT]
        );
        int ring_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT + 7];
        int ring_greatness = dmfast_greatness_from_xp(
            equipment_xp[i * DMFAST_NUM_EQUIPMENT + 7]
        );
        int total_strength = strength[i] + special_stats[i * DMFAST_NUM_STATS + DMFAST_STAT_STRENGTH];
        int total_luck = luck[i] + special_stats[i * DMFAST_NUM_STATS + DMFAST_STAT_LUCK];
        int32_t attack_result[10];

        dmfast_adventurer_attack_totals(
            &weapon_tier,
            &weapon_type,
            &weapon_greatness,
            equipment_specials + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
            &ring_id,
            &ring_greatness,
            beast_tier + i,
            beast_type + i,
            beast_level + i,
            beast_specials + (i * 3),
            &total_strength,
            &total_luck,
            adventurer_crit_hit_rnd + i,
            1,
            attack_result
        );

        out_attack_damage[i] = attack_result[9];
        out_attack_location[i] = attack_location_rnd[i];
        out_attack_critical_hit[i] = attack_result[4] > 0 ? 1 : 0;
        out_beast_level_if_dead[i] = 0;
        out_gold_reward[i] = 0;
        out_xp_reward[i] = 0;
        out_beast_attack_damage[i] = 0;
        out_beast_attack_location[i] = 0;
        out_beast_attack_critical_hit[i] = 0;

        if (out_attack_damage[i] >= beast_health[i]) {
            out_beast_dead[i] = 1;
            out_beast_level_if_dead[i] = beast_level[i];
            out_beast_health[i] = 0;
            dmfast_process_beast_deaths(
                adventurer_xp + i,
                gold + i,
                health + i,
                base_vitality + i,
                special_stats + (i * DMFAST_NUM_STATS),
                stat_upgrades_available + i,
                item_specials_seed + i,
                equipment_ids + (i * DMFAST_NUM_EQUIPMENT),
                equipment_xp + (i * DMFAST_NUM_EQUIPMENT),
                equipment_specials + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
                beast_tier + i,
                beast_level + i,
                item_specials_rnd + i,
                1,
                out_gold_reward + i,
                out_xp_reward + i,
                out_adventurer_xp + i,
                out_gold + i,
                out_health + i,
                out_equipment_xp + (i * DMFAST_NUM_EQUIPMENT),
                out_equipment_specials + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
                out_special_stats + (i * DMFAST_NUM_STATS),
                out_stat_upgrades_available + i,
                out_item_specials_seed + i
            );
        } else {
            out_beast_dead[i] = 0;
            out_beast_health[i] = beast_health[i] - out_attack_damage[i];
            out_adventurer_xp[i] = adventurer_xp[i];
            out_gold[i] = gold[i];
            dmfast_copy_state_unchanged(
                i,
                health,
                equipment_xp,
                equipment_specials,
                special_stats,
                stat_upgrades_available,
                item_specials_seed,
                out_health,
                out_equipment_xp,
                out_equipment_specials,
                out_special_stats,
                out_stat_upgrades_available,
                out_item_specials_seed
            );
            dmfast_beast_retaliation(
                i,
                health,
                equipment_ids,
                equipment_xp,
                equipment_specials,
                beast_tier,
                beast_type,
                beast_level,
                beast_specials,
                beast_crit_hit_rnd,
                attack_location_rnd,
                current_level,
                out_health,
                out_beast_attack_damage,
                out_beast_attack_location,
                out_beast_attack_critical_hit
            );
        }
    }
}

void dmfast_process_flees(
    const int32_t *adventurer_xp,
    const int32_t *health,
    const int32_t *dexterity,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *beast_health,
    const int32_t *beast_tier,
    const int32_t *beast_type,
    const int32_t *beast_level,
    const int32_t *beast_specials,
    const int32_t *flee_rnd,
    const int32_t *beast_crit_hit_rnd,
    const int32_t *attack_location_rnd,
    int32_t count,
    uint8_t *out_fled,
    int32_t *out_adventurer_xp,
    int32_t *out_health,
    int32_t *out_beast_health,
    int32_t *out_stat_upgrades_available,
    int32_t *out_beast_attack_damage,
    int32_t *out_beast_attack_location,
    uint8_t *out_beast_attack_critical_hit
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int current_level = dmfast_level_from_xp(adventurer_xp[i]);
        int total_dexterity = dexterity[i] + special_stats[i * DMFAST_NUM_STATS + DMFAST_STAT_DEXTERITY];
        int avoided;

        avoided = dmfast_attempt_flee_one(current_level, total_dexterity, flee_rnd[i]);
        out_fled[i] = avoided > 0 ? 1 : 0;
        out_beast_attack_damage[i] = 0;
        out_beast_attack_location[i] = 0;
        out_beast_attack_critical_hit[i] = 0;

        if (avoided) {
            int new_adv_xp = adventurer_xp[i] + 1;
            int new_level;

            if (new_adv_xp > DMFAST_MAX_ADVENTURER_XP) {
                new_adv_xp = DMFAST_MAX_ADVENTURER_XP;
            }
            new_level = dmfast_level_from_xp(new_adv_xp);
            out_adventurer_xp[i] = new_adv_xp;
            out_stat_upgrades_available[i] = stat_upgrades_available[i] + (new_level - current_level);
            out_health[i] = health[i];
            out_beast_health[i] = 0;
        } else {
            out_adventurer_xp[i] = adventurer_xp[i];
            out_stat_upgrades_available[i] = stat_upgrades_available[i];
            out_beast_health[i] = beast_health[i];
            dmfast_beast_retaliation(
                i,
                health,
                equipment_ids,
                equipment_xp,
                equipment_specials,
                beast_tier,
                beast_type,
                beast_level,
                beast_specials,
                beast_crit_hit_rnd,
                attack_location_rnd,
                current_level,
                out_health,
                out_beast_attack_damage,
                out_beast_attack_location,
                out_beast_attack_critical_hit
            );
        }
    }
}
