#include "dmfast_internal.h"

static void dmfast_clear_beast_outputs(
    int index,
    int32_t *out_beast_id,
    int32_t *out_beast_health,
    int32_t *out_beast_tier,
    int32_t *out_beast_type,
    int32_t *out_beast_level,
    int32_t *out_beast_specials,
    uint8_t *out_beast_is_ambush,
    int32_t *out_beast_attack_damage,
    int32_t *out_beast_attack_location,
    uint8_t *out_beast_attack_critical_hit
) {
    out_beast_id[index] = 0;
    out_beast_health[index] = 0;
    out_beast_tier[index] = 0;
    out_beast_type[index] = 0;
    out_beast_level[index] = 0;
    out_beast_specials[index * 3] = 0;
    out_beast_specials[index * 3 + 1] = 0;
    out_beast_specials[index * 3 + 2] = 0;
    out_beast_is_ambush[index] = 0;
    out_beast_attack_damage[index] = 0;
    out_beast_attack_location[index] = 0;
    out_beast_attack_critical_hit[index] = 0;
}

static void dmfast_clear_obstacle_outputs(
    int index,
    int32_t *out_obstacle_id,
    uint8_t *out_obstacle_dodged,
    int32_t *out_obstacle_damage,
    int32_t *out_obstacle_location,
    uint8_t *out_obstacle_critical_hit,
    int32_t *out_obstacle_xp_reward
) {
    out_obstacle_id[index] = 0;
    out_obstacle_dodged[index] = 0;
    out_obstacle_damage[index] = 0;
    out_obstacle_location[index] = 0;
    out_obstacle_critical_hit[index] = 0;
    out_obstacle_xp_reward[index] = 0;
}

static void dmfast_clear_discovery_outputs(
    int index,
    int32_t *out_discovery_type,
    int32_t *out_discovery_loot_item_id
) {
    out_discovery_type[index] = -1;
    out_discovery_loot_item_id[index] = 0;
}

void dmfast_process_explores(
    const int32_t *action_count,
    const int32_t *adventurer_xp,
    const int32_t *max_health,
    const int32_t *health,
    const int32_t *gold,
    const int32_t *intelligence,
    const int32_t *wisdom,
    const int32_t *base_vitality,
    const int32_t *base_damage_reduction,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *item_specials_seed,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *bag_ids,
    const int32_t *explore_result,
    const int32_t *beast_seed,
    const int32_t *beast_health_rnd,
    const int32_t *beast_level_rnd,
    const int32_t *beast_dmg_location_rnd,
    const int32_t *beast_crit_hit_rnd,
    const int32_t *beast_ambush_rnd,
    const int32_t *beast_special2_rnd,
    const int32_t *beast_special3_rnd,
    const int32_t *obstacle_seed,
    const int32_t *obstacle_level_rnd,
    const int32_t *obstacle_dmg_location_rnd,
    const int32_t *obstacle_crit_hit_rnd,
    const int32_t *obstacle_dodge_rnd,
    const int32_t *obstacle_item_specials_rnd,
    const int32_t *discovery_type_rnd,
    const int32_t *discovery_amount_rnd1,
    const int32_t *discovery_amount_rnd2,
    int32_t count,
    int32_t *out_action_count,
    int32_t *out_phase,
    int32_t *out_adventurer_xp,
    int32_t *out_health,
    int32_t *out_gold,
    int32_t *out_stat_upgrades_available,
    int32_t *out_item_specials_seed,
    int32_t *out_equipment_ids,
    int32_t *out_bag_ids,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_special_stats,
    int32_t *out_beast_id,
    int32_t *out_beast_health,
    int32_t *out_beast_tier,
    int32_t *out_beast_type,
    int32_t *out_beast_level,
    int32_t *out_beast_specials,
    uint8_t *out_beast_is_ambush,
    int32_t *out_beast_attack_damage,
    int32_t *out_beast_attack_location,
    uint8_t *out_beast_attack_critical_hit,
    int32_t *out_obstacle_id,
    uint8_t *out_obstacle_dodged,
    int32_t *out_obstacle_damage,
    int32_t *out_obstacle_location,
    uint8_t *out_obstacle_critical_hit,
    int32_t *out_obstacle_xp_reward,
    int32_t *out_discovery_type,
    int32_t *out_discovery_loot_item_id
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int eq_offset = i * DMFAST_NUM_EQUIPMENT;
        int eq_special_offset = i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS;
        int stat_offset = i * DMFAST_NUM_STATS;
        int bag_offset = i * DMFAST_NUM_BAG;

        out_action_count[i] = action_count[i] + 1;
        out_phase[i] = DMFAST_PHASE_MARKET;
        out_adventurer_xp[i] = adventurer_xp[i];
        out_health[i] = health[i];
        out_gold[i] = gold[i];
        out_stat_upgrades_available[i] = stat_upgrades_available[i];
        out_item_specials_seed[i] = item_specials_seed[i];
        dmfast_copy_ints(
            equipment_ids + eq_offset,
            DMFAST_NUM_EQUIPMENT,
            out_equipment_ids + eq_offset
        );
        dmfast_copy_ints(
            bag_ids + bag_offset,
            DMFAST_NUM_BAG,
            out_bag_ids + bag_offset
        );
        dmfast_copy_ints(
            equipment_xp + eq_offset,
            DMFAST_NUM_EQUIPMENT,
            out_equipment_xp + eq_offset
        );
        dmfast_copy_ints(
            equipment_specials + eq_special_offset,
            DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS,
            out_equipment_specials + eq_special_offset
        );
        dmfast_copy_ints(
            special_stats + stat_offset,
            DMFAST_NUM_STATS,
            out_special_stats + stat_offset
        );
        dmfast_clear_beast_outputs(
            i,
            out_beast_id,
            out_beast_health,
            out_beast_tier,
            out_beast_type,
            out_beast_level,
            out_beast_specials,
            out_beast_is_ambush,
            out_beast_attack_damage,
            out_beast_attack_location,
            out_beast_attack_critical_hit
        );
        dmfast_clear_obstacle_outputs(
            i,
            out_obstacle_id,
            out_obstacle_dodged,
            out_obstacle_damage,
            out_obstacle_location,
            out_obstacle_critical_hit,
            out_obstacle_xp_reward
        );
        dmfast_clear_discovery_outputs(i, out_discovery_type, out_discovery_loot_item_id);

        if (explore_result[i] == DMFAST_EXPLORE_BEAST) {
            out_phase[i] = DMFAST_PHASE_COMBAT;
            dmfast_process_beast_encounters(
                adventurer_xp + i,
                health + i,
                wisdom + i,
                base_damage_reduction + i,
                special_stats + stat_offset,
                equipment_ids + eq_offset,
                equipment_xp + eq_offset,
                equipment_specials + eq_special_offset,
                beast_seed + i,
                beast_health_rnd + i,
                beast_level_rnd + i,
                beast_dmg_location_rnd + i,
                beast_crit_hit_rnd + i,
                beast_ambush_rnd + i,
                beast_special2_rnd + i,
                beast_special3_rnd + i,
                1,
                out_beast_id + i,
                out_beast_health + i,
                out_beast_tier + i,
                out_beast_type + i,
                out_beast_level + i,
                out_beast_specials + (i * 3),
                out_beast_is_ambush + i,
                out_beast_attack_damage + i,
                out_beast_attack_location + i,
                out_beast_attack_critical_hit + i,
                out_health + i
            );
            continue;
        }

        if (explore_result[i] == DMFAST_EXPLORE_OBSTACLE) {
            dmfast_process_obstacles(
                adventurer_xp + i,
                health + i,
                intelligence + i,
                base_vitality + i,
                base_damage_reduction + i,
                special_stats + stat_offset,
                stat_upgrades_available + i,
                item_specials_seed + i,
                equipment_ids + eq_offset,
                equipment_xp + eq_offset,
                equipment_specials + eq_special_offset,
                obstacle_seed + i,
                obstacle_level_rnd + i,
                obstacle_dmg_location_rnd + i,
                obstacle_crit_hit_rnd + i,
                obstacle_dodge_rnd + i,
                obstacle_item_specials_rnd + i,
                1,
                out_obstacle_id + i,
                out_obstacle_dodged + i,
                out_obstacle_damage + i,
                out_obstacle_location + i,
                out_obstacle_critical_hit + i,
                out_obstacle_xp_reward + i,
                out_adventurer_xp + i,
                out_health + i,
                out_equipment_xp + eq_offset,
                out_equipment_specials + eq_special_offset,
                out_special_stats + stat_offset,
                out_stat_upgrades_available + i,
                out_item_specials_seed + i
            );
            continue;
        }

        if (explore_result[i] == DMFAST_EXPLORE_DISCOVERY) {
            dmfast_process_discoveries(
                discovery_type_rnd + i,
                discovery_amount_rnd1 + i,
                discovery_amount_rnd2 + i,
                max_health + i,
                health + i,
                gold + i,
                adventurer_xp + i,
                stat_upgrades_available + i,
                equipment_ids + eq_offset,
                bag_ids + bag_offset,
                1,
                out_discovery_type + i,
                out_discovery_loot_item_id + i,
                out_health + i,
                out_gold + i,
                out_adventurer_xp + i,
                out_stat_upgrades_available + i,
                out_equipment_ids + eq_offset,
                out_bag_ids + bag_offset
            );
        }
    }
}
