#include "dmfast_internal.h"

static int dmfast_inventory_is_jewelry(int item_id) {
    int slot = dmfast_loot_slot(item_id);
    return slot == 7 || slot == 8;
}

static void dmfast_inventory_recalculate_special_stats(
    int32_t health,
    int32_t base_vitality,
    const int32_t *previous_special_stats,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *bag_ids,
    const int32_t *bag_xp,
    const int32_t *bag_specials,
    int32_t *out_health,
    int32_t *out_special_stats
) {
    int32_t stats[DMFAST_NUM_STATS];
    int i;
    int neck_greatness = 0;
    int ring_greatness = 0;
    int bag_jewelry_greatness = 0;
    int health_after_delta;
    int max_health;

    memset(stats, 0, sizeof(stats));

    for (i = 0; i < DMFAST_NUM_EQUIPMENT; ++i) {
        int item_id = equipment_ids[i];
        int greatness = dmfast_greatness_from_xp(equipment_xp[i]);

        if (item_id == 0) {
            continue;
        }
        if (greatness >= DMFAST_SUFFIX_UNLOCK) {
            int suffix = equipment_specials[i * DMFAST_NUM_SPECIALS];
            dmfast_apply_suffix_boost(stats, suffix);
            dmfast_apply_bag_boost(stats, suffix);
        }
        if (i == 6) {
            neck_greatness = greatness;
        } else if (i == 7) {
            ring_greatness = greatness;
        }
    }

    for (i = 0; i < DMFAST_NUM_BAG; ++i) {
        int item_id = bag_ids[i];
        int greatness = dmfast_greatness_from_xp(bag_xp[i]);

        if (item_id == 0) {
            continue;
        }
        if (greatness >= DMFAST_SUFFIX_UNLOCK) {
            int suffix = bag_specials[i * DMFAST_NUM_SPECIALS];
            dmfast_apply_bag_boost(stats, suffix);
        }
        if (dmfast_inventory_is_jewelry(item_id)) {
            bag_jewelry_greatness += greatness;
        }
    }

    stats[DMFAST_STAT_LUCK] = neck_greatness + ring_greatness + bag_jewelry_greatness;
    if (equipment_ids[7] == DMFAST_SILVER_RING_ID) {
        stats[DMFAST_STAT_LUCK] += ring_greatness;
    }

    health_after_delta = health
        + (stats[DMFAST_STAT_VITALITY]
            - previous_special_stats[DMFAST_STAT_VITALITY])
            * DMFAST_HEALTH_PER_VITALITY;
    if (health_after_delta < 0) {
        health_after_delta = 0;
    }
    max_health = dmfast_max_health(
        base_vitality,
        stats[DMFAST_STAT_VITALITY]
    );
    if (health_after_delta > max_health) {
        health_after_delta = max_health;
    }

    dmfast_copy_ints(
        stats,
        DMFAST_NUM_STATS,
        out_special_stats
    );
    *out_health = health_after_delta;
}

static int dmfast_inventory_final_phase(int beast_present, int stat_upgrades_available) {
    if (!beast_present && stat_upgrades_available > 0) {
        return DMFAST_PHASE_UPGRADE;
    }
    return beast_present ? DMFAST_PHASE_COMBAT : DMFAST_PHASE_MARKET;
}

static void dmfast_inventory_append_switch_slot(int32_t *switch_buffer_slots, int slot) {
    int i;

    for (i = 0; i < DMFAST_NUM_EQUIPMENT; ++i) {
        if (switch_buffer_slots[i] == slot) {
            return;
        }
    }
    for (i = 0; i < DMFAST_NUM_EQUIPMENT; ++i) {
        if (switch_buffer_slots[i] == 0) {
            switch_buffer_slots[i] = slot;
            return;
        }
    }
}

void dmfast_process_equips(
    const int32_t *health,
    const int32_t *base_vitality,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *bag_ids,
    const int32_t *bag_xp,
    const int32_t *bag_specials,
    const int32_t *switch_buffer_slots,
    const uint8_t *beast_present,
    const int32_t *bag_index,
    int32_t count,
    uint8_t *out_valid,
    int32_t *out_phase,
    int32_t *out_health,
    int32_t *out_special_stats,
    int32_t *out_equipment_ids,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_bag_ids,
    int32_t *out_bag_xp,
    int32_t *out_bag_specials,
    int32_t *out_switch_buffer_slots
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int32_t *eq_ids_out = out_equipment_ids + ((size_t)i * DMFAST_NUM_EQUIPMENT);
        int32_t *eq_xp_out = out_equipment_xp + ((size_t)i * DMFAST_NUM_EQUIPMENT);
        int32_t *eq_specials_out = out_equipment_specials
            + ((size_t)i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS);
        int32_t *bag_ids_out = out_bag_ids + ((size_t)i * DMFAST_NUM_BAG);
        int32_t *bag_xp_out = out_bag_xp + ((size_t)i * DMFAST_NUM_BAG);
        int32_t *bag_specials_out = out_bag_specials
            + ((size_t)i * DMFAST_NUM_BAG * DMFAST_NUM_SPECIALS);
        int32_t *switch_out = out_switch_buffer_slots + ((size_t)i * DMFAST_NUM_EQUIPMENT);
        int32_t *stats_out = out_special_stats + ((size_t)i * DMFAST_NUM_STATS);
        int input_index = bag_index[i];
        int valid = 0;

        dmfast_copy_ints(
            equipment_ids + ((size_t)i * DMFAST_NUM_EQUIPMENT),
            DMFAST_NUM_EQUIPMENT,
            eq_ids_out
        );
        dmfast_copy_ints(
            equipment_xp + ((size_t)i * DMFAST_NUM_EQUIPMENT),
            DMFAST_NUM_EQUIPMENT,
            eq_xp_out
        );
        dmfast_copy_ints(
            equipment_specials + ((size_t)i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
            DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS,
            eq_specials_out
        );
        dmfast_copy_ints(
            bag_ids + ((size_t)i * DMFAST_NUM_BAG),
            DMFAST_NUM_BAG,
            bag_ids_out
        );
        dmfast_copy_ints(
            bag_xp + ((size_t)i * DMFAST_NUM_BAG),
            DMFAST_NUM_BAG,
            bag_xp_out
        );
        dmfast_copy_ints(
            bag_specials + ((size_t)i * DMFAST_NUM_BAG * DMFAST_NUM_SPECIALS),
            DMFAST_NUM_BAG * DMFAST_NUM_SPECIALS,
            bag_specials_out
        );
        dmfast_copy_ints(
            switch_buffer_slots + ((size_t)i * DMFAST_NUM_EQUIPMENT),
            DMFAST_NUM_EQUIPMENT,
            switch_out
        );

        if (input_index >= 0 && input_index < DMFAST_NUM_BAG) {
            int item_id = bag_ids_out[input_index];
            int slot = dmfast_loot_slot(item_id);
            int equipment_index = slot - 1;
            int j;
            int slot_in_buffer = 0;

            for (j = 0; j < DMFAST_NUM_EQUIPMENT; ++j) {
                if (switch_out[j] == slot) {
                    slot_in_buffer = 1;
                    break;
                }
            }

            if (
                item_id != 0
                && slot != DMFAST_SLOT_NONE
                && !slot_in_buffer
                && equipment_index >= 0
                && equipment_index < DMFAST_NUM_EQUIPMENT
            ) {
                int old_id = eq_ids_out[equipment_index];
                int old_xp = eq_xp_out[equipment_index];
                int old_s1 = eq_specials_out[equipment_index * DMFAST_NUM_SPECIALS];
                int old_s2 = eq_specials_out[equipment_index * DMFAST_NUM_SPECIALS + 1];
                int old_s3 = eq_specials_out[equipment_index * DMFAST_NUM_SPECIALS + 2];

                eq_ids_out[equipment_index] = bag_ids_out[input_index];
                eq_xp_out[equipment_index] = bag_xp_out[input_index];
                eq_specials_out[equipment_index * DMFAST_NUM_SPECIALS]
                    = bag_specials_out[input_index * DMFAST_NUM_SPECIALS];
                eq_specials_out[equipment_index * DMFAST_NUM_SPECIALS + 1]
                    = bag_specials_out[input_index * DMFAST_NUM_SPECIALS + 1];
                eq_specials_out[equipment_index * DMFAST_NUM_SPECIALS + 2]
                    = bag_specials_out[input_index * DMFAST_NUM_SPECIALS + 2];

                bag_ids_out[input_index] = old_id;
                bag_xp_out[input_index] = old_xp;
                bag_specials_out[input_index * DMFAST_NUM_SPECIALS] = old_s1;
                bag_specials_out[input_index * DMFAST_NUM_SPECIALS + 1] = old_s2;
                bag_specials_out[input_index * DMFAST_NUM_SPECIALS + 2] = old_s3;

                dmfast_inventory_append_switch_slot(switch_out, slot);
                valid = 1;
            }
        }

        dmfast_inventory_recalculate_special_stats(
            health[i],
            base_vitality[i],
            special_stats + ((size_t)i * DMFAST_NUM_STATS),
            eq_ids_out,
            eq_xp_out,
            eq_specials_out,
            bag_ids_out,
            bag_xp_out,
            bag_specials_out,
            out_health + i,
            stats_out
        );
        out_valid[i] = (uint8_t)valid;
        out_phase[i] = dmfast_inventory_final_phase(
            beast_present[i] != 0,
            stat_upgrades_available[i]
        );
    }
}

void dmfast_process_drops(
    const int32_t *health,
    const int32_t *base_vitality,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *bag_ids,
    const int32_t *bag_xp,
    const int32_t *bag_specials,
    const uint8_t *beast_present,
    const int32_t *bag_index,
    int32_t count,
    uint8_t *out_valid,
    int32_t *out_phase,
    int32_t *out_health,
    int32_t *out_special_stats,
    int32_t *out_bag_ids,
    int32_t *out_bag_xp,
    int32_t *out_bag_specials
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int32_t *bag_ids_out = out_bag_ids + ((size_t)i * DMFAST_NUM_BAG);
        int32_t *bag_xp_out = out_bag_xp + ((size_t)i * DMFAST_NUM_BAG);
        int32_t *bag_specials_out = out_bag_specials
            + ((size_t)i * DMFAST_NUM_BAG * DMFAST_NUM_SPECIALS);
        int32_t *stats_out = out_special_stats + ((size_t)i * DMFAST_NUM_STATS);
        int input_index = bag_index[i];
        int valid = 0;

        dmfast_copy_ints(
            bag_ids + ((size_t)i * DMFAST_NUM_BAG),
            DMFAST_NUM_BAG,
            bag_ids_out
        );
        dmfast_copy_ints(
            bag_xp + ((size_t)i * DMFAST_NUM_BAG),
            DMFAST_NUM_BAG,
            bag_xp_out
        );
        dmfast_copy_ints(
            bag_specials + ((size_t)i * DMFAST_NUM_BAG * DMFAST_NUM_SPECIALS),
            DMFAST_NUM_BAG * DMFAST_NUM_SPECIALS,
            bag_specials_out
        );

        if (input_index >= 0 && input_index < DMFAST_NUM_BAG && bag_ids_out[input_index] != 0) {
            bag_ids_out[input_index] = 0;
            bag_xp_out[input_index] = 0;
            bag_specials_out[input_index * DMFAST_NUM_SPECIALS] = 0;
            bag_specials_out[input_index * DMFAST_NUM_SPECIALS + 1] = 0;
            bag_specials_out[input_index * DMFAST_NUM_SPECIALS + 2] = 0;
            valid = 1;
        }

        dmfast_inventory_recalculate_special_stats(
            health[i],
            base_vitality[i],
            special_stats + ((size_t)i * DMFAST_NUM_STATS),
            equipment_ids + ((size_t)i * DMFAST_NUM_EQUIPMENT),
            equipment_xp + ((size_t)i * DMFAST_NUM_EQUIPMENT),
            equipment_specials + ((size_t)i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
            bag_ids_out,
            bag_xp_out,
            bag_specials_out,
            out_health + i,
            stats_out
        );
        out_valid[i] = (uint8_t)valid;
        out_phase[i] = dmfast_inventory_final_phase(
            beast_present[i] != 0,
            stat_upgrades_available[i]
        );
    }
}
