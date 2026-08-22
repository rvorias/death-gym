#include "dmfast_internal.h"

enum {
    DMFAST_NUM_DISCOVERY_ITEMS = 101,
};

static int dmfast_discovery_t1_pool[DMFAST_NUM_DISCOVERY_ITEMS];
static int dmfast_discovery_t2_pool[DMFAST_NUM_DISCOVERY_ITEMS];
static int dmfast_discovery_t3_pool[DMFAST_NUM_DISCOVERY_ITEMS];
static int dmfast_discovery_t4_pool[DMFAST_NUM_DISCOVERY_ITEMS];
static int dmfast_discovery_t5_pool[DMFAST_NUM_DISCOVERY_ITEMS];
static int dmfast_discovery_t1_count = 0;
static int dmfast_discovery_t2_count = 0;
static int dmfast_discovery_t3_count = 0;
static int dmfast_discovery_t4_count = 0;
static int dmfast_discovery_t5_count = 0;
static int dmfast_discovery_pools_ready = 0;

static void dmfast_init_discovery_pools(void) {
    int item_id;

    if (dmfast_discovery_pools_ready) {
        return;
    }

    dmfast_discovery_t1_count = 0;
    dmfast_discovery_t2_count = 0;
    dmfast_discovery_t3_count = 0;
    dmfast_discovery_t4_count = 0;
    dmfast_discovery_t5_count = 0;

    for (item_id = 1; item_id <= DMFAST_NUM_DISCOVERY_ITEMS; ++item_id) {
        int tier = dmfast_loot_tier(item_id);
        if (tier == DMFAST_TIER_T1) {
            dmfast_discovery_t1_pool[dmfast_discovery_t1_count++] = item_id;
        } else if (tier == DMFAST_TIER_T2) {
            dmfast_discovery_t2_pool[dmfast_discovery_t2_count++] = item_id;
        } else if (tier == DMFAST_TIER_T3) {
            dmfast_discovery_t3_pool[dmfast_discovery_t3_count++] = item_id;
        } else if (tier == DMFAST_TIER_T4) {
            dmfast_discovery_t4_pool[dmfast_discovery_t4_count++] = item_id;
        } else if (tier == DMFAST_TIER_T5) {
            dmfast_discovery_t5_pool[dmfast_discovery_t5_count++] = item_id;
        }
    }

    dmfast_discovery_pools_ready = 1;
}

static int dmfast_scale_u8_to_percent(int rnd) {
    return ((rnd * 100) + 127) / 255;
}

static int dmfast_market_price_for_tier(int tier) {
    if (tier == DMFAST_TIER_NONE) {
        return 0;
    }
    return (6 - tier) * DMFAST_TIER_PRICE;
}

static int dmfast_get_loot_discovery_one(int tier_rnd, int item_rnd) {
    int outcome;
    const int *pool;
    int pool_count;

    dmfast_init_discovery_pools();
    outcome = dmfast_scale_u8_to_percent(tier_rnd);
    if (outcome < 50) {
        pool = dmfast_discovery_t5_pool;
        pool_count = dmfast_discovery_t5_count;
    } else if (outcome < 80) {
        pool = dmfast_discovery_t4_pool;
        pool_count = dmfast_discovery_t4_count;
    } else if (outcome < 92) {
        pool = dmfast_discovery_t3_pool;
        pool_count = dmfast_discovery_t3_count;
    } else if (outcome < 98) {
        pool = dmfast_discovery_t2_pool;
        pool_count = dmfast_discovery_t2_count;
    } else {
        pool = dmfast_discovery_t1_pool;
        pool_count = dmfast_discovery_t1_count;
    }
    if (pool_count <= 0) {
        return 0;
    }
    return pool[item_rnd % pool_count];
}

static int dmfast_contains_id(const int32_t *items, int size, int item_id) {
    int i;

    for (i = 0; i < size; ++i) {
        if (items[i] == item_id) {
            return 1;
        }
    }
    return 0;
}

static int dmfast_is_bag_full(const int32_t *bag_ids) {
    int i;

    for (i = 0; i < DMFAST_NUM_BAG; ++i) {
        if (bag_ids[i] == 0) {
            return 0;
        }
    }
    return 1;
}

static void dmfast_copy_items(const int32_t *src, int size, int32_t *dst) {
    int i;

    for (i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

void dmfast_process_discoveries(
    const int32_t *discovery_type_rnd,
    const int32_t *amount_rnd1,
    const int32_t *amount_rnd2,
    const int32_t *max_health,
    const int32_t *health,
    const int32_t *gold,
    const int32_t *xp,
    const int32_t *stat_upgrades_available,
    const int32_t *equipment_ids,
    const int32_t *bag_ids,
    int32_t count,
    int32_t *out_discovery_type,
    int32_t *out_loot_item_id,
    int32_t *out_health,
    int32_t *out_gold,
    int32_t *out_xp,
    int32_t *out_stat_upgrades_available,
    int32_t *out_equipment_ids,
    int32_t *out_bag_ids
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        const int32_t *eq_in = equipment_ids + (i * DMFAST_NUM_EQUIPMENT);
        const int32_t *bag_in = bag_ids + (i * DMFAST_NUM_BAG);
        int32_t *eq_out = out_equipment_ids + (i * DMFAST_NUM_EQUIPMENT);
        int32_t *bag_out = out_bag_ids + (i * DMFAST_NUM_BAG);
        int prev_level = dmfast_level_from_xp(xp[i]);
        int new_xp = xp[i] + 1;
        int discovery_roll;
        int level_after_xp;

        if (new_xp > DMFAST_MAX_ADVENTURER_XP) {
            new_xp = DMFAST_MAX_ADVENTURER_XP;
        }

        dmfast_copy_items(eq_in, DMFAST_NUM_EQUIPMENT, eq_out);
        dmfast_copy_items(bag_in, DMFAST_NUM_BAG, bag_out);

        out_xp[i] = new_xp;
        out_health[i] = health[i];
        out_gold[i] = gold[i];
        out_loot_item_id[i] = 0;

        discovery_roll = dmfast_scale_u8_to_percent(discovery_type_rnd[i]);
        level_after_xp = dmfast_level_from_xp(new_xp);
        out_stat_upgrades_available[i] = stat_upgrades_available[i] + (level_after_xp - prev_level);

        /* Contract (_process_discovery): get_discovery() is called with the
         * adventurer's level BEFORE the +1 discovery XP is granted
         * (contracts.cairo:906 vs :909), so amounts scale with prev_level. */
        if (discovery_roll < 45) {
            out_discovery_type[i] = DMFAST_DISCOVERY_GOLD;
            out_gold[i] += (amount_rnd1[i] % prev_level) + 1;
            continue;
        }
        if (discovery_roll < 90) {
            int heal_amount = ((amount_rnd1[i] % prev_level) + 1) * 2;
            int new_health = health[i] + heal_amount;
            int allowed_health = max_health[i];

            out_discovery_type[i] = DMFAST_DISCOVERY_HEALTH;
            if (allowed_health < DMFAST_STARTING_HEALTH) {
                allowed_health = DMFAST_STARTING_HEALTH;
            }
            if (new_health > allowed_health) {
                new_health = allowed_health;
            }
            out_health[i] = new_health;
            continue;
        }

        {
            int discover_item_id = dmfast_get_loot_discovery_one(amount_rnd1[i], amount_rnd2[i]);
            int slot = dmfast_loot_slot(discover_item_id);
            int equipment_index = slot - 1;
            int already_item_at_slot = 0;
            int inventory_full = 0;

            out_discovery_type[i] = DMFAST_DISCOVERY_LOOT;
            out_loot_item_id[i] = discover_item_id;

            if (equipment_index >= 0 && equipment_index < DMFAST_NUM_EQUIPMENT) {
                already_item_at_slot = eq_out[equipment_index] != 0;
            }
            inventory_full = dmfast_is_bag_full(bag_out) && already_item_at_slot;

            if (
                dmfast_contains_id(eq_out, DMFAST_NUM_EQUIPMENT, discover_item_id)
                || dmfast_contains_id(bag_out, DMFAST_NUM_BAG, discover_item_id)
                || inventory_full
            ) {
                out_gold[i] += dmfast_market_price_for_tier(dmfast_loot_tier(discover_item_id));
                continue;
            }

            if (!already_item_at_slot && equipment_index >= 0 && equipment_index < DMFAST_NUM_EQUIPMENT) {
                eq_out[equipment_index] = discover_item_id;
            } else {
                int bag_index;

                for (bag_index = 0; bag_index < DMFAST_NUM_BAG; ++bag_index) {
                    if (bag_out[bag_index] == 0) {
                        bag_out[bag_index] = discover_item_id;
                        break;
                    }
                }
            }
        }
    }
}
