#include "dmfast_internal.h"

enum {
    DMFAST_ACT_EXPLORE = 0,
    DMFAST_ACT_ATTACK = 1,
    DMFAST_ACT_FLEE = 2,
    DMFAST_ACT_EQUIP = 3,
    DMFAST_ACT_DROP = 4,
    DMFAST_ACT_BUY_ITEM = 5,
    DMFAST_ACT_BUY_POTION = 6,
    DMFAST_ACT_MACRO_FIGHT = 7,
    DMFAST_ACT_MACRO_FLEE = 8,
    DMFAST_ACT_MACRO_EQUIP_BEST = 9,
    DMFAST_ACT_MACRO_EXPLORE_SOFTEN = 10,
};

static int dmfast_bag_is_full(const int32_t *bag_ids) {
    int i;
    for (i = 0; i < DMFAST_NUM_BAG; ++i) {
        if (bag_ids[i] == 0) {
            return 0;
        }
    }
    return 1;
}

static void dmfast_fill_slot_flags(const int32_t *slots, int n, uint8_t *slot_flags) {
    int i;
    memset(slot_flags, 0, (DMFAST_SLOT_RING + 1) * sizeof(uint8_t));
    for (i = 0; i < n; ++i) {
        int slot = slots[i];
        if (slot >= 0 && slot <= DMFAST_SLOT_RING) {
            slot_flags[slot] = 1;
        }
    }
}

static int dmfast_has_equippable_slot(const int32_t *bag_ids, const uint8_t *switch_slots) {
    int i;
    for (i = 0; i < DMFAST_NUM_BAG; ++i) {
        int slot = dmfast_loot_slot(bag_ids[i]);
        if (slot != DMFAST_SLOT_NONE && !switch_slots[slot]) {
            return 1;
        }
    }
    return 0;
}

static int dmfast_set_buy_item_mask(
    int gold,
    const int32_t *equipment_ids,
    const int32_t *bag_ids,
    const int32_t *market_ids,
    const int32_t *market_prices,
    const int32_t *buy_buffer_slots,
    uint8_t *market_mask
) {
    int i;
    uint8_t owned_ids[102];
    int best_tier[DMFAST_SLOT_RING + 1][6];
    uint8_t best_tier_present[DMFAST_SLOT_RING + 1][6];
    uint8_t buy_buffer_slot_flags[DMFAST_SLOT_RING + 1];
    int any_market = 0;

    memset(owned_ids, 0, sizeof(owned_ids));
    memset(best_tier_present, 0, sizeof(best_tier_present));
    memset(market_mask, 1, DMFAST_NUM_MARKET * sizeof(uint8_t));
    dmfast_fill_slot_flags(buy_buffer_slots, DMFAST_NUM_EQUIPMENT, buy_buffer_slot_flags);

    for (i = 0; i < DMFAST_NUM_EQUIPMENT; ++i) {
        int id = equipment_ids[i];
        int slot = dmfast_loot_slot(id);
        int type = dmfast_loot_type(id);
        int tier = dmfast_loot_tier(id);
        if (id >= 0 && id <= 101) {
            owned_ids[id] = 1;
        }
        if (slot != DMFAST_SLOT_NONE && slot != DMFAST_SLOT_NECK && slot != DMFAST_SLOT_RING) {
            best_tier[slot][type] = tier;
            best_tier_present[slot][type] = 1;
        }
    }

    for (i = 0; i < DMFAST_NUM_BAG; ++i) {
        int id = bag_ids[i];
        int slot = dmfast_loot_slot(id);
        int type = dmfast_loot_type(id);
        int tier = dmfast_loot_tier(id);
        if (id >= 0 && id <= 101) {
            owned_ids[id] = 1;
        }
        if (tier != 0) {
            if (best_tier_present[slot][type]) {
                if (tier < best_tier[slot][type]) {
                    best_tier[slot][type] = tier;
                }
            } else {
                best_tier[slot][type] = tier;
                best_tier_present[slot][type] = 1;
            }
        }
    }

    for (i = 0; i < DMFAST_NUM_MARKET; ++i) {
        int id = market_ids[i];
        int slot = dmfast_loot_slot(id);
        int type = dmfast_loot_type(id);
        int tier = dmfast_loot_tier(id);

        if (id >= 0 && id <= 101 && owned_ids[id]) {
            market_mask[i] = 0;
        } else if (market_prices[i] > gold) {
            market_mask[i] = 0;
        } else if (best_tier_present[slot][type] && tier >= best_tier[slot][type]) {
            market_mask[i] = 0;
        } else if (slot >= 0 && slot <= DMFAST_SLOT_RING && buy_buffer_slot_flags[slot]) {
            market_mask[i] = 0;
        } else if (id == 0) {
            market_mask[i] = 0;
        }

        if (market_mask[i]) {
            any_market = 1;
        }
    }

    return any_market;
}

static void dmfast_set_drop_mask(
    int game_phase,
    const int32_t *bag_ids,
    uint8_t *actions,
    uint8_t *bag_mask
) {
    if (!dmfast_bag_is_full(bag_ids)) {
        return;
    }
    if (game_phase == DMFAST_PHASE_MARKET) {
        actions[DMFAST_ACT_DROP] = 1;
    } else if (game_phase == DMFAST_PHASE_DROP) {
        memset(bag_mask, 1, DMFAST_NUM_BAG * sizeof(uint8_t));
    }
}

static void dmfast_set_equip_mask(
    int game_phase,
    const int32_t *bag_ids,
    const int32_t *switch_buffer_slots,
    uint8_t *actions,
    uint8_t *bag_mask
) {
    uint8_t switch_slots[DMFAST_SLOT_RING + 1];
    int i;
    int switch_count = 0;
    int equippable_count = 0;

    dmfast_fill_slot_flags(switch_buffer_slots, DMFAST_NUM_EQUIPMENT, switch_slots);
    for (i = 0; i < DMFAST_NUM_EQUIPMENT; ++i) {
        if (switch_buffer_slots[i] != 0) {
            switch_count += 1;
        }
    }

    if (game_phase == DMFAST_PHASE_EQUIP) {
        for (i = 0; i < DMFAST_NUM_BAG; ++i) {
            int slot = dmfast_loot_slot(bag_ids[i]);
            if (slot != DMFAST_SLOT_NONE && !switch_slots[slot]) {
                bag_mask[i] = 1;
            }
        }
        return;
    }

    if (game_phase == DMFAST_PHASE_COMBAT || game_phase == DMFAST_PHASE_MARKET) {
        equippable_count = dmfast_has_equippable_slot(bag_ids, switch_slots);
        if (switch_count < DMFAST_NUM_EQUIPMENT && equippable_count > 0) {
            actions[DMFAST_ACT_EQUIP] = 1;
        }
    }
}

static void dmfast_set_buy_potion_mask(
    int game_phase,
    int gold,
    int health,
    int max_health,
    int potion_cost,
    uint8_t *actions
) {
    int missing_hp;
    int max_without_overheal;
    int max_affordable;
    if (game_phase != DMFAST_PHASE_MARKET) {
        return;
    }
    if (potion_cost <= 0) {
        return;
    }
    missing_hp = max_health - health;
    if (missing_hp < 0) {
        missing_hp = 0;
    }
    max_without_overheal = missing_hp / DMFAST_POTION_HEALTH_AMOUNT;
    max_affordable = gold / potion_cost;
    if (max_without_overheal > 0 && max_affordable > 0) {
        actions[DMFAST_ACT_BUY_POTION] = 1;
    }
}

static void dmfast_set_upgrade_mask(const int32_t *stats, uint8_t *stat_mask) {
    int i;
    for (i = 0; i < DMFAST_NUM_STAT_ACTIONS; ++i) {
        stat_mask[i] = stats[i] < 31 ? 1 : 0;
    }
}

void dmfast_action_masks(
    const int32_t *game_phase,
    const int32_t *gold,
    const int32_t *health,
    const int32_t *max_health,
    const int32_t *potion_cost,
    const int32_t *stats,
    const int32_t *equipment_ids,
    const int32_t *bag_ids,
    const int32_t *market_ids,
    const int32_t *market_prices,
    const int32_t *switch_buffer_slots,
    const int32_t *buy_buffer_slots,
    int32_t count,
    uint8_t *out_masks
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        uint8_t *mask = out_masks + (i * DMFAST_ACTION_DIM);
        uint8_t *actions = mask;
        uint8_t *bag_mask = mask + DMFAST_NUM_BASE_ACTIONS;
        uint8_t *market_mask = bag_mask + DMFAST_NUM_BAG;
        uint8_t *stat_mask = market_mask + DMFAST_NUM_MARKET;
        int phase = game_phase[i];
        const int32_t *stats_i = stats + (i * DMFAST_NUM_STAT_ACTIONS);
        const int32_t *equipment_i = equipment_ids + (i * DMFAST_NUM_EQUIPMENT_SLOTS);
        const int32_t *bag_i = bag_ids + (i * DMFAST_NUM_BAG);
        const int32_t *market_i = market_ids + (i * DMFAST_NUM_MARKET);
        const int32_t *market_prices_i = market_prices + (i * DMFAST_NUM_MARKET);
        const int32_t *switch_slots_i = switch_buffer_slots + (i * DMFAST_NUM_EQUIPMENT_SLOTS);
        const int32_t *buy_slots_i = buy_buffer_slots + (i * DMFAST_NUM_EQUIPMENT_SLOTS);

        memset(mask, 0, DMFAST_ACTION_DIM * sizeof(uint8_t));

        if (phase == DMFAST_PHASE_UPGRADE) {
            dmfast_set_upgrade_mask(stats_i, stat_mask);
            continue;
        }

        if (phase == DMFAST_PHASE_MARKET) {
            uint8_t temp_market_mask[DMFAST_NUM_MARKET];
            dmfast_set_drop_mask(phase, bag_i, actions, bag_mask);
            dmfast_set_equip_mask(phase, bag_i, switch_slots_i, actions, bag_mask);
            if (dmfast_set_buy_item_mask(gold[i], equipment_i, bag_i, market_i, market_prices_i, buy_slots_i, temp_market_mask)) {
                actions[DMFAST_ACT_BUY_ITEM] = 1;
            }
            dmfast_set_buy_potion_mask(phase, gold[i], health[i], max_health[i], potion_cost[i], actions);
            actions[DMFAST_ACT_EXPLORE] = 1;
            actions[DMFAST_ACT_MACRO_EXPLORE_SOFTEN] = 1;
            continue;
        }

        if (phase == DMFAST_PHASE_DROP) {
            dmfast_set_drop_mask(phase, bag_i, actions, bag_mask);
            continue;
        }

        if (phase == DMFAST_PHASE_EQUIP) {
            dmfast_set_equip_mask(phase, bag_i, switch_slots_i, actions, bag_mask);
            continue;
        }

        if (phase == DMFAST_PHASE_COMBAT) {
            dmfast_set_equip_mask(phase, bag_i, switch_slots_i, actions, bag_mask);
            actions[DMFAST_ACT_ATTACK] = 1;
            actions[DMFAST_ACT_FLEE] = 1;
            actions[DMFAST_ACT_MACRO_FIGHT] = 1;
            actions[DMFAST_ACT_MACRO_FLEE] = 1;
            continue;
        }

        if (phase == DMFAST_PHASE_BUY) {
            dmfast_set_buy_item_mask(gold[i], equipment_i, bag_i, market_i, market_prices_i, buy_slots_i, market_mask);
            continue;
        }
    }
}
