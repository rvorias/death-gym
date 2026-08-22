#include "dmfast_internal.h"

#include <stdlib.h>

enum {
    DMFAST_START_NUM_BASE_STATS = 6,
    DMFAST_START_NUM_STAT_ROLLS = 13,
    DMFAST_START_STARTING_HEALTH = 100,
    DMFAST_START_HEALTH_INCREASE_PER_VITALITY = 15,
    DMFAST_START_STARTER_BEAST_ATTACK_DAMAGE = 10,
    DMFAST_START_STARTING_GOLD = 40,
    DMFAST_START_STARTING_XP = 4,
    DMFAST_START_STAT_UPGRADES_AVAILABLE = 1,
    DMFAST_START_ACTION_COUNT = 1,
};

static void dmfast_apply_starting_stats(
    const int32_t *stat_rnds,
    int32_t *out_stats
) {
    int32_t i;

    memset(
        out_stats,
        0,
        (size_t)DMFAST_NUM_STATS * sizeof(*out_stats)
    );
    for (i = 0; i < DMFAST_START_NUM_STAT_ROLLS; ++i) {
        int32_t stat_index = stat_rnds[i];
        if (stat_index >= 0 && stat_index < DMFAST_START_NUM_BASE_STATS) {
            out_stats[stat_index] += 1;
        }
    }
}

void dmfast_process_starts(
    const uint64_t *settings_id,
    const int32_t *starting_weapon_id,
    const int32_t *starting_stat_rnds,
    const uint64_t *game_seed,
    const uint64_t *market_seed,
    int32_t count,
    int32_t *out_phase,
    uint64_t *out_game_seed,
    uint64_t *out_market_seed,
    int32_t *out_health,
    int32_t *out_xp,
    int32_t *out_gold,
    int32_t *out_stats,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available,
    int32_t *out_action_count,
    int32_t *out_item_specials_seed,
    int32_t *out_equipment_ids,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_bag_ids,
    int32_t *out_market_ids
) {
    int32_t i;
    uint64_t *market_settings_id;

    if (count <= 0) {
        return;
    }

    market_settings_id = (uint64_t *)malloc((size_t)count * sizeof(*market_settings_id));

    for (i = 0; i < count; ++i) {
        int32_t *stats = out_stats + ((size_t)i * DMFAST_NUM_STATS);
        int32_t *special_stats = out_special_stats + ((size_t)i * DMFAST_NUM_STATS);
        int32_t *equipment_ids = out_equipment_ids + ((size_t)i * DMFAST_NUM_EQUIPMENT_SLOTS);
        int32_t *equipment_xp = out_equipment_xp + ((size_t)i * DMFAST_NUM_EQUIPMENT_SLOTS);
        int32_t *equipment_specials = out_equipment_specials
            + ((size_t)i * DMFAST_NUM_EQUIPMENT_SLOTS * DMFAST_NUM_SPECIALS);
        int32_t *bag_ids = out_bag_ids + ((size_t)i * DMFAST_NUM_BAG_SLOTS);
        const int32_t *stat_rnds = starting_stat_rnds + ((size_t)i * DMFAST_START_NUM_STAT_ROLLS);

        dmfast_apply_starting_stats(stat_rnds, stats);
        memset(
            special_stats,
            0,
            (size_t)DMFAST_NUM_STATS * sizeof(*special_stats)
        );
        memset(
            equipment_ids,
            0,
            (size_t)DMFAST_NUM_EQUIPMENT_SLOTS * sizeof(*equipment_ids)
        );
        memset(
            equipment_xp,
            0,
            (size_t)DMFAST_NUM_EQUIPMENT_SLOTS * sizeof(*equipment_xp)
        );
        memset(
            equipment_specials,
            0,
            (size_t)DMFAST_NUM_EQUIPMENT_SLOTS
                * DMFAST_NUM_SPECIALS
                * sizeof(*equipment_specials)
        );
        memset(
            bag_ids,
            0,
            (size_t)DMFAST_NUM_BAG_SLOTS * sizeof(*bag_ids)
        );

        out_phase[i] = DMFAST_PHASE_UPGRADE;
        out_game_seed[i] = game_seed[i];
        out_market_seed[i] = market_seed[i];
        out_health[i] = DMFAST_START_STARTING_HEALTH
            + stats[2] * DMFAST_START_HEALTH_INCREASE_PER_VITALITY
            - DMFAST_START_STARTER_BEAST_ATTACK_DAMAGE;
        out_xp[i] = DMFAST_START_STARTING_XP;
        out_gold[i] = DMFAST_START_STARTING_GOLD;
        out_stat_upgrades_available[i] = DMFAST_START_STAT_UPGRADES_AVAILABLE;
        out_action_count[i] = DMFAST_START_ACTION_COUNT;
        out_item_specials_seed[i] = 0;
        equipment_ids[0] = starting_weapon_id[i];

        if (market_settings_id != NULL) {
            market_settings_id[i] = settings_id[i];
        }
    }

    memset(
        out_market_ids,
        0,
        (size_t)count * DMFAST_NUM_MARKET_SLOTS * sizeof(*out_market_ids)
    );
    if (market_settings_id != NULL) {
        dmfast_market_items(
            market_settings_id,
            market_seed,
            count,
            DMFAST_NUM_MARKET_SLOTS,
            out_market_ids
        );
        free(market_settings_id);
    }
}
