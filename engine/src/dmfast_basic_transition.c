#include "dmfast_internal.h"

static int dmfast_basic_final_phase(int beast_present, int stat_upgrades_available) {
    if (!beast_present && stat_upgrades_available > 0) {
        return DMFAST_PHASE_UPGRADE;
    }
    return beast_present ? DMFAST_PHASE_COMBAT : DMFAST_PHASE_MARKET;
}

void dmfast_process_stat_upgrades(
    const int32_t *health,
    const int32_t *stats,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *stat_index,
    int32_t count,
    uint8_t *out_valid,
    int32_t *out_phase,
    int32_t *out_health,
    int32_t *out_stats,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        int32_t *stats_out = out_stats + ((size_t)i * DMFAST_NUM_STATS);
        int32_t *special_stats_out = out_special_stats + ((size_t)i * DMFAST_NUM_STATS);
        int upgrades = stat_upgrades_available[i];
        int valid = 0;

        dmfast_copy_ints(
            stats + ((size_t)i * DMFAST_NUM_STATS),
            DMFAST_NUM_STATS,
            stats_out
        );
        dmfast_copy_ints(
            special_stats + ((size_t)i * DMFAST_NUM_STATS),
            DMFAST_NUM_STATS,
            special_stats_out
        );

        out_health[i] = health[i];
        if (upgrades > 0 && stat_index[i] >= 0 && stat_index[i] < 6) {
            stats_out[stat_index[i]] += 1;
            upgrades -= 1;
            valid = 1;

            if (stat_index[i] == DMFAST_STAT_VITALITY) {
                int max_health = dmfast_max_health(
                    stats_out[DMFAST_STAT_VITALITY],
                    special_stats_out[DMFAST_STAT_VITALITY]
                );

                out_health[i] += DMFAST_HEALTH_PER_VITALITY;
                if (out_health[i] > max_health) {
                    out_health[i] = max_health;
                }
            }
        }

        out_valid[i] = (uint8_t)valid;
        out_stat_upgrades_available[i] = upgrades;
        out_phase[i] = upgrades > 0
            ? DMFAST_PHASE_UPGRADE
            : DMFAST_PHASE_MARKET;
    }
}

void dmfast_process_buy_potions(
    const int32_t *adventurer_xp,
    const int32_t *health,
    const int32_t *gold,
    const int32_t *stats,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const uint8_t *beast_present,
    int32_t count,
    uint8_t *out_valid,
    int32_t *out_phase,
    int32_t *out_health,
    int32_t *out_gold,
    int32_t *out_stats,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available,
    int32_t *out_potions_bought
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        const int32_t *stats_in = stats + ((size_t)i * DMFAST_NUM_STATS);
        const int32_t *special_stats_in = special_stats + ((size_t)i * DMFAST_NUM_STATS);
        int total_charisma = stats_in[DMFAST_STAT_CHARISMA]
            + special_stats_in[DMFAST_STAT_CHARISMA];
        int level = dmfast_level_from_xp(adventurer_xp[i]);
        int potion_cost = level - total_charisma * DMFAST_CHARISMA_POTION_DISC;
        int max_health;
        int missing_hp;
        int max_without_overheal;
        int max_affordable;
        int potions_to_buy;
        int valid = 0;

        if (potion_cost < DMFAST_MIN_POTION_PRICE) {
            potion_cost = DMFAST_MIN_POTION_PRICE;
        }

        dmfast_copy_ints(stats_in, DMFAST_NUM_STATS, out_stats + ((size_t)i * DMFAST_NUM_STATS));
        dmfast_copy_ints(
            special_stats_in,
            DMFAST_NUM_STATS,
            out_special_stats + ((size_t)i * DMFAST_NUM_STATS)
        );
        out_stat_upgrades_available[i] = stat_upgrades_available[i];
        out_gold[i] = gold[i];
        out_health[i] = health[i];
        out_potions_bought[i] = 0;

        max_health = dmfast_max_health(stats_in[DMFAST_STAT_VITALITY], special_stats_in[DMFAST_STAT_VITALITY]);
        missing_hp = max_health - health[i];
        if (missing_hp < 0) {
            missing_hp = 0;
        }
        max_without_overheal = missing_hp / DMFAST_POTION_HEALTH_AMOUNT;
        max_affordable = potion_cost > 0 ? gold[i] / potion_cost : 0;
        potions_to_buy = max_without_overheal < max_affordable ? max_without_overheal : max_affordable;

        if (potions_to_buy > 0) {
            valid = 1;
            out_gold[i] = gold[i] - potions_to_buy * potion_cost;
            out_health[i] = health[i] + potions_to_buy * DMFAST_POTION_HEALTH_AMOUNT;
            if (out_health[i] > max_health) {
                out_health[i] = max_health;
            }
            out_potions_bought[i] = potions_to_buy;
        }

        out_valid[i] = (uint8_t)valid;
        out_phase[i] = dmfast_basic_final_phase(
            beast_present[i] != 0,
            stat_upgrades_available[i]
        );
    }
}
