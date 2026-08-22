#include "dmfast_internal.h"

static int dmfast_vitality_unlock_amount(int suffix) {
    switch (suffix) {
        case 2:
            return 3;
        case 5:
            return 1;
        case 8:
            return 2;
        case 11:
            return 1;
        default:
            return 0;
    }
}

void dmfast_grant_item_xp(
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *health,
    const int32_t *base_vitality,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *item_specials_seed,
    const int32_t *xp_amount,
    const int32_t *item_specials_rnd,
    int32_t count,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_health,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available,
    int32_t *out_item_specials_seed
) {
    int32_t i;

    for (i = 0; i < count; ++i) {
        const int32_t *ids_in = equipment_ids + (i * DMFAST_NUM_EQUIPMENT);
        const int32_t *xp_in = equipment_xp + (i * DMFAST_NUM_EQUIPMENT);
        const int32_t *specials_in = equipment_specials
            + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS);
        int32_t *xp_out = out_equipment_xp + (i * DMFAST_NUM_EQUIPMENT);
        int32_t *specials_out = out_equipment_specials
            + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS);
        int32_t *stats_out = out_special_stats + (i * DMFAST_NUM_STATS);
        int current_seed = item_specials_seed[i];
        int current_health = health[i];
        int upgrades = stat_upgrades_available[i];
        int slot_index;

        dmfast_copy_ints(xp_in, DMFAST_NUM_EQUIPMENT, xp_out);
        dmfast_copy_ints(
            specials_in,
            DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS,
            specials_out
        );
        dmfast_copy_ints(
            special_stats + (i * DMFAST_NUM_STATS),
            DMFAST_NUM_STATS,
            stats_out
        );

        for (slot_index = 0; slot_index < DMFAST_NUM_EQUIPMENT; ++slot_index) {
            int item_id = ids_in[slot_index];

            if (item_id != 0) {
                int previous_xp = xp_in[slot_index];
                int new_xp = previous_xp + xp_amount[i];
                int previous_greatness;
                int new_greatness;

                if (new_xp > DMFAST_ITEM_MAX_XP) {
                    new_xp = DMFAST_ITEM_MAX_XP;
                }
                xp_out[slot_index] = new_xp;
                previous_greatness = dmfast_greatness_from_xp(previous_xp);
                new_greatness = dmfast_greatness_from_xp(new_xp);

                if (new_greatness > previous_greatness) {
                    int suffix_unlocked = previous_greatness < DMFAST_SUFFIX_UNLOCK
                        && new_greatness >= DMFAST_SUFFIX_UNLOCK;
                    int prefixes_unlocked = previous_greatness < DMFAST_PREFIXES_UNLOCK
                        && new_greatness >= DMFAST_PREFIXES_UNLOCK;

                    if (new_greatness == DMFAST_ITEM_MAX_GREATNESS) {
                        upgrades += 1;
                    }
                    if (suffix_unlocked || prefixes_unlocked) {
                        int s1;
                        int s2;
                        int s3;
                        int seed_for_specials = current_seed != 0 ? current_seed : item_specials_rnd[i];

                        dmfast_loot_specials_for_item(item_id, new_greatness, seed_for_specials, &s1, &s2, &s3);
                        specials_out[slot_index * DMFAST_NUM_SPECIALS] = s1;
                        specials_out[slot_index * DMFAST_NUM_SPECIALS + 1] = s2;
                        specials_out[slot_index * DMFAST_NUM_SPECIALS + 2] = s3;

                        if (current_seed == 0) {
                            current_seed = item_specials_rnd[i];
                        }
                        if (suffix_unlocked) {
                            int vitality_bonus;

                            dmfast_apply_suffix_boost(stats_out, s1);
                            dmfast_apply_bag_boost(stats_out, s1);
                            vitality_bonus = dmfast_vitality_unlock_amount(s1);
                            if (vitality_bonus != 0) {
                                int bonus_health = vitality_bonus * DMFAST_VITALITY_INSTANT_BONUS;
                                int max_health = dmfast_max_health(
                                    base_vitality[i],
                                    stats_out[DMFAST_STAT_VITALITY]
                                );

                                current_health += bonus_health;
                                if (current_health > max_health) {
                                    current_health = max_health;
                                }
                            }
                        }
                    }
                }
            }
        }

        out_health[i] = current_health;
        out_stat_upgrades_available[i] = upgrades;
        out_item_specials_seed[i] = current_seed;
    }
}
