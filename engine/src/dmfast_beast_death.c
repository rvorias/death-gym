#include "dmfast_internal.h"

static int dmfast_beast_gold_reward(int beast_tier, int beast_level) {
    return ((6 - beast_tier) * beast_level) / DMFAST_GOLD_REWARD_DIVISOR;
}

static int dmfast_beast_xp_reward(int beast_tier, int beast_level, int adventurer_level) {
    int level_decay_percentage = adventurer_level * 2;
    int reward_amount;
    int xp_reward;

    if (level_decay_percentage > 95) {
        level_decay_percentage = 95;
    }
    reward_amount = ((6 - beast_tier) * beast_level) / 2;
    xp_reward = (reward_amount * (100 - level_decay_percentage)) / 100;
    if (xp_reward < DMFAST_MIN_XP_REWARD) {
        return DMFAST_MIN_XP_REWARD;
    }
    return xp_reward;
}

void dmfast_process_beast_deaths(
    const int32_t *adventurer_xp,
    const int32_t *gold,
    const int32_t *health,
    const int32_t *base_vitality,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *item_specials_seed,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *beast_tier,
    const int32_t *beast_level,
    const int32_t *item_specials_rnd,
    int32_t count,
    int32_t *out_gold_reward,
    int32_t *out_xp_reward,
    int32_t *out_adventurer_xp,
    int32_t *out_gold,
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
        int base_gold_reward = dmfast_beast_gold_reward(beast_tier[i], beast_level[i]);
        int ring_id = equipment_ids[i * DMFAST_NUM_EQUIPMENT + 7];
        int ring_greatness = dmfast_greatness_from_xp(
            equipment_xp[i * DMFAST_NUM_EQUIPMENT + 7]
        );
        int ring_bonus = 0;
        int xp_reward;
        int new_adv_xp;
        int upgrades_after_level;
        int new_level;
        int item_xp_reward;

        if (ring_id == DMFAST_GOLD_RING_ID) {
            ring_bonus = (base_gold_reward * DMFAST_BEAST_GOLD_BONUS_PCT * ring_greatness) / 100;
        }
        out_gold_reward[i] = base_gold_reward + ring_bonus;
        out_gold[i] = gold[i] + out_gold_reward[i];

        xp_reward = dmfast_beast_xp_reward(beast_tier[i], beast_level[i], current_level);
        out_xp_reward[i] = xp_reward;
        new_adv_xp = adventurer_xp[i] + xp_reward;
        if (new_adv_xp > DMFAST_MAX_ADVENTURER_XP) {
            new_adv_xp = DMFAST_MAX_ADVENTURER_XP;
        }
        new_level = dmfast_level_from_xp(new_adv_xp);
        upgrades_after_level = stat_upgrades_available[i] + (new_level - current_level);
        out_adventurer_xp[i] = new_adv_xp;
        item_xp_reward = xp_reward * 2;

        dmfast_grant_item_xp(
            equipment_ids + (i * DMFAST_NUM_EQUIPMENT),
            equipment_xp + (i * DMFAST_NUM_EQUIPMENT),
            equipment_specials + (i * DMFAST_NUM_EQUIPMENT * DMFAST_NUM_SPECIALS),
            health + i,
            base_vitality + i,
            special_stats + (i * DMFAST_NUM_STATS),
            &upgrades_after_level,
            item_specials_seed + i,
            &item_xp_reward,
            item_specials_rnd + i,
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
