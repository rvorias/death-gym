#ifndef DMFAST_H
#define DMFAST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DMFAST_NUM_BASE_ACTIONS = 11,
    DMFAST_NUM_BAG_SLOTS = 15,
    DMFAST_NUM_MARKET_SLOTS = 25,
    DMFAST_NUM_STAT_ACTIONS = 6,
    DMFAST_NUM_EQUIPMENT_SLOTS = 8,
    DMFAST_NUM_ITEM_FIELDS = 9,
    DMFAST_ACTION_DIM = 57,
    DMFAST_OBS_DIM = 463,
};

void dmfast_item_vectors(
    const int32_t *item_ids,
    const int32_t *item_xp,
    int32_t specials_seed,
    int32_t count,
    float *out_vectors
);
void dmfast_item_specials(
    const int32_t *item_ids,
    const int32_t *greatness,
    int32_t seed,
    int32_t count,
    int32_t *out_specials
);
void dmfast_beast_vectors(
    const int32_t *beast_ids,
    const int32_t *starting_health,
    const int32_t *levels,
    const int32_t *special2,
    const int32_t *special3,
    int32_t count,
    float *out_vectors
);
void dmfast_beast_specials(
    const int32_t *special2_seed,
    const int32_t *special3_seed,
    int32_t count,
    int32_t *out_specials
);
void dmfast_calculate_damage(
    const int32_t *weapon_tier,
    const int32_t *weapon_type,
    const int32_t *weapon_level,
    const int32_t *weapon_specials,
    const int32_t *armor_tier,
    const int32_t *armor_type,
    const int32_t *armor_level,
    const int32_t *armor_specials,
    const int32_t *minimum_damage,
    const int32_t *attacker_strength,
    const int32_t *defender_strength,
    const int32_t *critical_hit_chance,
    const int32_t *critical_hit_rnd,
    int32_t count,
    int32_t *out_results
);
void dmfast_adventurer_attack_totals(
    const int32_t *weapon_tier,
    const int32_t *weapon_type,
    const int32_t *weapon_level,
    const int32_t *weapon_specials,
    const int32_t *ring_id,
    const int32_t *ring_greatness,
    const int32_t *beast_tier,
    const int32_t *beast_type,
    const int32_t *beast_level,
    const int32_t *beast_specials,
    const int32_t *attacker_strength,
    const int32_t *critical_hit_chance,
    const int32_t *critical_hit_rnd,
    int32_t count,
    int32_t *out_results
);
void dmfast_adventurer_defend_totals(
    const int32_t *beast_tier,
    const int32_t *beast_type,
    const int32_t *beast_level,
    const int32_t *beast_specials,
    const int32_t *armor_tier,
    const int32_t *armor_type,
    const int32_t *armor_level,
    const int32_t *armor_specials,
    const int32_t *neck_id,
    const int32_t *neck_greatness,
    const int32_t *critical_hit_chance,
    const int32_t *critical_hit_rnd,
    int32_t count,
    int32_t *out_results
);
void dmfast_weapon_special_bonuses(
    const int32_t *base_damage,
    const int32_t *weapon_specials,
    const int32_t *armor_specials,
    int32_t count,
    int32_t *out_bonuses
);
void dmfast_random_levels(
    const int32_t *adventurer_level,
    const int32_t *seed,
    int32_t count,
    int32_t *out_levels
);
void dmfast_random_starting_health(
    const int32_t *adventurer_level,
    const int32_t *seed,
    int32_t count,
    int32_t *out_health
);
void dmfast_avoid_threat(
    const int32_t *adventurer_level,
    const int32_t *relevant_stat,
    const int32_t *rnd,
    int32_t count,
    uint8_t *out_success
);
void dmfast_attack_locations(
    const int32_t *seed_u8,
    int32_t count,
    int32_t *out_slot
);
void dmfast_damage_reductions(
    const int32_t *adventurer_level,
    const int32_t *relevant_stat,
    int32_t count,
    int32_t *out_reduction
);
void dmfast_apply_damage_reductions(
    const int32_t *damage,
    const int32_t *damage_reduction,
    int32_t count,
    int32_t *out_damage
);
/* Exact P(adventurer dies before killing the beast) for a fight-to-the-end
 * attack policy. Kept for direct mask/reward gating (the obs-feature variant
 * was tried and discarded — policies don't exploit tail-risk observations). */
float dmfast_p_death_exact(
    int adventurer_health,
    int beast_health,
    float crit_chance,
    int dmg_nocrit,
    int dmg_crit,
    float beast_crit_chance,
    const int *beast_dmg_nocrit,
    const int *beast_dmg_crit
);
void dmfast_simulate_battles(
    const int32_t *adventurer_health,
    const int32_t *beast_health,
    const int32_t *strength,
    const int32_t *luck,
    const int32_t *adventurer_level,
    const int32_t *weapon_id,
    const int32_t *weapon_tier,
    const int32_t *weapon_type,
    const int32_t *weapon_level,
    const int32_t *weapon_specials,
    const int32_t *neck_id,
    const int32_t *neck_greatness,
    const int32_t *ring_id,
    const int32_t *ring_greatness,
    const int32_t *armor_ids,
    const int32_t *armor_tier,
    const int32_t *armor_type,
    const int32_t *armor_level,
    const int32_t *armor_specials,
    const int32_t *beast_tier,
    const int32_t *beast_type,
    const int32_t *beast_level,
    const int32_t *beast_specials,
    int32_t count,
    float *out_stats
);
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
);
void dmfast_market_items(
    const uint64_t *adventurer_ids,
    const uint64_t *market_seeds,
    int32_t market_count,
    int32_t market_size,
    int32_t *out_item_ids
);
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
);
void dmfast_market_discounted_prices(
    const int32_t *item_ids,
    const int32_t *discount,
    const int32_t *minimum_price,
    int32_t market_count,
    int32_t market_size,
    int32_t *out_prices
);
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
);
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
);
void dmfast_process_obstacles(
    const int32_t *adventurer_xp,
    const int32_t *health,
    const int32_t *intelligence,
    const int32_t *base_vitality,
    const int32_t *base_damage_reduction,
    const int32_t *special_stats,
    const int32_t *stat_upgrades_available,
    const int32_t *item_specials_seed,
    const int32_t *equipment_ids,
    const int32_t *equipment_xp,
    const int32_t *equipment_specials,
    const int32_t *obstacle_seed,
    const int32_t *level_rnd,
    const int32_t *dmg_location_rnd,
    const int32_t *crit_hit_rnd,
    const int32_t *dodge_rnd,
    const int32_t *items_specials_rnd,
    int32_t count,
    int32_t *out_obstacle_id,
    uint8_t *out_dodged,
    int32_t *out_damage,
    int32_t *out_location,
    uint8_t *out_critical_hit,
    int32_t *out_xp_reward,
    int32_t *out_adventurer_xp,
    int32_t *out_health,
    int32_t *out_equipment_xp,
    int32_t *out_equipment_specials,
    int32_t *out_special_stats,
    int32_t *out_stat_upgrades_available,
    int32_t *out_item_specials_seed
);
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
);
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
);
/* Standalone beast swing: picks an armor slot from attack_location_rnd,
 * resolves damage via defend_totals, applies to health. Used by combat
 * attack/flee kernels internally and by ex_step_equip to charge a free
 * beast swing when the adventurer swaps gear mid-encounter. index=0 for
 * a single-env call. */
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
);
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
);
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
);
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
);
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
);
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
);
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
);
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
);
void dmfast_reward_compute_batch(
    const float *config,
    const float *prev_obs,
    const float *prev_info,
    const uint8_t *prev_mask,
    const int32_t *actions,
    const float *post_obs,
    const float *post_info,
    const uint8_t *terminated,
    const uint8_t *truncated,
    int32_t count,
    float *out_rewards
);

/* ─── Exact parity engine (uses tested kernels) ──────────────────────── */

typedef struct DMFastExactBatch DMFastExactBatch;

DMFastExactBatch *dmfast_exact_create(int32_t batch_size, uint64_t seed, int32_t max_steps);
void dmfast_exact_destroy(DMFastExactBatch *batch);
void dmfast_exact_reset_all(DMFastExactBatch *batch, uint64_t seed);
void dmfast_exact_step(DMFastExactBatch *batch, const int32_t *actions, int32_t auto_reset);

int32_t dmfast_exact_state_size(void);
void dmfast_exact_save_state(DMFastExactBatch *batch, int32_t idx, uint8_t *out);
void dmfast_exact_load_state(DMFastExactBatch *batch, int32_t idx, const uint8_t *in);
void dmfast_exact_broadcast_state(DMFastExactBatch *batch, const uint8_t *in,
                                  int32_t count, uint64_t reseed);
float *dmfast_exact_obs_ptr(DMFastExactBatch *batch);
uint8_t *dmfast_exact_action_mask_ptr(DMFastExactBatch *batch);
float *dmfast_exact_reward_ptr(DMFastExactBatch *batch);
uint8_t *dmfast_exact_terminated_ptr(DMFastExactBatch *batch);
uint8_t *dmfast_exact_truncated_ptr(DMFastExactBatch *batch);
int32_t *dmfast_exact_phase_ptr(DMFastExactBatch *batch);
int32_t *dmfast_exact_step_count_ptr(DMFastExactBatch *batch);
float *dmfast_exact_episode_return_ptr(DMFastExactBatch *batch);
float *dmfast_exact_last_episode_return_ptr(DMFastExactBatch *batch);
int32_t *dmfast_exact_last_episode_length_ptr(DMFastExactBatch *batch);
float *dmfast_exact_last_episode_info_ptr(DMFastExactBatch *batch);
float *dmfast_exact_last_terminal_obs_ptr(DMFastExactBatch *batch);
uint8_t *dmfast_exact_last_terminal_action_mask_ptr(DMFastExactBatch *batch);

float *dmfast_exact_info_ptr(DMFastExactBatch *batch);
int32_t dmfast_exact_info_dim(void);

int32_t dmfast_exact_set_reward_param(DMFastExactBatch *batch, const char *name, float value);

void dmfast_exact_sample_masked_actions(DMFastExactBatch *batch, int32_t *out_actions, uint64_t *rng_state);

#ifdef __cplusplus
}
#endif

#endif
