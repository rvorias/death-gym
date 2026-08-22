#include "dmfast_internal.h"

enum {
    DMFAST_SPECIAL2_DAMAGE_MULTIPLIER = 8,
    DMFAST_SPECIAL3_DAMAGE_MULTIPLIER = 2,
    DMFAST_STRENGTH_DAMAGE_BONUS = 10,
    DMFAST_ELEMENTAL_DAMAGE_BONUS = 2,
    DMFAST_JEWELRY_BONUS_CRITICAL_HIT_PERCENT_PER_GREATNESS = 3,
    DMFAST_JEWELRY_BONUS_NAME_MATCH_PERCENT_PER_GREATNESS = 3,
    DMFAST_NUM_SIM_ARMOR_SLOTS = 5,
};

typedef struct {
    int base_attack;
    int base_armor;
    int elemental_adjusted_damage;
    int strength_bonus;
    int critical_hit_bonus;
    int weapon_special_bonus;
    int total_damage;
} DMFastCombatResult;

typedef struct {
    float survivability_adventurer;
    float survivability_beast;
    float survivability_risk;
    float average_damage_to_beast;
    float average_damage_to_adventurer;
    float average_damage_to_chest;
    float average_damage_to_head;
    float average_damage_to_waist;
    float average_damage_to_foot;
    float average_damage_to_hand;
} DMFastSimStats;

static int dmfast_tier_damage_multiplier(int tier) {
    if (tier >= DMFAST_TIER_T1 && tier <= DMFAST_TIER_T5) {
        return 6 - tier;
    }
    return 0;
}

static int dmfast_get_attack_hp(int tier, int level) {
    if (tier == DMFAST_TIER_NONE) {
        return 0;
    }
    return level * dmfast_tier_damage_multiplier(tier);
}

static int dmfast_get_armor_hp(int tier, int level) {
    if (tier == DMFAST_TIER_NONE) {
        return 0;
    }
    return level * dmfast_tier_damage_multiplier(tier);
}

static int dmfast_elemental_effectiveness(int weapon_type, int armor_type) {
    if (weapon_type == DMFAST_TYPE_NONE) {
        return 0;
    }
    if (weapon_type == DMFAST_TYPE_MAGIC_OR_CLOTH) {
        if (armor_type == DMFAST_TYPE_NONE || armor_type == DMFAST_TYPE_BLUDGEON_OR_METAL) {
            return 1;
        }
        if (armor_type == DMFAST_TYPE_BLADE_OR_HIDE) {
            return -1;
        }
        return 0;
    }
    if (weapon_type == DMFAST_TYPE_BLADE_OR_HIDE) {
        if (armor_type == DMFAST_TYPE_NONE || armor_type == DMFAST_TYPE_MAGIC_OR_CLOTH) {
            return 1;
        }
        if (armor_type == DMFAST_TYPE_BLUDGEON_OR_METAL) {
            return -1;
        }
        return 0;
    }
    if (weapon_type == DMFAST_TYPE_BLUDGEON_OR_METAL) {
        if (armor_type == DMFAST_TYPE_NONE || armor_type == DMFAST_TYPE_BLADE_OR_HIDE) {
            return 1;
        }
        if (armor_type == DMFAST_TYPE_MAGIC_OR_CLOTH) {
            return -1;
        }
        return 0;
    }
    return 0;
}

static int dmfast_elemental_adjusted_damage(int damage, int weapon_type, int armor_type) {
    int elemental_effect = damage / DMFAST_ELEMENTAL_DAMAGE_BONUS;
    int effectiveness = dmfast_elemental_effectiveness(weapon_type, armor_type);
    if (effectiveness < 0) {
        return damage - elemental_effect;
    }
    if (effectiveness > 0) {
        return damage + elemental_effect;
    }
    return damage;
}

static int dmfast_is_critical_hit(int chance, int rnd) {
    int scaled_chance = chance * 256;
    return scaled_chance > 100 * rnd;
}

static int dmfast_critical_hit_bonus(int base_damage, int critical_hit_chance, int rnd) {
    if (dmfast_is_critical_hit(critical_hit_chance, rnd)) {
        return base_damage;
    }
    return 0;
}

static int dmfast_special2_bonus(int base_damage, int weapon_special2, int armor_special2) {
    if (weapon_special2 != 0 && weapon_special2 == armor_special2) {
        return base_damage * DMFAST_SPECIAL2_DAMAGE_MULTIPLIER;
    }
    return 0;
}

static int dmfast_special3_bonus(int base_damage, int weapon_special3, int armor_special3) {
    if (weapon_special3 != 0 && weapon_special3 == armor_special3) {
        return base_damage * DMFAST_SPECIAL3_DAMAGE_MULTIPLIER;
    }
    return 0;
}

static int dmfast_weapon_special_bonus(int base_damage, const int32_t *weapon_specials, const int32_t *armor_specials) {
    int weapon_special2 = weapon_specials[1];
    int weapon_special3 = weapon_specials[2];
    int armor_special2 = armor_specials[1];
    int armor_special3 = armor_specials[2];
    return dmfast_special2_bonus(base_damage, weapon_special2, armor_special2)
        + dmfast_special3_bonus(base_damage, weapon_special3, armor_special3);
}

static int dmfast_strength_bonus(int damage, int strength) {
    if (strength == 0) {
        return 0;
    }
    return (damage * strength * DMFAST_STRENGTH_DAMAGE_BONUS) / 100;
}

static DMFastCombatResult dmfast_calculate_damage_one(
    int weapon_tier,
    int weapon_type,
    int weapon_level,
    const int32_t *weapon_specials,
    int armor_tier,
    int armor_type,
    int armor_level,
    const int32_t *armor_specials,
    int minimum_damage,
    int attacker_strength,
    int critical_hit_chance,
    int critical_hit_rnd
) {
    DMFastCombatResult result;
    int total_attack;

    result.base_attack = dmfast_get_attack_hp(weapon_tier, weapon_level);
    result.base_armor = dmfast_get_armor_hp(armor_tier, armor_level);
    result.elemental_adjusted_damage = dmfast_elemental_adjusted_damage(
        result.base_attack,
        weapon_type,
        armor_type
    );
    result.strength_bonus = dmfast_strength_bonus(result.elemental_adjusted_damage, attacker_strength);
    result.critical_hit_bonus = dmfast_critical_hit_bonus(
        result.elemental_adjusted_damage,
        critical_hit_chance,
        critical_hit_rnd
    );
    result.weapon_special_bonus = dmfast_weapon_special_bonus(
        result.elemental_adjusted_damage,
        weapon_specials,
        armor_specials
    );

    total_attack = result.elemental_adjusted_damage
        + result.strength_bonus
        + result.critical_hit_bonus
        + result.weapon_special_bonus;
    result.total_damage = minimum_damage;
    if (total_attack > result.base_armor + minimum_damage) {
        result.total_damage = total_attack - result.base_armor;
    }
    return result;
}

static int dmfast_ring_name_match_bonus(int ring_id, int ring_greatness, int weapon_special_bonus) {
    /* Contract: only PlatinumRing (id 6) grants the name-match bonus
     * (adventurer.cairo name_match_bonus_damage checks ItemId::PlatinumRing). */
    if (ring_id != 6) {
        return 0;
    }
    return (weapon_special_bonus * DMFAST_JEWELRY_BONUS_NAME_MATCH_PERCENT_PER_GREATNESS * ring_greatness) / 100;
}

static int dmfast_ring_critical_hit_bonus(int ring_id, int ring_greatness, int critical_hit_bonus) {
    /* Contract: only TitaniumRing (id 7) grants the critical-hit bonus. */
    if (ring_id != 7) {
        return 0;
    }
    return (critical_hit_bonus * DMFAST_JEWELRY_BONUS_CRITICAL_HIT_PERCENT_PER_GREATNESS * ring_greatness) / 100;
}

static int dmfast_necklace_armor_bonus(int neck_id, int neck_greatness, int armor_type, int base_armor) {
    if (armor_type == DMFAST_TYPE_MAGIC_OR_CLOTH) {
        if (neck_id != 3) {
            return 0;
        }
    } else if (armor_type == DMFAST_TYPE_BLADE_OR_HIDE) {
        if (neck_id != 1) {
            return 0;
        }
    } else if (armor_type == DMFAST_TYPE_BLUDGEON_OR_METAL) {
        if (neck_id != 2) {
            return 0;
        }
    } else {
        return 0;
    }
    return (base_armor * DMFAST_NECKLACE_ARMOR_BONUS * neck_greatness) / 100;
}

static int dmfast_attack_location_one(int seed_u8) {
    static const int mapping[5] = {
        DMFAST_SLOT_CHEST,
        DMFAST_SLOT_HEAD,
        DMFAST_SLOT_WAIST,
        DMFAST_SLOT_FOOT,
        DMFAST_SLOT_HAND,
    };
    return mapping[seed_u8 % 5];
}

static int dmfast_damage_reduction_one(int adventurer_level, int relevant_stat) {
    int64_t scale = 1000000;
    int64_t ratio;
    int64_t r2;
    int64_t r3;
    int64_t smooth;
    if (adventurer_level <= 0) {
        return 0;
    }
    ratio = (scale * relevant_stat) / adventurer_level;
    if (ratio > scale) {
        ratio = scale;
    }
    r2 = (ratio * ratio) / scale;
    r3 = (r2 * ratio) / scale;
    smooth = 3 * r2 - 2 * r3;
    return (int)((100 * smooth) / scale);
}

static int dmfast_apply_damage_reduction_one(int damage, int damage_reduction) {
    return (damage * (100 - damage_reduction)) / 100;
}

static int dmfast_beast_critical_hit_chance(int adventurer_level) {
    if (adventurer_level > 100) {
        return 100;
    }
    if (adventurer_level < 0) {
        return 0;
    }
    return adventurer_level;
}

static float dmfast_clip_upper(float value, float limit) {
    if (value > limit) {
        return limit;
    }
    return value;
}

/* Exact P(adventurer dies before killing the beast), fighting to the end
 * with attack actions. The fight is a race of two independent monotone
 * random walks whose per-round distributions are known exactly:
 *   - adventurer damage: 2 atoms (crit w.p. p / no-crit), so the kill time
 *     is a binomial first-passage: P(kill <= n) = P(crits_n >= c_n)
 *   - beast damage: 10 atoms (5 uniform slots x crit w.p. q / no-crit),
 *     handled by truncated convolution over integer HP
 * Round order matches the engine: adventurer strikes first; the beast only
 * retaliates if it survives, so a kill at round n means n-1 beast attacks.
 *   P(win) = sum_n P(kill time = n) * P(cum beast damage_{n-1} < hp)
 * Truncation remainder (cap 64 rounds; mass ~0 for non-degenerate fights
 * since both sides deal >= 2-4 minimum damage) is assigned to death. */
float dmfast_p_death_exact(
    int adventurer_health,
    int beast_health,
    float crit_chance,
    int dmg_nocrit,
    int dmg_crit,
    float beast_crit_chance,
    const int *beast_dmg_nocrit,
    const int *beast_dmg_crit
) {
    enum { PD_MAX_ROUNDS = 64, PD_MAX_HP = 1024 };
    double binom[PD_MAX_ROUNDS + 1];
    double pmf[PD_MAX_HP];
    double p, q, win, alive_prev, killed_cum_prev;
    int ah, n, k, v;

    if (adventurer_health <= 0) return 1.0f;
    if (beast_health <= 0) return 0.0f;
    ah = adventurer_health < PD_MAX_HP ? adventurer_health : PD_MAX_HP - 1;
    if (dmg_nocrit < 1) dmg_nocrit = 1;
    if (dmg_crit < dmg_nocrit) dmg_crit = dmg_nocrit;
    p = crit_chance < 0.0 ? 0.0 : (crit_chance > 1.0 ? 1.0 : (double)crit_chance);
    q = beast_crit_chance < 0.0 ? 0.0 : (beast_crit_chance > 1.0 ? 1.0 : (double)beast_crit_chance);

    binom[0] = 1.0;
    memset(pmf, 0, sizeof(pmf));
    pmf[0] = 1.0;
    alive_prev = 1.0;       /* P(cum beast dmg after 0 attacks < ah) */
    killed_cum_prev = 0.0;
    win = 0.0;

    for (n = 1; n <= PD_MAX_ROUNDS; ++n) {
        double killed_cum, kill_at_n;
        int need, c_n;

        /* advance binomial crit-count pmf to n rounds (in place, high k first) */
        binom[n] = 0.0;
        for (k = n; k >= 1; --k) {
            binom[k] = binom[k] * (1.0 - p) + binom[k - 1] * p;
        }
        binom[0] *= (1.0 - p);

        /* crits needed so that n*dmg_nocrit + crits*delta >= beast_health */
        need = beast_health - n * dmg_nocrit;
        if (need <= 0) {
            c_n = 0;
        } else if (dmg_crit == dmg_nocrit) {
            c_n = n + 1; /* unreachable this round */
        } else {
            c_n = (need + (dmg_crit - dmg_nocrit) - 1) / (dmg_crit - dmg_nocrit);
        }
        killed_cum = 0.0;
        for (k = c_n; k <= n; ++k) {
            killed_cum += binom[k];
        }
        kill_at_n = killed_cum - killed_cum_prev;
        if (kill_at_n > 0.0) {
            win += kill_at_n * alive_prev;
        }
        killed_cum_prev = killed_cum;
        if (killed_cum > 1.0 - 1e-9) break;
        if (alive_prev < 1e-9) break; /* dead if the kill hasn't landed */

        /* beast attack #n: convolve cum-damage pmf (descending v is safe:
         * every atom moves mass strictly upward, already-visited cells). */
        for (v = ah - 1; v >= 0; --v) {
            double mass = pmf[v];
            int s;
            if (mass <= 0.0) continue;
            pmf[v] = 0.0;
            for (s = 0; s < DMFAST_NUM_SIM_ARMOR_SLOTS; ++s) {
                int nv_nc = v + beast_dmg_nocrit[s];
                int nv_c = v + beast_dmg_crit[s];
                if (nv_nc < ah) pmf[nv_nc] += mass * 0.2 * (1.0 - q);
                if (nv_c < ah) pmf[nv_c] += mass * 0.2 * q;
            }
        }
        alive_prev = 0.0;
        for (v = 0; v < ah; ++v) {
            alive_prev += pmf[v];
        }
    }

    win = win < 0.0 ? 0.0 : (win > 1.0 ? 1.0 : win);
    return (float)(1.0 - win);
}

static DMFastSimStats dmfast_simulate_battle_one(
    int adventurer_health,
    int beast_health,
    int strength,
    int luck,
    int adventurer_level,
    int weapon_tier,
    int weapon_type,
    int weapon_level,
    const int32_t *weapon_specials,
    int neck_id,
    int neck_greatness,
    int ring_id,
    int ring_greatness,
    const int32_t *armor_tier,
    const int32_t *armor_type,
    const int32_t *armor_level,
    const int32_t *armor_specials,
    int beast_tier,
    int beast_type,
    int beast_level,
    const int32_t *beast_specials
) {
    DMFastSimStats stats;
    DMFastCombatResult base_damage_to_beast;
    float crit_chance;
    float average_damage_to_beast;
    float survivability_beast;
    float beast_crit_chance;
    float damage_by_slot[DMFAST_NUM_SIM_ARMOR_SLOTS];
    int slot_dmg_nocrit[DMFAST_NUM_SIM_ARMOR_SLOTS];
    int slot_dmg_crit[DMFAST_NUM_SIM_ARMOR_SLOTS];
    int adv_dmg_nocrit, adv_dmg_crit;
    float average_damage_to_adventurer;
    float survivability_adventurer;
    float survivability_risk;
    int slot_index;

    stats.survivability_adventurer = 0.0f;
    stats.survivability_beast = 0.0f;
    stats.survivability_risk = 0.0f;
    stats.average_damage_to_beast = 0.0f;
    stats.average_damage_to_adventurer = 0.0f;
    stats.average_damage_to_chest = 0.0f;
    stats.average_damage_to_head = 0.0f;
    stats.average_damage_to_waist = 0.0f;
    stats.average_damage_to_foot = 0.0f;
    stats.average_damage_to_hand = 0.0f;

    if (beast_tier == DMFAST_TIER_NONE || beast_health <= 0 || adventurer_health <= 0) {
        return stats;
    }

    base_damage_to_beast = dmfast_calculate_damage_one(
        weapon_tier,
        weapon_type,
        weapon_level,
        weapon_specials,
        beast_tier,
        beast_type,
        beast_level,
        beast_specials,
        DMFAST_MIN_DAMAGE_TO_BEASTS,
        strength,
        0,
        0
    );
    crit_chance = (float)luck / 100.0f;
    {
        int ring_name_bonus = dmfast_ring_name_match_bonus(
            ring_id, ring_greatness, base_damage_to_beast.weapon_special_bonus);
        int ring_crit_bonus = dmfast_ring_critical_hit_bonus(
            ring_id, ring_greatness, base_damage_to_beast.elemental_adjusted_damage);
        average_damage_to_beast = ((1.0f - crit_chance) * (float)base_damage_to_beast.total_damage)
            + (crit_chance * (float)(base_damage_to_beast.total_damage
                + base_damage_to_beast.elemental_adjusted_damage + ring_crit_bonus))
            + (float)ring_name_bonus;
        adv_dmg_nocrit = base_damage_to_beast.total_damage + ring_name_bonus;
        adv_dmg_crit = base_damage_to_beast.total_damage
            + base_damage_to_beast.elemental_adjusted_damage + ring_crit_bonus + ring_name_bonus;
    }
    survivability_beast = (float)beast_health / average_damage_to_beast;

    beast_crit_chance = (float)dmfast_beast_critical_hit_chance(adventurer_level) / 100.0f;
    for (slot_index = 0; slot_index < DMFAST_NUM_SIM_ARMOR_SLOTS; ++slot_index) {
        DMFastCombatResult result = dmfast_calculate_damage_one(
            beast_tier,
            beast_type,
            beast_level,
            beast_specials,
            armor_tier[slot_index],
            armor_type[slot_index],
            armor_level[slot_index],
            armor_specials + (slot_index * 3),
            DMFAST_MIN_DAMAGE_FROM_BEASTS,
            0,
            0,
            0
        );
        int jewelry_bonus = dmfast_necklace_armor_bonus(
            neck_id,
            neck_greatness,
            armor_type[slot_index],
            result.base_armor
        );
        int damage_no_crit = result.total_damage;
        int damage_crit = result.total_damage + result.elemental_adjusted_damage;

        if (damage_no_crit > jewelry_bonus + DMFAST_MIN_DAMAGE_FROM_BEASTS) {
            damage_no_crit -= jewelry_bonus;
        } else {
            damage_no_crit = DMFAST_MIN_DAMAGE_FROM_BEASTS;
        }
        if (damage_crit > jewelry_bonus + DMFAST_MIN_DAMAGE_FROM_BEASTS) {
            damage_crit -= jewelry_bonus;
        } else {
            damage_crit = DMFAST_MIN_DAMAGE_FROM_BEASTS;
        }
        damage_by_slot[slot_index] = ((1.0f - beast_crit_chance) * (float)damage_no_crit)
            + (beast_crit_chance * (float)damage_crit);
        slot_dmg_nocrit[slot_index] = damage_no_crit;
        slot_dmg_crit[slot_index] = damage_crit;
    }

    average_damage_to_adventurer = (
        damage_by_slot[0] + damage_by_slot[1] + damage_by_slot[2] + damage_by_slot[3] + damage_by_slot[4]
    ) / 5.0f;
    survivability_adventurer = (float)adventurer_health / average_damage_to_adventurer;
    survivability_risk = survivability_beast / survivability_adventurer;

    stats.survivability_adventurer = dmfast_clip_upper(survivability_adventurer, 10.0f);
    stats.survivability_beast = dmfast_clip_upper(survivability_beast, 10.0f);
    stats.survivability_risk = dmfast_clip_upper(survivability_risk, 10.0f);
    /* p_death obs experiment (2026-06-11): emitting exact P(death) here in
     * place of surv_beast scored 230.1 vs bar 234.5 at 250M — the policy
     * does not exploit tail-risk observations at this budget. The exact
     * helper (dmfast_p_death_exact, declared in dmfast.h) is kept for
     * direct mask/reward gating, where no learning is required. */
    (void)adv_dmg_nocrit;
    (void)adv_dmg_crit;
    (void)slot_dmg_nocrit;
    (void)slot_dmg_crit;
    stats.average_damage_to_beast = average_damage_to_beast / 100.0f;
    stats.average_damage_to_adventurer = average_damage_to_adventurer / 100.0f;
    stats.average_damage_to_chest = damage_by_slot[0] / 100.0f;
    stats.average_damage_to_head = damage_by_slot[1] / 100.0f;
    stats.average_damage_to_waist = damage_by_slot[2] / 100.0f;
    stats.average_damage_to_foot = damage_by_slot[3] / 100.0f;
    stats.average_damage_to_hand = damage_by_slot[4] / 100.0f;
    return stats;
}

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
) {
    int32_t i;
    (void)defender_strength;
    for (i = 0; i < count; ++i) {
        DMFastCombatResult result = dmfast_calculate_damage_one(
            weapon_tier[i],
            weapon_type[i],
            weapon_level[i],
            weapon_specials + (i * 3),
            armor_tier[i],
            armor_type[i],
            armor_level[i],
            armor_specials + (i * 3),
            minimum_damage[i],
            attacker_strength[i],
            critical_hit_chance[i],
            critical_hit_rnd[i]
        );
        out_results[i * 7 + 0] = result.base_attack;
        out_results[i * 7 + 1] = result.base_armor;
        out_results[i * 7 + 2] = result.elemental_adjusted_damage;
        out_results[i * 7 + 3] = result.strength_bonus;
        out_results[i * 7 + 4] = result.critical_hit_bonus;
        out_results[i * 7 + 5] = result.weapon_special_bonus;
        out_results[i * 7 + 6] = result.total_damage;
    }
}

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
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        DMFastCombatResult result = dmfast_calculate_damage_one(
            weapon_tier[i],
            weapon_type[i],
            weapon_level[i],
            weapon_specials + (i * 3),
            beast_tier[i],
            beast_type[i],
            beast_level[i],
            beast_specials + (i * 3),
            DMFAST_MIN_DAMAGE_TO_BEASTS,
            attacker_strength[i],
            critical_hit_chance[i],
            critical_hit_rnd[i]
        );
        int name_bonus = dmfast_ring_name_match_bonus(
            ring_id[i],
            ring_greatness[i],
            result.weapon_special_bonus
        );
        int crit_bonus = dmfast_ring_critical_hit_bonus(
            ring_id[i],
            ring_greatness[i],
            result.critical_hit_bonus
        );
        int final_total = result.total_damage + name_bonus + crit_bonus;
        out_results[i * 10 + 0] = result.base_attack;
        out_results[i * 10 + 1] = result.base_armor;
        out_results[i * 10 + 2] = result.elemental_adjusted_damage;
        out_results[i * 10 + 3] = result.strength_bonus;
        out_results[i * 10 + 4] = result.critical_hit_bonus;
        out_results[i * 10 + 5] = result.weapon_special_bonus;
        out_results[i * 10 + 6] = result.total_damage;
        out_results[i * 10 + 7] = name_bonus;
        out_results[i * 10 + 8] = crit_bonus;
        out_results[i * 10 + 9] = final_total;
    }
}

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
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        DMFastCombatResult result = dmfast_calculate_damage_one(
            beast_tier[i],
            beast_type[i],
            beast_level[i],
            beast_specials + (i * 3),
            armor_tier[i],
            armor_type[i],
            armor_level[i],
            armor_specials + (i * 3),
            DMFAST_MIN_DAMAGE_FROM_BEASTS,
            0,
            critical_hit_chance[i],
            critical_hit_rnd[i]
        );
        int jewelry_bonus = dmfast_necklace_armor_bonus(
            neck_id[i],
            neck_greatness[i],
            armor_type[i],
            result.base_armor
        );
        int final_total = DMFAST_MIN_DAMAGE_FROM_BEASTS;
        if (result.total_damage > jewelry_bonus + DMFAST_MIN_DAMAGE_FROM_BEASTS) {
            final_total = result.total_damage - jewelry_bonus;
        }
        out_results[i * 9 + 0] = result.base_attack;
        out_results[i * 9 + 1] = result.base_armor;
        out_results[i * 9 + 2] = result.elemental_adjusted_damage;
        out_results[i * 9 + 3] = result.strength_bonus;
        out_results[i * 9 + 4] = result.critical_hit_bonus;
        out_results[i * 9 + 5] = result.weapon_special_bonus;
        out_results[i * 9 + 6] = result.total_damage;
        out_results[i * 9 + 7] = jewelry_bonus;
        out_results[i * 9 + 8] = final_total;
    }
}

void dmfast_weapon_special_bonuses(
    const int32_t *base_damage,
    const int32_t *weapon_specials,
    const int32_t *armor_specials,
    int32_t count,
    int32_t *out_bonuses
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        out_bonuses[i] = dmfast_weapon_special_bonus(
            base_damage[i],
            weapon_specials + (i * 3),
            armor_specials + (i * 3)
        );
    }
}

void dmfast_random_levels(
    const int32_t *adventurer_level,
    const int32_t *seed,
    int32_t count,
    int32_t *out_levels
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        out_levels[i] = dmfast_random_level(adventurer_level[i], seed[i]);
    }
}

void dmfast_random_starting_health(
    const int32_t *adventurer_level,
    const int32_t *seed,
    int32_t count,
    int32_t *out_health
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        out_health[i] = dmfast_random_beast_health(adventurer_level[i], seed[i]);
    }
}

void dmfast_avoid_threat(
    const int32_t *adventurer_level,
    const int32_t *relevant_stat,
    const int32_t *rnd,
    int32_t count,
    uint8_t *out_success
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        out_success[i] = (uint8_t)dmfast_avoid_threat_one(
            adventurer_level[i],
            relevant_stat[i],
            rnd[i]
        );
    }
}

void dmfast_attack_locations(
    const int32_t *seed_u8,
    int32_t count,
    int32_t *out_slot
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        out_slot[i] = dmfast_attack_location_one(seed_u8[i]);
    }
}

void dmfast_damage_reductions(
    const int32_t *adventurer_level,
    const int32_t *relevant_stat,
    int32_t count,
    int32_t *out_reduction
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        out_reduction[i] = dmfast_damage_reduction_one(
            adventurer_level[i],
            relevant_stat[i]
        );
    }
}

void dmfast_apply_damage_reductions(
    const int32_t *damage,
    const int32_t *damage_reduction,
    int32_t count,
    int32_t *out_damage
) {
    int32_t i;
    for (i = 0; i < count; ++i) {
        out_damage[i] = dmfast_apply_damage_reduction_one(
            damage[i],
            damage_reduction[i]
        );
    }
}

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
) {
    int32_t i;

    (void)weapon_id;
    (void)armor_ids;

    for (i = 0; i < count; ++i) {
        DMFastSimStats stats = dmfast_simulate_battle_one(
            adventurer_health[i],
            beast_health[i],
            strength[i],
            luck[i],
            adventurer_level[i],
            weapon_tier[i],
            weapon_type[i],
            weapon_level[i],
            weapon_specials + (i * 3),
            neck_id[i],
            neck_greatness[i],
            ring_id[i],
            ring_greatness[i],
            armor_tier + (i * DMFAST_NUM_SIM_ARMOR_SLOTS),
            armor_type + (i * DMFAST_NUM_SIM_ARMOR_SLOTS),
            armor_level + (i * DMFAST_NUM_SIM_ARMOR_SLOTS),
            armor_specials + (i * DMFAST_NUM_SIM_ARMOR_SLOTS * 3),
            beast_tier[i],
            beast_type[i],
            beast_level[i],
            beast_specials + (i * 3)
        );

        out_stats[i * 10 + 0] = stats.survivability_adventurer;
        out_stats[i * 10 + 1] = stats.survivability_beast;
        out_stats[i * 10 + 2] = stats.survivability_risk;
        out_stats[i * 10 + 3] = stats.average_damage_to_beast;
        out_stats[i * 10 + 4] = stats.average_damage_to_adventurer;
        out_stats[i * 10 + 5] = stats.average_damage_to_chest;
        out_stats[i * 10 + 6] = stats.average_damage_to_head;
        out_stats[i * 10 + 7] = stats.average_damage_to_waist;
        out_stats[i * 10 + 8] = stats.average_damage_to_foot;
        out_stats[i * 10 + 9] = stats.average_damage_to_hand;
    }
}
