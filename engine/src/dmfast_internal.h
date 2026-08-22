/*
 * dmfast_internal.h — Shared constants, types, and utility functions.
 *
 * All kernel .c files include this instead of redeclaring their own
 * copies of level_from_xp, greatness_from_xp, copy_ints, etc.
 */
#ifndef DMFAST_INTERNAL_H
#define DMFAST_INTERNAL_H

#include "dmfast.h"
#include "dmfast_loot_internal.h"

#include <stdint.h>
#include <string.h>

/* ─── Game phases ────────────────────────────────────────────────────── */

enum {
    DMFAST_PHASE_UPGRADE   = 0,
    DMFAST_PHASE_MARKET    = 1,
    DMFAST_PHASE_DROP      = 2,
    DMFAST_PHASE_EQUIP     = 3,
    DMFAST_PHASE_COMBAT    = 4,
    DMFAST_PHASE_BUY       = 5,
};

/* ─── Item tiers ─────────────────────────────────────────────────────── */

enum {
    DMFAST_TIER_NONE = 0,
    DMFAST_TIER_T1   = 1,
    DMFAST_TIER_T2   = 2,
    DMFAST_TIER_T3   = 3,
    DMFAST_TIER_T4   = 4,
    DMFAST_TIER_T5   = 5,
};

/* ─── Item/beast types ───────────────────────────────────────────────── */

enum {
    DMFAST_TYPE_NONE             = 0,
    DMFAST_TYPE_MAGIC_OR_CLOTH   = 1,
    DMFAST_TYPE_BLADE_OR_HIDE    = 2,
    DMFAST_TYPE_BLUDGEON_OR_METAL = 3,
    DMFAST_TYPE_NECKLACE         = 4,
    DMFAST_TYPE_RING             = 5,
};

/* ─── Equipment slots ────────────────────────────────────────────────── */

enum {
    DMFAST_SLOT_NONE   = 0,
    DMFAST_SLOT_WEAPON = 1,
    DMFAST_SLOT_CHEST  = 2,
    DMFAST_SLOT_HEAD   = 3,
    DMFAST_SLOT_WAIST  = 4,
    DMFAST_SLOT_FOOT   = 5,
    DMFAST_SLOT_HAND   = 6,
    DMFAST_SLOT_NECK   = 7,
    DMFAST_SLOT_RING   = 8,
};

/* ─── Stat indices ───────────────────────────────────────────────────── */

enum {
    DMFAST_STAT_STRENGTH     = 0,
    DMFAST_STAT_DEXTERITY    = 1,
    DMFAST_STAT_VITALITY     = 2,
    DMFAST_STAT_INTELLIGENCE = 3,
    DMFAST_STAT_WISDOM       = 4,
    DMFAST_STAT_CHARISMA     = 5,
    DMFAST_STAT_LUCK         = 6,
    DMFAST_NUM_STATS         = 7,
};

/* ─── Layout sizes ───────────────────────────────────────────────────── */

enum {
    DMFAST_NUM_EQUIPMENT = 8,
    DMFAST_NUM_BAG       = 15,
    DMFAST_NUM_MARKET    = 25,
    DMFAST_NUM_SPECIALS  = 3,
};

/* ─── Game constants ─────────────────────────────────────────────────── */

enum {
    DMFAST_MAX_ADVENTURER_XP      = 32767,
    DMFAST_MAX_ADVENTURER_HEALTH  = 1023,
    DMFAST_MAX_GOLD               = 511,
    DMFAST_STARTING_HEALTH        = 100,
    DMFAST_HEALTH_PER_VITALITY    = 15,
    DMFAST_VITALITY_INSTANT_BONUS = 15,
    DMFAST_POTION_HEALTH_AMOUNT   = 10,
    DMFAST_MIN_POTION_PRICE       = 1,
    DMFAST_CHARISMA_POTION_DISC   = 2,
    DMFAST_CHARISMA_ITEM_DISC     = 1,
    DMFAST_TIER_PRICE             = 4,
    DMFAST_MIN_ITEM_PRICE         = 1,
    DMFAST_BASE_DAMAGE_REDUCTION  = 25,
    DMFAST_BEAST_MAX_HEALTH       = 1023,
    DMFAST_ITEM_MAX_XP            = 400,
    DMFAST_ITEM_MAX_GREATNESS     = 20,
    DMFAST_SUFFIX_UNLOCK          = 15,
    DMFAST_PREFIXES_UNLOCK        = 19,
    DMFAST_NECKLACE_ARMOR_BONUS   = 3,
    DMFAST_MIN_DAMAGE_TO_BEASTS   = 4,
    DMFAST_MIN_DAMAGE_FROM_BEASTS = 2,
    DMFAST_MIN_DAMAGE_FROM_OBSTACLES = 4,
    DMFAST_MIN_OBSTACLE_XP_REWARD = 4,
    DMFAST_MIN_XP_REWARD          = 4,
    DMFAST_BEAST_MAX_ID           = 75,
    DMFAST_OBSTACLE_MAX_ID        = 75,
    DMFAST_BEAST_SPECIAL_UNLOCK   = 19,
    DMFAST_BEAST_STARTER_HEALTH   = 3,
    DMFAST_BEAST_MAX_SPECIAL2     = 69,
    DMFAST_BEAST_MAX_SPECIAL3     = 18,
    DMFAST_GOLD_RING_ID           = 8,
    DMFAST_SILVER_RING_ID         = 4,
    DMFAST_BEAST_GOLD_BONUS_PCT   = 3,
    DMFAST_GOLD_REWARD_DIVISOR    = 2,
};

/* ─── Explore result types ───────────────────────────────────────────── */

enum {
    DMFAST_EXPLORE_BEAST     = 0,
    DMFAST_EXPLORE_OBSTACLE  = 1,
    DMFAST_EXPLORE_DISCOVERY = 2,
};

/* ─── Discovery types ────────────────────────────────────────────────── */

enum {
    DMFAST_DISCOVERY_GOLD   = 0,
    DMFAST_DISCOVERY_HEALTH = 1,
    DMFAST_DISCOVERY_LOOT   = 2,
};

/* ─── Shared utility functions ───────────────────────────────────────── */

static inline int32_t dmfast_level_from_xp(int32_t xp) {
    int32_t level = 1;
    if (xp <= 0) return 1;
    while ((level + 1) * (level + 1) <= xp) ++level;
    return level;
}

static inline int32_t dmfast_greatness_from_xp(int32_t xp) {
    int32_t g = 1;
    if (xp <= 0) return 1;
    while ((g + 1) * (g + 1) <= xp && g < DMFAST_ITEM_MAX_GREATNESS) ++g;
    return g;
}

static inline void dmfast_copy_ints(const int32_t *src, int32_t count, int32_t *dst) {
    for (int32_t i = 0; i < count; ++i) dst[i] = src[i];
}

static inline int32_t dmfast_max_health(int32_t base_vit, int32_t special_vit) {
    int32_t mh = DMFAST_STARTING_HEALTH + (base_vit + special_vit) * DMFAST_HEALTH_PER_VITALITY;
    return mh > DMFAST_MAX_ADVENTURER_HEALTH ? DMFAST_MAX_ADVENTURER_HEALTH : mh;
}

static inline int32_t dmfast_random_level(int32_t adventurer_level, int32_t seed) {
    int32_t base = 1 + (seed % (adventurer_level * 3));
    if (adventurer_level >= 50) return base + 80;
    if (adventurer_level >= 40) return base + 40;
    if (adventurer_level >= 30) return base + 20;
    if (adventurer_level >= 20) return base + 10;
    return base;
}

static inline int32_t dmfast_random_beast_health(int32_t adventurer_level, int32_t seed) {
    int32_t h = 1 + (seed % (adventurer_level * 20));
    if (adventurer_level >= 50) h += 500;
    else if (adventurer_level >= 40) h += 400;
    else if (adventurer_level >= 30) h += 200;
    else if (adventurer_level >= 20) h += 100;
    else h += 10;
    return h > DMFAST_BEAST_MAX_HEALTH ? DMFAST_BEAST_MAX_HEALTH : h;
}

static inline int32_t dmfast_attempt_flee_one(int32_t adv_level, int32_t relevant_stat, int32_t rnd) {
    if (relevant_stat >= adv_level) return 1;
    return relevant_stat > (adv_level * rnd) / 255;
}

static inline int32_t dmfast_avoid_threat_one(int32_t adv_level, int32_t relevant_stat, int32_t rnd) {
    if (relevant_stat >= adv_level) return 1;
    return relevant_stat > (adv_level * rnd) / 255;
}

static inline void dmfast_apply_suffix_boost(int32_t *stats, int32_t suffix) {
    switch (suffix) {
        case 1:  stats[DMFAST_STAT_STRENGTH] += 3; break;
        case 3:  stats[DMFAST_STAT_STRENGTH] += 2; break;
        case 4:  stats[DMFAST_STAT_DEXTERITY] += 3; break;
        case 5:  stats[DMFAST_STAT_STRENGTH] += 1; stats[DMFAST_STAT_DEXTERITY] += 1; break;
        case 6:  stats[DMFAST_STAT_INTELLIGENCE] += 3; break;
        case 7:  stats[DMFAST_STAT_WISDOM] += 3; break;
        case 8:  stats[DMFAST_STAT_DEXTERITY] += 1; break;
        case 9:  stats[DMFAST_STAT_STRENGTH] += 2; stats[DMFAST_STAT_DEXTERITY] += 1; break;
        case 10: stats[DMFAST_STAT_STRENGTH] += 1; stats[DMFAST_STAT_WISDOM] += 1; break;
        case 11: stats[DMFAST_STAT_INTELLIGENCE] += 1; break;
        case 12: stats[DMFAST_STAT_INTELLIGENCE] += 2; stats[DMFAST_STAT_WISDOM] += 1; break;
        case 13: stats[DMFAST_STAT_DEXTERITY] += 2; break;
        case 14: stats[DMFAST_STAT_WISDOM] += 2; stats[DMFAST_STAT_DEXTERITY] += 1; break;
        case 15: stats[DMFAST_STAT_INTELLIGENCE] += 1; stats[DMFAST_STAT_WISDOM] += 2; break;
        default: break;
    }
}

static inline void dmfast_apply_bag_boost(int32_t *stats, int32_t suffix) {
    switch (suffix) {
        case 2:  stats[DMFAST_STAT_VITALITY] += 3; break;
        case 3:  stats[DMFAST_STAT_CHARISMA] += 1; break;
        case 5:  stats[DMFAST_STAT_VITALITY] += 1; break;
        case 8:  stats[DMFAST_STAT_VITALITY] += 2; break;
        case 10: stats[DMFAST_STAT_CHARISMA] += 1; break;
        case 11: stats[DMFAST_STAT_VITALITY] += 1; stats[DMFAST_STAT_CHARISMA] += 1; break;
        case 13: stats[DMFAST_STAT_CHARISMA] += 1; break;
        case 16: stats[DMFAST_STAT_CHARISMA] += 3; break;
        default: break;
    }
}

#endif /* DMFAST_INTERNAL_H */
