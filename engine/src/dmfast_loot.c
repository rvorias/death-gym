#include "dmfast.h"
#include "dmfast_loot_internal.h"

#include <math.h>
#include <stdint.h>

enum {
    DMFAST_LOOT_TYPE_NONE = 0,
    DMFAST_LOOT_TYPE_MAGIC_OR_CLOTH = 1,
    DMFAST_LOOT_TYPE_BLADE_OR_HIDE = 2,
    DMFAST_LOOT_TYPE_BLUDGEON_OR_METAL = 3,
    DMFAST_LOOT_TYPE_NECKLACE = 4,
    DMFAST_LOOT_TYPE_RING = 5,
};

enum {
    DMFAST_LOOT_SLOT_NONE = 0,
    DMFAST_LOOT_SLOT_WEAPON = 1,
    DMFAST_LOOT_SLOT_CHEST = 2,
    DMFAST_LOOT_SLOT_HEAD = 3,
    DMFAST_LOOT_SLOT_WAIST = 4,
    DMFAST_LOOT_SLOT_FOOT = 5,
    DMFAST_LOOT_SLOT_HAND = 6,
    DMFAST_LOOT_SLOT_NECK = 7,
    DMFAST_LOOT_SLOT_RING = 8,
};

enum {
    DMFAST_LOOT_NAME_PREFIX_LENGTH = 69,
    DMFAST_LOOT_NAME_SUFFIX_LENGTH = 18,
    DMFAST_LOOT_ITEM_SUFFIX_LENGTH = 16,
    DMFAST_LOOT_NUM_ITEMS = 101,
    DMFAST_LOOT_SUFFIX_UNLOCK_GREATNESS = 15,
    DMFAST_LOOT_PREFIXES_UNLOCK_GREATNESS = 19,
};

typedef struct {
    int id;
    const char *name;
    int tier;
    int slot;
    int type;
} DMFastLootInfo;

static const DMFastLootInfo DMFAST_LOOT_INFO[102] = {
    {0, "None", 0, 0, 0},
    {1, "Pendant", 1, 7, 4},
    {2, "Necklace", 1, 7, 4},
    {3, "Amulet", 1, 7, 4},
    {4, "SilverRing", 2, 8, 5},
    {5, "BronzeRing", 3, 8, 5},
    {6, "PlatinumRing", 1, 8, 5},
    {7, "TitaniumRing", 1, 8, 5},
    {8, "GoldRing", 1, 8, 5},
    {9, "GhostWand", 1, 1, 1},
    {10, "GraveWand", 2, 1, 1},
    {11, "BoneWand", 3, 1, 1},
    {12, "Wand", 5, 1, 1},
    {13, "Grimoire", 1, 1, 1},
    {14, "Chronicle", 2, 1, 1},
    {15, "Tome", 3, 1, 1},
    {16, "Book", 5, 1, 1},
    {17, "DivineRobe", 1, 2, 1},
    {18, "SilkRobe", 2, 2, 1},
    {19, "LinenRobe", 3, 2, 1},
    {20, "Robe", 4, 2, 1},
    {21, "Shirt", 5, 2, 1},
    {22, "Crown", 1, 3, 1},
    {23, "DivineHood", 2, 3, 1},
    {24, "SilkHood", 3, 3, 1},
    {25, "LinenHood", 4, 3, 1},
    {26, "Hood", 5, 3, 1},
    {27, "BrightsilkSash", 1, 4, 1},
    {28, "SilkSash", 2, 4, 1},
    {29, "WoolSash", 3, 4, 1},
    {30, "LinenSash", 4, 4, 1},
    {31, "Sash", 5, 4, 1},
    {32, "DivineSlippers", 1, 5, 1},
    {33, "SilkSlippers", 2, 5, 1},
    {34, "WoolShoes", 3, 5, 1},
    {35, "LinenShoes", 4, 5, 1},
    {36, "Shoes", 5, 5, 1},
    {37, "DivineGloves", 1, 6, 1},
    {38, "SilkGloves", 2, 6, 1},
    {39, "WoolGloves", 3, 6, 1},
    {40, "LinenGloves", 4, 6, 1},
    {41, "Gloves", 5, 6, 1},
    {42, "Katana", 1, 1, 2},
    {43, "Falchion", 2, 1, 2},
    {44, "Scimitar", 3, 1, 2},
    {45, "LongSword", 4, 1, 2},
    {46, "ShortSword", 5, 1, 2},
    {47, "DemonHusk", 1, 2, 2},
    {48, "DragonskinArmor", 2, 2, 2},
    {49, "StuddedLeatherArmor", 3, 2, 2},
    {50, "HardLeatherArmor", 4, 2, 2},
    {51, "LeatherArmor", 5, 2, 2},
    {52, "DemonCrown", 1, 3, 2},
    {53, "DragonsCrown", 2, 3, 2},
    {54, "WarCap", 3, 3, 2},
    {55, "LeatherCap", 4, 3, 2},
    {56, "Cap", 5, 3, 2},
    {57, "DemonhideBelt", 1, 4, 2},
    {58, "DragonskinBelt", 2, 4, 2},
    {59, "StuddedLeatherBelt", 3, 4, 2},
    {60, "HardLeatherBelt", 4, 4, 2},
    {61, "LeatherBelt", 5, 4, 2},
    {62, "DemonhideBoots", 1, 5, 2},
    {63, "DragonskinBoots", 2, 5, 2},
    {64, "StuddedLeatherBoots", 3, 5, 2},
    {65, "HardLeatherBoots", 4, 5, 2},
    {66, "LeatherBoots", 5, 5, 2},
    {67, "DemonsHands", 1, 6, 2},
    {68, "DragonskinGloves", 2, 6, 2},
    {69, "StuddedLeatherGloves", 3, 6, 2},
    {70, "HardLeatherGloves", 4, 6, 2},
    {71, "LeatherGloves", 5, 6, 2},
    {72, "Warhammer", 1, 1, 3},
    {73, "Quarterstaff", 2, 1, 3},
    {74, "Maul", 3, 1, 3},
    {75, "Mace", 4, 1, 3},
    {76, "Club", 5, 1, 3},
    {77, "HolyChestplate", 1, 2, 3},
    {78, "OrnateChestplate", 2, 2, 3},
    {79, "PlateMail", 3, 2, 3},
    {80, "ChainMail", 4, 2, 3},
    {81, "RingMail", 5, 2, 3},
    {82, "AncientHelm", 1, 3, 3},
    {83, "OrnateHelm", 2, 3, 3},
    {84, "GreatHelm", 3, 3, 3},
    {85, "FullHelm", 4, 3, 3},
    {86, "Helm", 5, 3, 3},
    {87, "OrnateBelt", 1, 4, 3},
    {88, "WarBelt", 2, 4, 3},
    {89, "PlatedBelt", 3, 4, 3},
    {90, "MeshBelt", 4, 4, 3},
    {91, "HeavyBelt", 5, 4, 3},
    {92, "HolyGreaves", 1, 5, 3},
    {93, "OrnateGreaves", 2, 5, 3},
    {94, "Greaves", 3, 5, 3},
    {95, "ChainBoots", 4, 5, 3},
    {96, "HeavyBoots", 5, 5, 3},
    {97, "HolyGauntlets", 1, 6, 3},
    {98, "OrnateGauntlets", 2, 6, 3},
    {99, "Gauntlets", 3, 6, 3},
    {100, "ChainGloves", 4, 6, 3},
    {101, "HeavyGloves", 5, 6, 3},
};

static const DMFastLootInfo *dmfast_loot_info(int id) {
    if (id < 0 || id > DMFAST_LOOT_NUM_ITEMS) {
        return &DMFAST_LOOT_INFO[0];
    }
    return &DMFAST_LOOT_INFO[id];
}

static int dmfast_loot_is_necklace(int id) {
    return id < 4;
}

static int dmfast_loot_is_ring(int id) {
    return id > 3 && id < 9;
}

static int dmfast_loot_is_weapon(int id) {
    return ((id > 8 && id < 17) || (id > 41 && id < 47) || (id > 71 && id < 77));
}

static int dmfast_loot_is_chest_armor(int id) {
    return ((id > 16 && id < 22) || (id > 46 && id < 52) || (id > 76 && id < 82));
}

static int dmfast_loot_is_head_armor(int id) {
    return ((id > 21 && id < 27) || (id > 51 && id < 57) || (id > 81 && id < 87));
}

static int dmfast_loot_is_waist_armor(int id) {
    return ((id > 26 && id < 32) || (id > 56 && id < 62) || (id > 86 && id < 92));
}

static int dmfast_loot_is_foot_armor(int id) {
    return ((id > 31 && id < 37) || (id > 61 && id < 67) || (id > 91 && id < 97));
}

static int dmfast_loot_is_hand_armor(int id) {
    return ((id > 36 && id < 42) || (id > 66 && id < 72) || id > 96);
}

int dmfast_loot_type(int id) {
    return dmfast_loot_info(id)->type;
}

int dmfast_loot_slot(int id) {
    return dmfast_loot_info(id)->slot;
}

static int dmfast_loot_specials_slot(int id) {
    if (dmfast_loot_is_necklace(id)) {
        return DMFAST_LOOT_SLOT_NECK;
    }
    if (dmfast_loot_is_ring(id)) {
        return DMFAST_LOOT_SLOT_RING;
    }
    if (dmfast_loot_is_weapon(id)) {
        return DMFAST_LOOT_SLOT_WEAPON;
    }
    if (dmfast_loot_is_chest_armor(id)) {
        return DMFAST_LOOT_SLOT_CHEST;
    }
    if (dmfast_loot_is_head_armor(id)) {
        return DMFAST_LOOT_SLOT_HEAD;
    }
    if (dmfast_loot_is_waist_armor(id)) {
        return DMFAST_LOOT_SLOT_WAIST;
    }
    if (dmfast_loot_is_foot_armor(id)) {
        return DMFAST_LOOT_SLOT_FOOT;
    }
    if (dmfast_loot_is_hand_armor(id)) {
        return DMFAST_LOOT_SLOT_HAND;
    }
    return DMFAST_LOOT_SLOT_NONE;
}

int dmfast_loot_tier(int id) {
    return dmfast_loot_info(id)->tier;
}

static int dmfast_loot_tier_monotone(int id) {
    int tier = dmfast_loot_tier(id);
    return tier == 0 ? 0 : 6 - tier;
}

static int dmfast_loot_slot_length(int slot) {
    if (slot == DMFAST_LOOT_SLOT_WEAPON) {
        return 18;
    }
    if (slot == DMFAST_LOOT_SLOT_NECK) {
        return 3;
    }
    if (slot == DMFAST_LOOT_SLOT_RING) {
        return 5;
    }
    if (slot >= DMFAST_LOOT_SLOT_CHEST && slot <= DMFAST_LOOT_SLOT_HAND) {
        return 15;
    }
    return 0;
}

static int dmfast_loot_item_index(int id) {
    if (id >= 72 && id <= 76) {
        return id - 72;
    }
    if (id >= 42 && id <= 46) {
        return 5 + (id - 42);
    }
    if (id >= 9 && id <= 12) {
        return 10 + (id - 9);
    }
    if (id >= 13 && id <= 16) {
        return 14 + (id - 13);
    }
    if (id >= 17 && id <= 21) {
        return id - 17;
    }
    if (id >= 47 && id <= 51) {
        return 5 + (id - 47);
    }
    if (id >= 77 && id <= 81) {
        return 10 + (id - 77);
    }
    if (id >= 82 && id <= 86) {
        return id - 82;
    }
    if (id >= 52 && id <= 56) {
        return 5 + (id - 52);
    }
    if (id >= 22 && id <= 26) {
        return 10 + (id - 22);
    }
    if (id >= 87 && id <= 91) {
        return id - 87;
    }
    if (id >= 57 && id <= 61) {
        return 5 + (id - 57);
    }
    if (id >= 27 && id <= 31) {
        return 10 + (id - 27);
    }
    if (id >= 92 && id <= 96) {
        return id - 92;
    }
    if (id >= 62 && id <= 66) {
        return 5 + (id - 62);
    }
    if (id >= 32 && id <= 36) {
        return 10 + (id - 32);
    }
    if (id >= 97 && id <= 101) {
        return id - 97;
    }
    if (id >= 67 && id <= 71) {
        return 5 + (id - 67);
    }
    if (id >= 37 && id <= 41) {
        return 10 + (id - 37);
    }
    if (id == 2) {
        return 0;
    }
    if (id == 3) {
        return 1;
    }
    if (id == 1) {
        return 2;
    }
    if (id == 8) {
        return 0;
    }
    if (id == 4) {
        return 1;
    }
    if (id == 5) {
        return 2;
    }
    if (id == 6) {
        return 3;
    }
    if (id == 7) {
        return 4;
    }
    return 0;
}

static int dmfast_loot_specials_seed(int item_id, int entropy) {
    uint16_t item_id_u16 = (uint16_t)item_id;
    uint16_t entropy_u16 = (uint16_t)entropy;
    uint16_t item_entropy;
    /* Held in a variable rather than compared inline: gcc 12+ reads
       `0xFFFFu - x` on a uint16 as a promoted bitwise complement and rejects
       the comparison under -Wsign-compare. Same value, portable warning-free. */
    uint16_t headroom = (uint16_t)(0xFFFFu - item_id_u16);
    if (entropy_u16 > headroom) {
        item_entropy = (uint16_t)(entropy_u16 - item_id_u16);
    } else {
        item_entropy = (uint16_t)(entropy_u16 + item_id_u16);
    }
    int rnd = item_entropy % DMFAST_LOOT_NUM_ITEMS;
    int item_index = dmfast_loot_item_index(item_id);
    int slot_length = dmfast_loot_slot_length(dmfast_loot_specials_slot(item_id));
    return rnd * slot_length + item_index;
}

static int dmfast_loot_prefix1(int item_id, int seed) {
    return (dmfast_loot_specials_seed(item_id, seed) % DMFAST_LOOT_NAME_PREFIX_LENGTH) + 1;
}

static int dmfast_loot_prefix2(int item_id, int seed) {
    return (dmfast_loot_specials_seed(item_id, seed) % DMFAST_LOOT_NAME_SUFFIX_LENGTH) + 1;
}

static int dmfast_loot_suffix(int item_id, int seed) {
    return (dmfast_loot_specials_seed(item_id, seed) % DMFAST_LOOT_ITEM_SUFFIX_LENGTH) + 1;
}

void dmfast_loot_specials_for_item(int item_id, int greatness, int seed, int *s1, int *s2, int *s3) {
    if (greatness < DMFAST_LOOT_SUFFIX_UNLOCK_GREATNESS) {
        *s1 = 0;
        *s2 = 0;
        *s3 = 0;
        return;
    }
    *s1 = dmfast_loot_suffix(item_id, seed);
    if (greatness < DMFAST_LOOT_PREFIXES_UNLOCK_GREATNESS) {
        *s2 = 0;
        *s3 = 0;
        return;
    }
    *s2 = dmfast_loot_prefix1(item_id, seed);
    *s3 = dmfast_loot_prefix2(item_id, seed);
}

static int dmfast_item_greatness(int xp) {
    if (xp <= 0) {
        return 1;
    }
    int greatness = (int)floor(sqrt((double)xp));
    if (greatness > 20) {
        greatness = 20;
    }
    return greatness;
}

void dmfast_item_specials(
    const int32_t *item_ids,
    const int32_t *greatness,
    int32_t seed,
    int32_t count,
    int32_t *out_specials
) {
    for (int32_t i = 0; i < count; ++i) {
        int s1 = 0;
        int s2 = 0;
        int s3 = 0;
        dmfast_loot_specials_for_item(item_ids[i], greatness[i], seed, &s1, &s2, &s3);
        out_specials[i * 3 + 0] = s1;
        out_specials[i * 3 + 1] = s2;
        out_specials[i * 3 + 2] = s3;
    }
}

void dmfast_item_vectors(
    const int32_t *item_ids,
    const int32_t *item_xp,
    int32_t specials_seed,
    int32_t count,
    float *out_vectors
) {
    for (int32_t i = 0; i < count; ++i) {
        int id = item_ids[i];
        int xp = item_xp[i];
        int s1 = 0;
        int s2 = 0;
        int s3 = 0;
        if (specials_seed != 0) {
            dmfast_loot_specials_for_item(id, 20, specials_seed, &s1, &s2, &s3);
        }
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 0] = (float)id;
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 1] = (float)dmfast_loot_tier_monotone(id) / 5.0f;
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 2] = (float)dmfast_loot_type(id);
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 3] = (float)dmfast_loot_slot(id);
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 4] = (float)s1;
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 5] = (float)s2;
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 6] = (float)s3;
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 7] = (float)xp / 400.0f;
        out_vectors[i * DMFAST_NUM_ITEM_FIELDS + 8] = (float)(dmfast_item_greatness(xp) - 1) / 20.0f;
    }
}
