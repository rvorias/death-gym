#include "dmfast.h"

#include <math.h>
#include <stdint.h>

enum {
    DMFAST_BEAST_MAX_ID = 75,
    DMFAST_BEAST_MAX_PACKABLE_HEALTH = 1023,
    DMFAST_BEAST_MAX_OBS_COMBAT_LEVEL = 640,
    DMFAST_BEAST_MAX_SPECIAL2 = 69,
    DMFAST_BEAST_MAX_SPECIAL3 = 18,
};

typedef struct {
    int id;
    const char *name;
    int tier;
    int type;
} DMFastBeastInfo;

static const DMFastBeastInfo DMFAST_BEAST_INVALID_INFO = {-1, "Invalid", 0, 0};

static const DMFastBeastInfo DMFAST_BEAST_INFO[76] = {
    {0, "None", 5, 1},
    {1, "Warlock", 1, 1},
    {2, "Typhon", 1, 1},
    {3, "Jiangshi", 1, 1},
    {4, "Anansi", 1, 1},
    {5, "Basilisk", 1, 1},
    {6, "Gorgon", 2, 1},
    {7, "Kitsune", 2, 1},
    {8, "Lich", 2, 1},
    {9, "Chimera", 2, 1},
    {10, "Wendigo", 2, 1},
    {11, "Rakshasa", 3, 1},
    {12, "Werewolf", 3, 1},
    {13, "Banshee", 3, 1},
    {14, "Draugr", 3, 1},
    {15, "Vampire", 3, 1},
    {16, "Goblin", 4, 1},
    {17, "Ghoul", 4, 1},
    {18, "Wraith", 4, 1},
    {19, "Sprite", 4, 1},
    {20, "Kappa", 4, 1},
    {21, "Fairy", 5, 1},
    {22, "Leprechaun", 5, 1},
    {23, "Kelpie", 5, 1},
    {24, "Pixie", 5, 1},
    {25, "Gnome", 5, 1},
    {26, "Griffin", 1, 2},
    {27, "Manticore", 1, 2},
    {28, "Phoenix", 1, 2},
    {29, "Dragon", 1, 2},
    {30, "Minotaur", 1, 2},
    {31, "Qilin", 2, 2},
    {32, "Ammit", 2, 2},
    {33, "Nue", 2, 2},
    {34, "Skinwalker", 2, 2},
    {35, "Chupacabra", 2, 2},
    {36, "Weretiger", 3, 2},
    {37, "Wyvern", 3, 2},
    {38, "Roc", 3, 2},
    {39, "Harpy", 3, 2},
    {40, "Pegasus", 3, 2},
    {41, "Hippogriff", 4, 2},
    {42, "Fenrir", 4, 2},
    {43, "Jaguar", 4, 2},
    {44, "Satori", 4, 2},
    {45, "DireWolf", 4, 2},
    {46, "Bear", 5, 2},
    {47, "Wolf", 5, 2},
    {48, "Mantis", 5, 2},
    {49, "Spider", 5, 2},
    {50, "Rat", 5, 2},
    {51, "Kraken", 1, 3},
    {52, "Colossus", 1, 3},
    {53, "Balrog", 1, 3},
    {54, "Leviathan", 1, 3},
    {55, "Tarrasque", 1, 3},
    {56, "Titan", 2, 3},
    {57, "Nephilim", 2, 3},
    {58, "Behemoth", 2, 3},
    {59, "Hydra", 2, 3},
    {60, "Juggernaut", 2, 3},
    {61, "Oni", 3, 3},
    {62, "Jotunn", 3, 3},
    {63, "Ettin", 3, 3},
    {64, "Cyclops", 3, 3},
    {65, "Giant", 3, 3},
    {66, "NemeanLion", 4, 3},
    {67, "Berserker", 4, 3},
    {68, "Yeti", 4, 3},
    {69, "Golem", 4, 3},
    {70, "Ent", 4, 3},
    {71, "Troll", 5, 3},
    {72, "Bigfoot", 5, 3},
    {73, "Ogre", 5, 3},
    {74, "Orc", 5, 3},
    {75, "Skeleton", 5, 3},
};

static const DMFastBeastInfo *dmfast_beast_info(int id) {
    if (id < 0 || id > DMFAST_BEAST_MAX_ID) {
        return &DMFAST_BEAST_INVALID_INFO;
    }
    return &DMFAST_BEAST_INFO[id];
}

static int dmfast_positive_mod(int value, int modulus) {
    int remainder = value % modulus;
    if (remainder < 0) {
        remainder += modulus;
    }
    return remainder;
}

static float dmfast_beast_level_norm(int level) {
    if (level < 0) {
        level = 0;
    }
    return (float)(log1p((double)level) / log1p((double)DMFAST_BEAST_MAX_OBS_COMBAT_LEVEL));
}

void dmfast_beast_specials(
    const int32_t *special2_seed,
    const int32_t *special3_seed,
    int32_t count,
    int32_t *out_specials
) {
    for (int32_t i = 0; i < count; ++i) {
        out_specials[i * 3 + 0] = 0;
        out_specials[i * 3 + 1] = 1 + dmfast_positive_mod(special2_seed[i], DMFAST_BEAST_MAX_SPECIAL2);
        out_specials[i * 3 + 2] = 1 + dmfast_positive_mod(special3_seed[i], DMFAST_BEAST_MAX_SPECIAL3);
    }
}

void dmfast_beast_vectors(
    const int32_t *beast_ids,
    const int32_t *starting_health,
    const int32_t *levels,
    const int32_t *special2,
    const int32_t *special3,
    int32_t count,
    float *out_vectors
) {
    for (int32_t i = 0; i < count; ++i) {
        const DMFastBeastInfo *info = dmfast_beast_info(beast_ids[i]);
        int level = levels[i];
        out_vectors[i * 7 + 0] = (float)starting_health[i] / (float)DMFAST_BEAST_MAX_PACKABLE_HEALTH;
        out_vectors[i * 7 + 1] = info->tier == 0 ? 0.0f : (float)(6 - info->tier) / 5.0f;
        out_vectors[i * 7 + 2] = dmfast_beast_level_norm(level);
        out_vectors[i * 7 + 3] = (float)info->type;
        out_vectors[i * 7 + 4] = 0.0f;
        out_vectors[i * 7 + 5] = (float)special2[i];
        out_vectors[i * 7 + 6] = (float)special3[i];
    }
}
