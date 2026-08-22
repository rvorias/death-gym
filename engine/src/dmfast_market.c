#include "dmfast.h"
#include "dmfast_loot_internal.h"

#include <stdint.h>
#include <string.h>

enum {
    DMFAST_MARKET_NUM_ITEMS = 101,
    DMFAST_MARKET_MAX_SIZE = 25,
    DMFAST_MARKET_MAX_PASSES = 31,
    DMFAST_MARKET_SET_MAX_TABLE = 128,
    DMFAST_MARKET_SET_LINEAR_PROBES = 9,
    DMFAST_MARKET_SHA512_BLOCK_SIZE = 128,
    DMFAST_MARKET_SHA512_DIGEST_SIZE = 64,
    DMFAST_MARKET_TIER_PRICE = 4,
};

static const uint64_t DMFAST_MARKET_SHA512_H0[8] = {
    0x6a09e667f3bcc908ULL,
    0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL,
    0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL,
    0x5be0cd19137e2179ULL,
};

static const uint64_t DMFAST_MARKET_SHA512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL,
    0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
    0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
    0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL,
    0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
    0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL,
    0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL,
    0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL,
    0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL,
    0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
    0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL,
    0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
    0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
    0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL,
    0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL,
    0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
    0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
    0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL,
    0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
    0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
};

static uint64_t dmfast_market_rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

static uint64_t dmfast_market_load_be64(const uint8_t *p) {
    return ((uint64_t)p[0] << 56)
        | ((uint64_t)p[1] << 48)
        | ((uint64_t)p[2] << 40)
        | ((uint64_t)p[3] << 32)
        | ((uint64_t)p[4] << 24)
        | ((uint64_t)p[5] << 16)
        | ((uint64_t)p[6] << 8)
        | (uint64_t)p[7];
}

static void dmfast_market_store_be64(uint64_t value, uint8_t *p) {
    p[0] = (uint8_t)(value >> 56);
    p[1] = (uint8_t)(value >> 48);
    p[2] = (uint8_t)(value >> 40);
    p[3] = (uint8_t)(value >> 32);
    p[4] = (uint8_t)(value >> 24);
    p[5] = (uint8_t)(value >> 16);
    p[6] = (uint8_t)(value >> 8);
    p[7] = (uint8_t)value;
}

static void dmfast_market_sha512_16bytes(uint64_t a, uint64_t b, uint8_t out_digest[DMFAST_MARKET_SHA512_DIGEST_SIZE]) {
    uint8_t block[DMFAST_MARKET_SHA512_BLOCK_SIZE];
    uint64_t w[80];
    uint64_t h[8];
    uint64_t va;
    uint64_t vb;
    uint64_t vc;
    uint64_t vd;
    uint64_t ve;
    uint64_t vf;
    uint64_t vg;
    uint64_t vh;
    int i;

    memset(block, 0, sizeof(block));
    dmfast_market_store_be64(a, block);
    dmfast_market_store_be64(b, block + 8);
    block[16] = 0x80;
    dmfast_market_store_be64(0, block + 112);
    dmfast_market_store_be64(128, block + 120);

    for (i = 0; i < 16; ++i) {
        w[i] = dmfast_market_load_be64(block + (i * 8));
    }
    for (i = 16; i < 80; ++i) {
        uint64_t s0 = dmfast_market_rotr64(w[i - 15], 1)
            ^ dmfast_market_rotr64(w[i - 15], 8)
            ^ (w[i - 15] >> 7);
        uint64_t s1 = dmfast_market_rotr64(w[i - 2], 19)
            ^ dmfast_market_rotr64(w[i - 2], 61)
            ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    for (i = 0; i < 8; ++i) {
        h[i] = DMFAST_MARKET_SHA512_H0[i];
    }

    va = h[0];
    vb = h[1];
    vc = h[2];
    vd = h[3];
    ve = h[4];
    vf = h[5];
    vg = h[6];
    vh = h[7];

    for (i = 0; i < 80; ++i) {
        uint64_t s1 = dmfast_market_rotr64(ve, 14)
            ^ dmfast_market_rotr64(ve, 18)
            ^ dmfast_market_rotr64(ve, 41);
        uint64_t ch = (ve & vf) ^ ((~ve) & vg);
        uint64_t temp1 = vh + s1 + ch + DMFAST_MARKET_SHA512_K[i] + w[i];
        uint64_t s0 = dmfast_market_rotr64(va, 28)
            ^ dmfast_market_rotr64(va, 34)
            ^ dmfast_market_rotr64(va, 39);
        uint64_t maj = (va & vb) ^ (va & vc) ^ (vb & vc);
        uint64_t temp2 = s0 + maj;

        vh = vg;
        vg = vf;
        vf = ve;
        ve = vd + temp1;
        vd = vc;
        vc = vb;
        vb = va;
        va = temp1 + temp2;
    }

    h[0] += va;
    h[1] += vb;
    h[2] += vc;
    h[3] += vd;
    h[4] += ve;
    h[5] += vf;
    h[6] += vg;
    h[7] += vh;

    for (i = 0; i < 8; ++i) {
        dmfast_market_store_be64(h[i], out_digest + (i * 8));
    }
}

typedef struct {
    int32_t mask;
    int32_t fill;
    int32_t used;
    int32_t slots[DMFAST_MARKET_SET_MAX_TABLE];
} DMFastMarketIntSet;

static void dmfast_market_set_reset(DMFastMarketIntSet *set) {
    memset(set->slots, 0, sizeof(set->slots));
    set->mask = 7;
    set->fill = 0;
    set->used = 0;
}

static int32_t dmfast_market_set_lookup(const int32_t *slots, int32_t mask, int32_t key, int32_t *found) {
    int32_t i = key & mask;
    int32_t probes;
    int32_t j;
    int32_t entry = slots[i];
    if (entry == 0 || entry == key) {
        *found = (entry == key);
        return i;
    }

    while (1) {
        probes = (i + DMFAST_MARKET_SET_LINEAR_PROBES <= mask) ? DMFAST_MARKET_SET_LINEAR_PROBES : 0;
        j = i + 1;
        while (probes > 0) {
            entry = slots[j];
            if (entry == 0 || entry == key) {
                *found = (entry == key);
                return j;
            }
            j += 1;
            probes -= 1;
        }

        {
            uint64_t perturb = (uint64_t)key;
            while (1) {
                perturb >>= 5;
                i = (int32_t)(((uint64_t)(i * 5 + 1) + perturb) & (uint64_t)mask);
                entry = slots[i];
                if (entry == 0 || entry == key) {
                    *found = (entry == key);
                    return i;
                }
                probes = (i + DMFAST_MARKET_SET_LINEAR_PROBES <= mask) ? DMFAST_MARKET_SET_LINEAR_PROBES : 0;
                j = i + 1;
                while (probes > 0) {
                    entry = slots[j];
                    if (entry == 0 || entry == key) {
                        *found = (entry == key);
                        return j;
                    }
                    j += 1;
                    probes -= 1;
                }
            }
        }
    }
}

static void dmfast_market_set_resize(DMFastMarketIntSet *set, int32_t minused) {
    int32_t old_limit = set->mask + 1;
    int32_t old_slots[DMFAST_MARKET_SET_MAX_TABLE];
    int32_t newsize = 8;
    int32_t minsize = minused * 4;
    int32_t i;

    memcpy(old_slots, set->slots, sizeof(old_slots));
    while (newsize <= minsize) {
        newsize <<= 1;
    }

    memset(set->slots, 0, sizeof(set->slots));
    set->mask = newsize - 1;
    set->fill = set->used;

    for (i = 0; i < old_limit; ++i) {
        int32_t key = old_slots[i];
        if (key != 0) {
            int32_t found = 0;
            int32_t index = dmfast_market_set_lookup(set->slots, set->mask, key, &found);
            set->slots[index] = key;
        }
    }
}

static void dmfast_market_set_add(DMFastMarketIntSet *set, int32_t key) {
    int32_t found = 0;
    int32_t index = dmfast_market_set_lookup(set->slots, set->mask, key, &found);
    if (found) {
        return;
    }
    set->slots[index] = key;
    set->used += 1;
    set->fill += 1;
    if (set->fill * 5 >= set->mask * 3) {
        dmfast_market_set_resize(set, set->used);
    }
}

void dmfast_market_items(
    const uint64_t *adventurer_ids,
    const uint64_t *market_seeds,
    int32_t market_count,
    int32_t market_size,
    int32_t *out_item_ids
) {
    int32_t env;
    if (market_size < 0) {
        market_size = 0;
    }
    if (market_size > DMFAST_MARKET_MAX_SIZE) {
        market_size = DMFAST_MARKET_MAX_SIZE;
    }

    for (env = 0; env < market_count; ++env) {
        uint8_t digest[DMFAST_MARKET_SHA512_DIGEST_SIZE];
        DMFastMarketIntSet set;
        int32_t passes = 0;
        int32_t out_index = 0;
        int32_t i;

        dmfast_market_sha512_16bytes(adventurer_ids[env], market_seeds[env], digest);
        dmfast_market_set_reset(&set);

        while (passes < DMFAST_MARKET_MAX_PASSES && set.used < market_size) {
            uint8_t byte = digest[passes];
            passes += 1;
            if (byte < 202) {
                int32_t item_id = (int32_t)(byte % DMFAST_MARKET_NUM_ITEMS) + 1;
                dmfast_market_set_add(&set, item_id);
            }
        }

        for (i = 0; i <= set.mask && out_index < market_size; ++i) {
            int32_t key = set.slots[i];
            if (key != 0) {
                out_item_ids[env * market_size + out_index] = key;
                out_index += 1;
            }
        }
        while (out_index < market_size) {
            out_item_ids[env * market_size + out_index] = 0;
            out_index += 1;
        }
    }
}

void dmfast_market_discounted_prices(
    const int32_t *item_ids,
    const int32_t *discount,
    const int32_t *minimum_price,
    int32_t market_count,
    int32_t market_size,
    int32_t *out_prices
) {
    int32_t env;
    if (market_size < 0) {
        market_size = 0;
    }
    for (env = 0; env < market_count; ++env) {
        int32_t item_index;
        for (item_index = 0; item_index < market_size; ++item_index) {
            int32_t id = item_ids[env * market_size + item_index];
            int32_t tier = dmfast_loot_tier(id);
            int32_t base_price = tier == 0 ? 0 : (6 - tier) * DMFAST_MARKET_TIER_PRICE;
            int32_t adjusted = base_price - discount[env];
            if (adjusted < minimum_price[env]) {
                adjusted = minimum_price[env];
            }
            out_prices[env * market_size + item_index] = adjusted;
        }
    }
}
