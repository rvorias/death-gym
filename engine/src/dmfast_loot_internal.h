#ifndef DMFAST_LOOT_INTERNAL_H
#define DMFAST_LOOT_INTERNAL_H

int dmfast_loot_type(int id);
int dmfast_loot_slot(int id);
int dmfast_loot_tier(int id);
void dmfast_loot_specials_for_item(int item_id, int greatness, int seed, int *s1, int *s2, int *s3);

#endif
