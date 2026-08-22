CC ?= gcc
# Portable by default, because participants build this on compilers we have
# never seen. -Werror turns a new compiler's new warning into a failed build
# on code that did not change; -march=native bakes the build host's ISA into
# the .so and is worth under 2% here, inside benchmark noise. Both are back on
# in `just build-strict`, which is the maintainer's check.
CFLAGS ?= -O3 -fPIC -std=c11 -Wall -Wextra
LDFLAGS ?= -shared -lm

BUILD_DIR := build
SRC_DIR := src
TARGET := $(BUILD_DIR)/libdmfast.so

SOURCES := $(wildcard $(SRC_DIR)/dmfast_*.c)
HEADERS := $(SRC_DIR)/dmfast.h $(SRC_DIR)/dmfast_internal.h $(SRC_DIR)/dmfast_loot_internal.h
OBJECTS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
