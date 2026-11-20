# Out-of-tree build: make BUILD=~/build/vc2-endpoint  (caspar: nice -n 19 taskset -c 10,11 make ...)
BUILD ?= build
CC ?= gcc
ARCH := $(shell uname -m)
CFLAGS ?= -O3 -g -Wall -Wextra -std=gnu11 -fno-strict-aliasing -ffp-contract=off
ifeq ($(ARCH),x86_64)
CFLAGS += -mavx2 -mbmi2 -mlzcnt
endif
ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=native
endif
LIB := $(BUILD)/vc2_common.o $(BUILD)/vc2_dec.o $(BUILD)/vc2_enc.o
TOOLS := $(BUILD)/vc2enc $(BUILD)/vc2dec $(BUILD)/psnr $(BUILD)/vc2bench

all: $(TOOLS)
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/%.o: lib/%.c lib/vc2.h lib/vc2_priv.h lib/vc2_lift.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/libvc2.a: $(LIB)
	ar rcs $@ $^
$(BUILD)/%: tools/%.c $(BUILD)/libvc2.a
	$(CC) $(CFLAGS) $< -o $@ $(BUILD)/libvc2.a -lpthread -lm
clean:
	rm -rf $(BUILD)
.PHONY: all clean
