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
LIB := $(BUILD)/vc2_common.o $(BUILD)/vc2_dec.o $(BUILD)/vc2_enc.o $(BUILD)/vc2_rtp.o
# GPU path for the decoder's last wavelet level, optional: make GPU=cl (OpenCL on Mali / libmali),
# GPU=vk (Vulkan compute), GPU="cl vk". Without it nothing here needs OpenCL or Vulkan.
GPU ?=
GPULIBS :=
ifneq ($(filter cl,$(GPU)),)
CFLAGS += -DVC2_OPENCL
LIB += $(BUILD)/vc2_cl.o
GPULIBS += -l:libOpenCL.so.1
endif
ifneq ($(filter vk,$(GPU)),)
CFLAGS += -DVC2_VULKAN
LIB += $(BUILD)/vc2_vk.o
GPULIBS += -lvulkan
GLSLC ?= glslc   # or GLSLC="glslangValidator -V"
endif
TOOLS := $(BUILD)/vc2enc $(BUILD)/vc2dec $(BUILD)/psnr $(BUILD)/vc2bench $(BUILD)/vc2send $(BUILD)/vc2recv

all: $(TOOLS)
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/%.o: lib/%.c lib/vc2.h lib/vc2_priv.h lib/vc2_lift.h lib/vc2_gpu.h lib/vc2_dec_gpu.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/vc2_idwt.spv: lib/vc2_idwt.comp | $(BUILD)
	$(GLSLC) -fshader-stage=compute -O -o $@ $<
$(BUILD)/vc2_vk.o: lib/vc2_vk.c $(BUILD)/vc2_idwt.spv lib/vc2.h lib/vc2_gpu.h | $(BUILD)
	$(CC) $(CFLAGS) -DVC2_IDWT_SPV='"$(abspath $(BUILD)/vc2_idwt.spv)"' -c $< -o $@
$(BUILD)/vc2_cl.o: lib/vc2_cl.c lib/vc2_idwt.cl lib/vc2.h lib/vc2_gpu.h | $(BUILD)
	$(CC) $(CFLAGS) -DVC2_IDWT_CL='"$(abspath lib/vc2_idwt.cl)"' -c $< -o $@
$(BUILD)/libvc2.a: $(LIB)
	ar rcs $@ $^
$(BUILD)/%: tools/%.c $(BUILD)/libvc2.a
	$(CC) $(CFLAGS) $< -o $@ $(BUILD)/libvc2.a $(GPULIBS) -lpthread -lm
clean:
	rm -rf $(BUILD)
.PHONY: all clean player

# the DRM player (Firefly): make player BUILD=...
player: $(BUILD)/vc2play
$(BUILD)/vc2play: player/vc2play.c $(BUILD)/libvc2.a
	$(CC) $(CFLAGS) $< -o $@ $(BUILD)/libvc2.a $(GPULIBS) $(shell pkg-config --cflags --libs libdrm) -lpthread -lm
