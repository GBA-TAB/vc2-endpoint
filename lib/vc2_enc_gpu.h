// The encoder's GPU front end (OpenCL, Mali / libmali): the two finest forward levels in 16 bits, the same
// numbers as analyse_level16 in vc2_enc.c. Built with make GPU=cl; without it vc2_encgpu_new returns 0.
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef struct vc2_encgpu vc2_encgpu;
#ifndef VC2_OPENCL
static inline vc2_encgpu *vc2_encgpu_new(const int pw[3], int ph, int nx) { (void)pw; (void)ph; (void)nx; return 0; }
static inline int vc2_encgpu_base(const vc2_encgpu *g, int c, int level, int o) { (void)g; (void)c; (void)level; (void)o; return 0; }
static inline const uint32_t *vc2_encgpu_count(vc2_encgpu *g, int q, const float r16[18], double *gpu_ms) { (void)g; (void)q; (void)r16; (void)gpu_ms; return 0; }
static inline int16_t *vc2_encgpu_input(vc2_encgpu *g, int c) { (void)g; (void)c; return 0; }
static inline int vc2_encgpu_run(vc2_encgpu *g, int16_t *b4[3][4], int16_t *b3[3][4], double *gpu_ms) { (void)g; (void)b4; (void)b3; (void)gpu_ms; return -1; }
static inline void vc2_encgpu_release(vc2_encgpu *g) { (void)g; }
static inline void vc2_encgpu_free(vc2_encgpu *g) { (void)g; }
static inline int vc2_encgpu_use_picture(vc2_encgpu *g, const uint16_t *const plane[3], const ptrdiff_t stride[3], int h, int bit_depth) { (void)g; (void)plane; (void)stride; (void)h; (void)bit_depth; return -1; }
#else
// pw[c]: padded component widths (multiples of 64), ph: padded height (multiple of 16), nx: slices across
vc2_encgpu *vc2_encgpu_new(const int pw[3], int ph, int nx);
// the input planes (pw x ph, int16, offset removed) to fill; valid until vc2_encgpu_run
int16_t *vc2_encgpu_input(vc2_encgpu *g, int c);
// runs both levels and maps the bands for reading: b4[c][0..3] = LL, HL, LH, HH of the finest level (LL = the
// next level's input), b3[c][0..3] of the next; valid until vc2_encgpu_release. Returns 0, or -1 on failure.
int vc2_encgpu_run(vc2_encgpu *g, int16_t *b4[3][4], int16_t *b3[3][4], double *gpu_ms);
void vc2_encgpu_release(vc2_encgpu *g);
void vc2_encgpu_free(vc2_encgpu *g);
// The next vc2_encgpu_run reads the caller's picture itself (imported, no copy; clamp, offset and the padding rows
// in the kernel) instead of the input buffer. 0 if it can (the planes at least pw wide), -1 to fill the input.
int vc2_encgpu_use_picture(vc2_encgpu *g, const uint16_t *const plane[3], const ptrdiff_t stride[3], int h, int bit_depth);
// The counting of the 16-bit bands at quantiser index q (one result per quantiser and frame is kept):
// r16[i] = the float reciprocal of band i, in the order c = 0..2, level 4 then 3, orientation 1..3. Returns per
// (band row, slice) the pair { bits, truncation cost } at index vc2_encgpu_base(c, level, o) + row * nx + slice,
// valid until the next vc2_encgpu_run; 0 when it cannot (the caller counts on the CPU).
int vc2_encgpu_base(const vc2_encgpu *g, int c, int level, int o);
const uint32_t *vc2_encgpu_count(vc2_encgpu *g, int q, const float r16[18], double *gpu_ms);
#endif
