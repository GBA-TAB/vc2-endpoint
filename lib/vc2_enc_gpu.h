// The encoder's GPU front end (OpenCL, Mali / libmali): the two finest forward levels in 16 bits, the same
// numbers as analyse_level16 in vc2_enc.c. Built with make GPU=cl; without it vc2_encgpu_new returns 0.
#pragma once
#include <stdint.h>
typedef struct vc2_encgpu vc2_encgpu;
#ifndef VC2_OPENCL
static inline vc2_encgpu *vc2_encgpu_new(const int pw[3], int ph) { (void)pw; (void)ph; return 0; }
static inline int16_t *vc2_encgpu_input(vc2_encgpu *g, int c) { (void)g; (void)c; return 0; }
static inline int vc2_encgpu_run(vc2_encgpu *g, int16_t *b4[3][4], int16_t *b3[3][4], double *gpu_ms) { (void)g; (void)b4; (void)b3; (void)gpu_ms; return -1; }
static inline void vc2_encgpu_release(vc2_encgpu *g) { (void)g; }
static inline void vc2_encgpu_free(vc2_encgpu *g) { (void)g; }
#else
// pw[c]: padded component widths (multiples of 64), ph: padded height (multiple of 16)
vc2_encgpu *vc2_encgpu_new(const int pw[3], int ph);
// the input planes (pw x ph, int16, offset removed) to fill; valid until vc2_encgpu_run
int16_t *vc2_encgpu_input(vc2_encgpu *g, int c);
// runs both levels and maps the bands for reading: b4[c][0..3] = LL, HL, LH, HH of the finest level (LL = the
// next level's input), b3[c][0..3] of the next; valid until vc2_encgpu_release. Returns 0, or -1 on failure.
int vc2_encgpu_run(vc2_encgpu *g, int16_t *b4[3][4], int16_t *b3[3][4], double *gpu_ms);
void vc2_encgpu_release(vc2_encgpu *g);
void vc2_encgpu_free(vc2_encgpu *g);
#endif
