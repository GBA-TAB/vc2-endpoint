#ifndef VC2_PRIV_H
#define VC2_PRIV_H
#include "vc2.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define VC2_NQ 116 // quantiser indices 0..115
extern uint32_t vc2_qf[VC2_NQ], vc2_qo[VC2_NQ]; // quantisation factor, reconstruction offset (+2 rounding included)
void vc2_tables_init(void);
int vc2_default_qm(int wavelet, int depth, uint8_t qm[VC2_MAX_DEPTH + 1][4]);

// One lifting step, written for synthesis: target[i] += sign * f(other).
//   tgt 0: the even (low) samples from the odd ones, taps at other[i-2], [i-1], [i], [i+1]
//   tgt 1: the odd (high) samples from the even ones, taps at other[i-1], [i], [i+1], [i+2]
// AVG2 / MUL2 use the two middle taps, DD4 all four (-1, 9, 9, -1), ONE a single tap ([i] in both cases).
enum { VC2_K_AVG2, VC2_K_DD4, VC2_K_MUL2, VC2_K_ONE };
typedef struct { uint8_t tgt, kind; int8_t sign; uint8_t shift; int16_t mul; } vc2_step;
typedef struct { int nsteps; vc2_step st[4]; int shift; int ok16; } vc2_wavelet;
const vc2_wavelet *vc2_wavelet_get(int idx);

// geometry shared by encoder and decoder
typedef struct {
    int w, h;                        // picture size of this component
    int pw, ph;                      // padded to a multiple of 2^depth
    int bw[VC2_MAX_DEPTH + 1], bh[VC2_MAX_DEPTH + 1]; // band size per level (level 0 = LL = level 1)
} vc2_geom;
void vc2_geom_init(vc2_geom *g, int w, int h, int depth);

// bit writer for headers (MSB first)
typedef struct { uint8_t *p; size_t pos; } vc2_bw;
void vc2_bw_bit(vc2_bw *b, int v);
void vc2_bw_uint(vc2_bw *b, uint32_t v);
void vc2_bw_align(vc2_bw *b);
size_t vc2_write_parse_info(uint8_t *p, int code, uint32_t next, uint32_t prev);
size_t vc2_write_seq(uint8_t *p, const vc2_seq *s);

// thread pool: run(fn, arg, n) calls fn(arg, job, thread) for job 0..n-1 on all threads, caller included
typedef struct vc2_pool vc2_pool;
typedef void (*vc2_job_fn)(void *arg, int job, int thread);
vc2_pool *vc2_pool_new(int threads, const int *cpus);
void vc2_pool_free(vc2_pool *p);
void vc2_pool_run(vc2_pool *p, vc2_job_fn fn, void *arg, int n);
int vc2_pool_threads(const vc2_pool *p);

static inline void *vc2_alloc(size_t n) { void *p = 0; if (posix_memalign(&p, 64, n + 64)) return 0; memset(p, 0, n + 64); return p; }
static inline int vc2_clz64(uint64_t v) { return __builtin_clzll(v); }
#endif
