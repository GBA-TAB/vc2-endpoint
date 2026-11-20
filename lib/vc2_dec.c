// VC-2 HQ decoder.
//
// A "cascade" decodes slice rows in order: entropy decoding into linear buffers, dequantisation into the
// subbands, then the inverse wavelet of every level as far as the data allows. The subbands and the
// intermediate pictures are ring buffers of a few rows, so the working set stays in the core's cache and
// picture lines come out a fixed number of lines behind the slice row being decoded (sub-frame latency).
// For a whole frame at once, each thread runs its own cascade over a band of slice rows, started a few
// slice rows early: after that overlap its output no longer depends on what was above it.
// The deep levels run in 32 bits, the last level (3/4 of the samples) in 16 bits.
#define _GNU_SOURCE
#include "vc2_priv.h"
#include <time.h>
#ifdef VC2_PROF
#include <stdio.h>
#endif

#define T int32_t
#define SUF _32
#include "vc2_lift.h"
#undef T
#undef SUF
#define T int16_t
#define SUF _16
#define T16
#include "vc2_lift.h"
#undef T16
#undef T
#undef SUF

#define NL (VC2_MAX_DEPTH + 1)
#define LINE_RING 32 // NV20 staging rows (>= 2 * the largest strip)

typedef struct {          // one component: geometry, shared by all cascades
    vc2_geom g;
    int *xs[NL], *ys[NL]; // slice boundaries per level
    int brows[NL];        // ring rows of the subbands of a level ([0] = coded LL)
    int lrows[NL];        // ring rows of the output of level l (the LL input of level l + 1)
    uint16_t *pix;
    ptrdiff_t pstride;
} comp_t;

typedef struct { const uint8_t *buf; size_t bp, endbit; int16_t *out; int i, n; int *coded; } vlc_stream;

typedef struct {          // one cascade
    void *band[3][NL][4]; // [0][0] = coded LL; [l][1..3] = HL, LH, HH
    void *ll[3][NL];
    int syn_done[3][NL];  // subband rows already transformed, per level
    int sr_done;
    int line0, line1;     // luma lines this cascade outputs
    void *strip;
    int16_t *coef;
    int32_t *coef32;
    vlc_stream *st;
    int *coded, *base;
    uint8_t *qi;
    uint16_t *line[3];    // NV20 staging rings
    int ll0[3], ll1[3];   // rows of the last level's LL this cascade writes (all of them, except on the GPU path)
    int gblk, gcl0, gcl1; // GPU path: the next block to launch, the rows handed over so far
    double t[3];
} casc_t;

struct vc2_dec {
    vc2_pool *pool;
    int threads;
    vc2_seq seq;
    vc2_pic pic;
    int have_seq, configured;
    int cfg_key[10];
    const vc2_wavelet *wl;
    int fin16, up, down, strip_rows, max_coefs, ref_entropy;
    comp_t pl[3];
    casc_t *cs;                // per thread
    const uint8_t *slices;     // first slice
    size_t slices_pos;         // offset of the slice data within the access unit
    size_t *off;               // slice offsets from `slices`, nslices + 1
    int idx_done, nchunks;
    int chunk_row[65];         // slice row where each thread's band starts
    float weight[64];
    double t[2];
    uint8_t *nv20_y, *nv20_c;
    ptrdiff_t nv20_pitch;
    struct dec_gpu *gpu;       // the last level on a GPU (vc2_dec_gpu.h)
    int gpu_on;
};

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline int band16(const vc2_dec *d, int l) { return d->fin16 && (l ? l : 1) == d->pic.depth; }
static inline uint64_t load_be64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return __builtin_bswap64(v); }

// --- entropy decoding -------------------------------------------------------------------------------------
// One lookup of VLC_LB bits yields every coefficient that is complete within them: the count, the bits used,
// and up to three nonzero values with their positions (unused slots repeat a used one, so the three stores
// need no branch). The values go to a linear buffer of quantised coefficients; dequantisation is a second,
// vectorised pass per subband row.
#define VLC_LB 12
typedef struct { uint8_t nbits, ncoef, pos[3]; int8_t val[3]; } vlc_entry;
static vlc_entry vlc_lut[1 << VLC_LB];

static void vlc_lut_init(void)
{
    for (int idx = 0; idx < 1 << VLC_LB; idx++) {
        vlc_entry e = { 0, 0, { 0, 0, 0 }, { 0, 0, 0 } };
        int pos = 0, nnz = 0;
#define BIT(i) ((idx >> (VLC_LB - 1 - (i))) & 1)
        while (pos < VLC_LB) {
            if (BIT(pos)) { pos++; e.ncoef++; continue; }
            int k = 1, v = 1;
            while (pos + 2 * k < VLC_LB && !BIT(pos + 2 * k)) k++;
            if (pos + 2 * k + 1 >= VLC_LB || nnz == 3) break; // incomplete, or no slot left
            for (int i = 0; i < k; i++) v = (v << 1) | BIT(pos + 2 * i + 1);
            v -= 1;
            e.pos[nnz] = e.ncoef; e.val[nnz] = BIT(pos + 2 * k + 1) ? -v : v;
            nnz++; e.ncoef++;
            pos += 2 * k + 2;
        }
#undef BIT
        e.nbits = pos;
        for (int i = nnz; i < 3; i++) { e.pos[i] = e.pos[0]; e.val[i] = e.val[0]; }
        vlc_lut[idx] = e;
    }
}

// One lookup. Bits after the end of the component read as 1 (zero coefficients). Returns 0 when the stream
// is finished: all coefficients decoded or the data used up (*coded says how many are there; -1 = a value
// does not fit 16 bits).
static inline __attribute__((always_inline)) int vlc_step(vlc_stream *s)
{
    if (s->i >= s->n || s->bp >= s->endbit) { *s->coded = s->i < s->n ? s->i : s->n; return 0; }
    uint64_t w = load_be64(s->buf + (s->bp >> 3)) << (s->bp & 7);
    if (__builtin_expect(s->bp + 64 > s->endbit, 0)) w |= ~0ull >> (s->endbit - s->bp);
    vlc_entry e;
    memcpy(&e, &vlc_lut[w >> (64 - VLC_LB)], sizeof e);
    if (__builtin_expect(e.ncoef == 0, 0)) { // a long code: bit by bit
        uint64_t m = w & 0xAAAAAAAAAAAAAAAAull;
        int k = m ? vc2_clz64(m) >> 1 : 27;
        if (k > 27) k = 27;
        uint32_t mag = 1;
        for (int j = 0; j < k; j++) mag = (mag << 1) | ((w >> (62 - 2 * j)) & 1);
        mag -= 1;
        if (mag > 32767) { *s->coded = -1; return 0; }
        s->out[s->i++] = (w >> (62 - 2 * k)) & 1 ? -(int16_t)mag : (int16_t)mag;
        s->bp += 2 * k + 2;
        return 1;
    }
    int16_t *o = s->out + s->i;
    o[e.pos[0]] = e.val[0];
    o[e.pos[1]] = e.val[1];
    o[e.pos[2]] = e.val[2];
    s->i += e.ncoef; s->bp += e.nbits;
    return 1;
}

// Decodes a list of streams, four at a time: the lookups form one dependency chain per stream, so four
// streams side by side run about as fast as the core can execute them.
static void vlc_decode_all(vlc_stream *st, int count)
{
    vlc_stream a, b, c, d;
    int next = 0;
    if (count >= 4) {
        a = st[0]; b = st[1]; c = st[2]; d = st[3];
        next = 4;
        for (;;) {
            if (!vlc_step(&a)) { if (next == count) { st[0] = b; st[1] = c; st[2] = d; count = 3; goto rest; } a = st[next++]; }
            if (!vlc_step(&b)) { if (next == count) { st[0] = a; st[1] = c; st[2] = d; count = 3; goto rest; } b = st[next++]; }
            if (!vlc_step(&c)) { if (next == count) { st[0] = a; st[1] = b; st[2] = d; count = 3; goto rest; } c = st[next++]; }
            if (!vlc_step(&d)) { if (next == count) { st[0] = a; st[1] = b; st[2] = c; count = 3; goto rest; } d = st[next++]; }
        }
    }
rest:
    for (int j = 0; j < count; j++) {
        a = st[j];
        while (vlc_step(&a)) {}
    }
}

// Reference decoder: bit-serial, 32-bit values, any coefficient size. out is zeroed, n entries.
static void vlc_decode_ref(const uint8_t *p, size_t len, int32_t *out, int n)
{
    const uint8_t *end = p + len;
    uint64_t w = 0;
    int nb = 0, i = 0;
    while (i < n) {
        if (nb < 44) { // refill to at least 57 bits; bits after the end read as 1
            ptrdiff_t rem = end - p;
            uint64_t v = rem >= 8 ? load_be64(p) : rem <= 0 ? ~0ull : load_be64(p) | (~0ull >> (rem * 8));
            w |= v >> nb;
            int adv = (64 - nb) >> 3;
            p += adv; nb += adv * 8;
        }
        uint64_t inv = ~w;
        int ones = inv ? vc2_clz64(inv) : 64;
        if (ones) {
            if (ones > nb) ones = nb;
            if (ones > n - i) ones = n - i;
            i += ones; nb -= ones;
            w = ones == 64 ? 0 : w << ones;
            continue;
        }
        uint64_t m = w & 0xAAAAAAAAAAAAAAAAull;
        int k = m ? vc2_clz64(m) >> 1 : 21;
        if (k > 21) k = 21; // not a valid stream; keep going without overrunning the window
        uint32_t mag = 1;
        for (int j = 0; j < k; j++) mag = (mag << 1) | ((w >> (62 - 2 * j)) & 1);
        mag -= 1;
        out[i++] = (w >> (62 - 2 * k)) & 1 ? -(int32_t)mag : (int32_t)mag;
        w <<= 2 * k + 2; nb -= 2 * k + 2;
    }
}

// dequantise one row of quantised values
static inline void dequant_row_16(const int16_t *v, int16_t *dst, int w, uint32_t qf, uint32_t qo)
{
    int x = 0;
    if (qf < 32768 * 4 && qo < 32768) {
        const int16_t qh = (int16_t)(qf >> 2), ql = (int16_t)(qf & 3), o = (int16_t)qo;
        for (; x + 8 <= w; x += 8) {
            vc2_v16 a = *(const vc2_v16 *)(v + x), sg = a >> 15, m = (a ^ sg) - sg;
            vc2_v16 c = m * qh + ((m * ql + o) >> 2);
            c = ((c ^ sg) - sg) & (a != 0);
            *(vc2_v16 *)(dst + x) = c;
        }
    }
    for (; x < w; x++) {
        int a = v[x], m = a < 0 ? -a : a;
        int32_t c = (int32_t)(((uint32_t)m * qf + qo) >> 2);
        dst[x] = (int16_t)(a ? (a < 0 ? -c : c) : 0);
    }
}
typedef int32_t vc2_v32 __attribute__((vector_size(16), aligned(4), may_alias));
typedef uint32_t vc2_u32 __attribute__((vector_size(16), aligned(4), may_alias));
typedef int16_t vc2_v16h __attribute__((vector_size(8), aligned(2), may_alias));
static inline void dequant_row_32(const int16_t *v, int32_t *dst, int w, uint32_t qf, uint32_t qo)
{
    int x = 0;
    for (; x + 4 <= w; x += 4) {
        vc2_v32 a = __builtin_convertvector(*(const vc2_v16h *)(v + x), vc2_v32), sg = a >> 31;
        vc2_v32 c = (vc2_v32)(((vc2_u32)((a ^ sg) - sg) * qf + qo) >> 2);
        *(vc2_v32 *)(dst + x) = ((c ^ sg) - sg) & (a != 0);
    }
    for (; x < w; x++) {
        int a = v[x], m = a < 0 ? -a : a;
        int32_t c = (int32_t)(((uint32_t)m * qf + qo) >> 2);
        dst[x] = a ? (a < 0 ? -c : c) : 0;
    }
}
static void dequant_row_ref(const int32_t *v, void *dst, int is16, int w, uint32_t qf, uint32_t qo)
{
    for (int x = 0; x < w; x++) {
        int32_t a = v[x];
        uint32_t m = a < 0 ? -(uint32_t)a : (uint32_t)a;
        int32_t c = (int32_t)((m * qf + qo) >> 2);
        c = a ? (a < 0 ? -c : c) : 0;
        if (is16) ((int16_t *)dst)[x] = (int16_t)c; else ((int32_t *)dst)[x] = c;
    }
}

static inline void *band_row(const vc2_dec *d, const casc_t *k, int c, int l, int o, int y)
{
    const comp_t *p = &d->pl[c];
    return (uint8_t *)k->band[c][l][o] + (size_t)(y & (p->brows[l] - 1)) * p->g.bw[l] * (band16(d, l) ? 2 : 4);
}

// Decode slice row sy: all its bit streams into linear buffers first, then dequantisation subband row by
// subband row, so the subband memory is written in order.
static void decode_slice_row(vc2_dec *d, casc_t *k, int sy)
{
    const int depth = d->pic.depth, nx = d->pic.slices_x, scaler = d->pic.size_scaler;
    const int per = d->max_coefs + 32; // buffer entries per slice component
    int nst = 0, ref = d->ref_entropy;
    for (int sx = 0; sx < nx; sx++) {
        const uint8_t *s = d->slices + d->off[sy * nx + sx] + d->pic.prefix_bytes;
        k->qi[sx] = *s++;
        for (int c = 0; c < 3; c++) {
            const comp_t *p = &d->pl[c];
            size_t len = (size_t)*s++ * scaler;
            int n = 0;
            for (int l = 0; l <= depth; l++) n += (l ? 3 : 1) * (p->xs[l][sx + 1] - p->xs[l][sx]) * (p->ys[l][sy + 1] - p->ys[l][sy]);
            int16_t *out = k->coef + (size_t)nst * per;
            if (!ref) memset(out, 0, (size_t)n * 2);
            k->st[nst] = (vlc_stream){ s, 0, len * 8, out, 0, n, &k->coded[nst] };
            nst++;
            s += len;
        }
    }
    if (!ref) {
        vlc_decode_all(k->st, nst);
        for (int i = 0; i < nst; i++) ref |= k->coded[i] < 0; // a coefficient beyond 16 bits (very fine quantiser)
    }
    if (ref) {
        if (!k->coef32) k->coef32 = vc2_alloc((size_t)per * 3 * nx * 4);
        for (int i = 0; i < nst; i++) {
            // vlc_decode_all may have reordered k->st: find the stream again from the slice data
            k->coded[i] = INT32_MAX;
        }
        int i = 0;
        for (int sx = 0; sx < nx; sx++) {
            const uint8_t *s = d->slices + d->off[sy * nx + sx] + d->pic.prefix_bytes + 1;
            for (int c = 0; c < 3; c++, i++) {
                const comp_t *p = &d->pl[c];
                size_t len = (size_t)*s++ * scaler;
                int n = 0;
                for (int l = 0; l <= depth; l++) n += (l ? 3 : 1) * (p->xs[l][sx + 1] - p->xs[l][sx]) * (p->ys[l][sy + 1] - p->ys[l][sy]);
                memset(k->coef32 + (size_t)i * per, 0, (size_t)n * 4);
                vlc_decode_ref(s, len, k->coef32 + (size_t)i * per, n);
                s += len;
            }
        }
    }
    double t1 = now();
    for (int c = 0; c < 3; c++) {
        const comp_t *p = &d->pl[c];
        int *base = k->base; // start of the current band in each slice's buffer
        memset(base, 0, nx * sizeof *base);
        for (int l = 0; l <= depth; l++) {
            const int *xs = p->xs[l];
            int y0 = p->ys[l][sy], h = p->ys[l][sy + 1] - y0, is16 = band16(d, l);
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
                for (int r = 0; r < h; r++) {
                    void *row = band_row(d, k, c, l, o, y0 + r);
                    for (int sx = 0; sx < nx; sx++) {
                        int w = xs[sx + 1] - xs[sx], si = sx * 3 + c, idx = base[sx] + r * w;
                        if (w <= 0) continue;
                        int q = k->qi[sx] - d->pic.qm[l][o];
                        q = q < 0 ? 0 : q >= VC2_NQ ? VC2_NQ - 1 : q;
                        if (ref) dequant_row_ref(k->coef32 + (size_t)si * per + idx, (uint8_t *)row + (size_t)xs[sx] * (is16 ? 2 : 4), is16, w, vc2_qf[q], vc2_qo[q]);
                        else if (is16) {
                            int16_t *dst = (int16_t *)row + xs[sx];
                            if (idx >= k->coded[si]) memset(dst, 0, w * 2);
                            else dequant_row_16(k->coef + (size_t)si * per + idx, dst, w, vc2_qf[q], vc2_qo[q]);
                        } else {
                            int32_t *dst = (int32_t *)row + xs[sx];
                            if (idx >= k->coded[si]) memset(dst, 0, w * 4);
                            else dequant_row_32(k->coef + (size_t)si * per + idx, dst, w, vc2_qf[q], vc2_qo[q]);
                        }
                    }
                }
                for (int sx = 0; sx < nx; sx++) base[sx] += h * (xs[sx + 1] - xs[sx]);
            }
        }
    }
    k->t[2] += now() - t1;
}

// --- output -------------------------------------------------------------------------------------------------
// 4 samples of 10 bits -> 5 bytes, sample 0 in the low bits (the NV15/NV20 packing). dst needs 3 spare bytes.
static inline void pack4x(uint8_t *dst, uint64_t x) // x = the four samples as 16-bit fields
{
    uint64_t v = (x & 0x3ff) | ((x >> 6) & (0x3ffull << 10)) | ((x >> 12) & (0x3ffull << 20)) | ((x >> 18) & (0x3ffull << 30));
    memcpy(dst, &v, 8);
}
#ifdef __aarch64__
#include <arm_neon.h>
// 8 samples -> 10 bytes (16 are stored, so dst needs 6 spare bytes)
static inline void pack8_neon(uint8_t *dst, uint16x8_t s)
{
    static const uint8_t idx[16] = { 0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 255, 255, 255, 255, 255, 255 };
    uint32x4_t x = vreinterpretq_u32_u16(s);
    x = vorrq_u32(vandq_u32(x, vdupq_n_u32(0x3ff)), vshlq_n_u32(vshrq_n_u32(x, 16), 10));   // 20 bits per pair
    uint64x2_t y = vreinterpretq_u64_u32(x);
    y = vorrq_u64(vandq_u64(y, vdupq_n_u64(0xfffff)), vshlq_n_u64(vshrq_n_u64(y, 32), 20)); // 40 bits per four
    vst1q_u8(dst, vqtbl1q_u8(vreinterpretq_u8_u64(y), vld1q_u8(idx)));
}
#endif
static void pack_y(uint8_t *dst, const uint16_t *y, int w)
{
    int x = 0;
#ifdef __aarch64__
    for (; x + 16 <= w; x += 8, dst += 10) pack8_neon(dst, vld1q_u16(y + x));
#endif
    for (; x + 8 <= w; x += 4, dst += 5) { uint64_t v; memcpy(&v, y + x, 8); pack4x(dst, v); }
    for (; x < w; x += 4, dst += 5) {
        uint64_t v = 0, o;
        memcpy(&v, y + x, (w - x < 4 ? w - x : 4) * 2);
        o = (v & 0x3ff) | ((v >> 6) & (0x3ffull << 10)) | ((v >> 12) & (0x3ffull << 20)) | ((v >> 18) & (0x3ffull << 30));
        memcpy(dst, &o, 5);
    }
}
static void pack_c(uint8_t *dst, const uint16_t *cb, const uint16_t *cr, int cw)
{
    int x = 0;
#ifdef __aarch64__
    for (; x + 16 <= cw; x += 8, dst += 20) {
        uint16x8_t u = vld1q_u16(cb + x), v = vld1q_u16(cr + x);
        pack8_neon(dst, vzip1q_u16(u, v));
        pack8_neon(dst + 10, vzip2q_u16(u, v));
    }
#endif
    for (; x + 2 <= cw; x += 2, dst += 5) {
        uint32_t a, b;
        memcpy(&a, cb + x, 4); memcpy(&b, cr + x, 4);
        uint64_t v = (uint64_t)(a & 0x3ff) | (uint64_t)(b & 0x3ff) << 10 | (uint64_t)((a >> 16) & 0x3ff) << 20 | (uint64_t)((b >> 16) & 0x3ff) << 30;
        if (x + 4 <= cw) memcpy(dst, &v, 8); else memcpy(dst, &v, 5);
    }
}

#include "vc2_dec_gpu.h"

// --- inverse wavelet --------------------------------------------------------------------------------------
static void synth_strip(vc2_dec *d, casc_t *k, int c, int l, int y0, int y1)
{
    const comp_t *p = &d->pl[c];
    const int depth = d->pic.depth, last = l == depth, nv20 = last && d->nv20_y;
    const void *in_ll = l == 1 ? k->band[c][0][0] : k->ll[c][l - 1];
    const int in_mask = (l == 1 ? p->brows[0] : p->lrows[l - 1]) - 1, bm = p->brows[l] - 1;
    int lines0 = c && d->seq.chroma == 2 ? k->line0 / 2 : k->line0, lines1 = c && d->seq.chroma == 2 ? (k->line1 + 1) / 2 : k->line1;
    if (!last) { lines0 = k->ll0[c]; lines1 = k->ll1[c]; } // a deep level: the row filter of its 16-bit output
    void *out = last ? (nv20 ? (void *)k->line[c] : (void *)p->pix) : k->ll[c][l];
    int omask = last ? (nv20 ? LINE_RING - 1 : -1) : p->lrows[l] - 1;
    ptrdiff_t ostride = last ? (nv20 ? p->g.pw : p->pstride) : 2 * p->g.bw[l];
    if (band16(d, l)) {
        vc2_level_16 L = { d->wl, p->g.bw[l], p->g.bh[l], { in_ll, k->band[c][l][1], k->band[c][l][2], k->band[c][l][3] },
                           { in_mask, bm, bm, bm }, p->g.bw[l], VC2_OUT_PIX, out, omask, ostride, p->g.w, lines0, lines1,
                           1 << (d->seq.bit_depth - 1), (1 << d->seq.bit_depth) - 1 };
        vc2_synth_strip_16(&L, y0, y1, k->strip);
    } else {
        int next16 = !last && band16(d, l + 1);
        vc2_level_32 L = { d->wl, p->g.bw[l], p->g.bh[l], { in_ll, k->band[c][l][1], k->band[c][l][2], k->band[c][l][3] },
                           { in_mask, bm, bm, bm }, p->g.bw[l], last ? VC2_OUT_PIX : next16 ? VC2_OUT_I16 : VC2_OUT_I32, out, omask,
                           ostride, p->g.w, lines0, lines1, 1 << (d->seq.bit_depth - 1), (1 << d->seq.bit_depth) - 1 };
        vc2_synth_strip_32(&L, y0, y1, k->strip);
    }
    if (!nv20 || c == 1) return;
    int a = 2 * y0 < lines0 ? lines0 : 2 * y0, b = 2 * y1 > lines1 ? lines1 : 2 * y1;
    for (int y = a; y < b; y++) {
        if (c == 0) pack_y(d->nv20_y + (size_t)y * d->nv20_pitch, k->line[0] + (size_t)(y & (LINE_RING - 1)) * p->g.pw, p->g.w);
        else pack_c(d->nv20_c + (size_t)y * d->nv20_pitch, k->line[1] + (size_t)(y & (LINE_RING - 1)) * p->g.pw,
                    k->line[2] + (size_t)(y & (LINE_RING - 1)) * p->g.pw, p->g.w);
    }
}

// rows of level l of component c that can be transformed now
static int synth_limit(const vc2_dec *d, const casc_t *k, int c, int l)
{
    const comp_t *p = &d->pl[c];
    int avail = p->ys[l][k->sr_done], in_ll = l == 1 ? p->ys[0][k->sr_done] : 2 * k->syn_done[c][l - 1];
    if (in_ll < avail) avail = in_ll;
    return k->sr_done == d->pic.slices_y || avail >= p->g.bh[l] ? avail : avail - d->down;
}

// run every level as far as the decoded slice rows allow
static void synth_advance(vc2_dec *d, casc_t *k)
{
    const int depth = d->pic.depth;
    for (int l = 1; l <= depth; l++) {
        if (l == depth && d->gpu_on) { // the GPU runs this level: only note how far it could go
            for (int c = 0; c < 3; c++) {
                int lim = synth_limit(d, k, c, l);
                if (lim > k->syn_done[c][l]) k->syn_done[c][l] = lim;
            }
            gpu_progress(d, k, 0);
            break;
        }
        const int pair = l == depth && d->nv20_y; // Cb and Cr strip by strip, for the interleaved output
        for (int c = 0; c < 3; c++) {
            const comp_t *p = &d->pl[c];
            int lim = synth_limit(d, k, c, l);
            int rows = d->strip_rows * 960 / (p->g.bw[l] > 60 ? p->g.bw[l] : 60);
            rows = rows < 2 ? 2 : rows > 16 ? 16 : rows;
            for (int y = k->syn_done[c][l]; y < lim; y += rows) {
                int y1 = y + rows < lim ? y + rows : lim;
                synth_strip(d, k, c, l, y, y1);
                if (pair && c == 1) synth_strip(d, k, 2, l, y, y1);
            }
            if (lim > k->syn_done[c][l]) k->syn_done[c][l] = lim;
            if (pair && c == 1) { k->syn_done[2][l] = k->syn_done[1][l]; break; }
        }
    }
}

static void casc_start(vc2_dec *d, casc_t *k, int sy, int line0, int line1)
{
    k->sr_done = sy; k->line0 = line0; k->line1 = line1;
    for (int c = 0; c < 3; c++) { k->ll0[c] = 0; k->ll1[c] = INT_MAX; }
    for (int c = 0; c < 3; c++)
        for (int l = 1; l <= d->pic.depth; l++) k->syn_done[c][l] = d->pl[c].ys[l][sy];
}
static int casc_lines(const vc2_dec *d, const casc_t *k)
{
    int lines = d->seq.height;
    for (int c = 0; c < 3; c++) {
        int l = 2 * k->syn_done[c][d->pic.depth];
        if (c && d->seq.chroma == 2) l *= 2;
        if (l < lines) lines = l;
    }
    return lines;
}
static void casc_row(vc2_dec *d, casc_t *k)
{
    double t0 = now();
    decode_slice_row(d, k, k->sr_done);
    k->sr_done++;
    double t1 = now();
    synth_advance(d, k);
    k->t[0] += t1 - t0; k->t[1] += now() - t1;
}

// A cascade started at slice row sy has garbage above; this is the first luma line it gets right.
static int clean_line(const vc2_dec *d, int sy)
{
    int worst = 0;
    if (sy == 0) return 0;
    for (int c = 0; c < 3; c++) {
        const comp_t *p = &d->pl[c];
        int g = p->ys[0][sy];
        for (int l = 1; l <= d->pic.depth; l++) g = 2 * ((g > p->ys[l][sy] ? g : p->ys[l][sy]) + d->up);
        if (c && d->seq.chroma == 2) g *= 2;
        if (g > worst) worst = g;
    }
    return worst;
}

static void chunk_job(void *arg, int job, int thread)
{
    vc2_dec *d = arg;
    casc_t *k = &d->cs[thread];
    const int ny = d->pic.slices_y, depth = d->pic.depth;
    int r0 = d->chunk_row[job], r1 = d->chunk_row[job + 1];
    if (r1 <= r0) return;
    int line0 = 2 * d->pl[0].ys[depth][r0], line1 = r1 == ny ? d->seq.height : 2 * d->pl[0].ys[depth][r1];
    int ra = r0;
    while (ra > 0 && clean_line(d, ra) > line0) ra--;
    casc_start(d, k, ra, line0, line1 < d->seq.height ? line1 : d->seq.height);
    if (d->gpu_on) gpu_casc_start(d, k, r0, r1);
    while (k->sr_done < ny && casc_lines(d, k) < k->line1) casc_row(d, k);
    if (d->gpu_on) gpu_progress(d, k, 1);
}

// --- setup --------------------------------------------------------------------------------------------------
static void free_all(vc2_dec *d)
{
    gpu_unconfigure(d);
    for (int c = 0; c < 3; c++) {
        comp_t *p = &d->pl[c];
        for (int l = 0; l < NL; l++) { free(p->xs[l]); free(p->ys[l]); }
        free(p->pix);
        memset(p, 0, sizeof *p);
    }
    for (int i = 0; d->cs && i < d->threads; i++) {
        casc_t *k = &d->cs[i];
        for (int c = 0; c < 3; c++) {
            for (int l = 0; l < NL; l++) {
                for (int o = 0; o < 4; o++) free(k->band[c][l][o]);
                free(k->ll[c][l]);
            }
            free(k->line[c]);
        }
        free(k->strip); free(k->coef); free(k->coef32); free(k->st); free(k->coded); free(k->base); free(k->qi);
        memset(k, 0, sizeof *k);
    }
    free(d->off); d->off = 0;
}

static int pow2ceil(int v) { int r = 1; while (r < v) r <<= 1; return r; }
static void wl_reach(const vc2_wavelet *wl, int *up, int *down)
{
    int need0[2] = { 0, 0 }, need1[2] = { 0, 0 };
    for (int j = wl->nsteps - 1; j >= 0; j--) {
        int lo, hi, t = wl->st[j].tgt;
        vc2_step_reach(&wl->st[j], &lo, &hi);
        if (need0[t] + lo < need0[!t]) need0[!t] = need0[t] + lo;
        if (need1[t] + hi > need1[!t]) need1[!t] = need1[t] + hi;
    }
    *up = -(need0[0] < need0[1] ? need0[0] : need0[1]);
    *down = need1[0] > need1[1] ? need1[0] : need1[1];
}

static int configure(vc2_dec *d)
{
    const vc2_seq *s = &d->seq;
    const vc2_pic *pc = &d->pic;
    int key[10] = { s->width, s->height, s->chroma, s->bit_depth, pc->wavelet, pc->depth, pc->slices_x, pc->slices_y, 0, 0 };
    if (d->configured && !memcmp(key, d->cfg_key, sizeof key)) return 0;
    free_all(d);
    memcpy(d->cfg_key, key, sizeof key);
    d->wl = vc2_wavelet_get(pc->wavelet);
    d->fin16 = d->wl->ok16 && s->bit_depth <= 10 && !getenv("VC2_NO16");
    wl_reach(d->wl, &d->up, &d->down);
    d->strip_rows = getenv("VC2_STRIP") ? atoi(getenv("VC2_STRIP")) : 4;
    d->ref_entropy = !!getenv("VC2_REF_ENTROPY");
    int maxw = 0;
    for (int c = 0; c < 3; c++) {
        comp_t *p = &d->pl[c];
        int w = c && s->chroma ? s->width / 2 : s->width, h = c && s->chroma == 2 ? s->height / 2 : s->height;
        vc2_geom_init(&p->g, w, h, pc->depth);
        int lag = 0;
        for (int l = 0; l <= pc->depth; l++) {
            p->xs[l] = malloc((pc->slices_x + 1) * sizeof(int));
            p->ys[l] = malloc((pc->slices_y + 1) * sizeof(int));
            int hmax = 0;
            for (int i = 0; i <= pc->slices_x; i++) p->xs[l][i] = (int)((int64_t)p->g.bw[l] * i / pc->slices_x);
            for (int i = 0; i <= pc->slices_y; i++) {
                p->ys[l][i] = (int)((int64_t)p->g.bh[l] * i / pc->slices_y);
                if (i && p->ys[l][i] - p->ys[l][i - 1] > hmax) hmax = p->ys[l][i] - p->ys[l][i - 1];
            }
            // how far the transform of a level can trail the decoded rows: its own reach plus twice the level above's
            if (l) lag = d->down + 2 * lag;
            p->brows[l] = pow2ceil(hmax + (l ? lag : d->down) + d->up + 4);
            if (l && l < pc->depth) p->lrows[l] = pow2ceil(2 * hmax + (d->down + 2 * lag) + d->up + 6);
        }
        p->pstride = (w + 31) & ~31;
        p->pix = vc2_alloc((size_t)p->pstride * h * 2);
        if (p->g.pw > maxw) maxw = p->g.pw;
    }
    d->max_coefs = 0;
    for (int l = 0; l <= pc->depth; l++)
        d->max_coefs += (l ? 3 : 1) * (d->pl[0].g.bw[l] / pc->slices_x + 1) * (d->pl[0].g.bh[l] / pc->slices_y + 1);
    for (int i = 0; i < d->threads; i++) {
        casc_t *k = &d->cs[i];
        for (int c = 0; c < 3; c++) {
            const comp_t *p = &d->pl[c];
            for (int l = 0; l <= pc->depth; l++) {
                size_t es = band16(d, l) ? 2 : 4;
                for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) k->band[c][l][o] = vc2_alloc((size_t)p->g.bw[l] * p->brows[l] * es);
                if (l && l < pc->depth) k->ll[c][l] = vc2_alloc((size_t)2 * p->g.bw[l] * p->lrows[l] * (band16(d, l + 1) ? 2 : 4));
            }
            k->line[c] = vc2_alloc((size_t)p->g.pw * LINE_RING * 2);
        }
        k->strip = vc2_alloc(vc2_strip_scratch_32(maxw / 2, 16) * 4);
        k->coef = vc2_alloc((size_t)(d->max_coefs + 32) * 3 * pc->slices_x * 2);
        k->st = malloc((size_t)3 * pc->slices_x * sizeof(vlc_stream));
        k->coded = malloc((size_t)3 * pc->slices_x * sizeof(int));
        k->base = malloc((size_t)pc->slices_x * sizeof(int));
        k->qi = malloc(pc->slices_x);
    }
    d->off = malloc(((size_t)pc->slices_x * pc->slices_y + 1) * sizeof *d->off);
    gpu_configure(d);
    d->configured = 1;
    return 0;
}

vc2_dec *vc2_dec_new(int threads, const int *cpus)
{
    vc2_tables_init();
    vlc_lut_init();
    vc2_dec *d = calloc(1, sizeof *d);
    d->threads = threads < 1 ? 1 : threads > 64 ? 64 : threads;
    d->pool = vc2_pool_new(d->threads, cpus);
    d->cs = calloc(d->threads, sizeof *d->cs);
    for (int i = 0; i < 64; i++) d->weight[i] = 1;
    return d;
}
void vc2_dec_free(vc2_dec *d)
{
    if (!d) return;
#ifdef VC2_PROF
    fprintf(stderr, "thread 0: bit streams %.3f s, dequantisation %.3f s, wavelet %.3f s\n", d->cs[0].t[0] - d->cs[0].t[2], d->cs[0].t[2], d->cs[0].t[1]);
#endif
    vc2_pool_free(d->pool);
    free_all(d);
    free(d->gpu);
    free(d->cs);
    free(d);
}
void vc2_dec_set_weights(vc2_dec *d, const float *w) { for (int i = 0; i < d->threads; i++) d->weight[i] = w[i] > 0.01f ? w[i] : 0.01f; }
const vc2_seq *vc2_dec_seq(const vc2_dec *d) { return &d->seq; }
const vc2_pic *vc2_dec_pic(const vc2_dec *d) { return &d->pic; }
void vc2_dec_times(const vc2_dec *d, double t[2]) { t[0] = d->t[0]; t[1] = d->t[1]; }
void vc2_dec_set_nv20(vc2_dec *d, uint8_t *y, uint8_t *cbcr, ptrdiff_t pitch) { d->nv20_y = y; d->nv20_c = cbcr; d->nv20_pitch = pitch; }

int vc2_dec_begin(vc2_dec *d, const uint8_t *au, size_t len, vc2_frame *out)
{
    size_t pos = 0;
    int code = -1, found = 0;
    while (pos + 13 <= len) {
        if (memcmp(au + pos, "BBCD", 4)) return -1;
        code = au[pos + 4];
        size_t next = (size_t)au[pos + 5] << 24 | au[pos + 6] << 16 | au[pos + 7] << 8 | au[pos + 8];
        if (code == VC2_PC_PICTURE_HQ) { found = 1; break; }
        if (!next || pos + next > len) return -1;
        if (code == VC2_PC_SEQ_HEADER) {
            if (vc2_parse_seq(au + pos + 13, next - 13, &d->seq)) return -1;
            d->have_seq = 1;
        }
        pos += next;
    }
    if (!found || !d->have_seq || d->seq.picture_coding_mode || d->seq.bit_depth < 8) return -1;
    if (len < pos + 13 + 8) return -1; // the picture header is short; ask for a little slack so that it is complete
    size_t avail = len - pos - 13;
    if (vc2_parse_pic(au + pos + 13, avail < 256 ? avail : 256, &d->seq, &d->pic)) return -1;
    if (d->nv20_y && (d->seq.chroma != 1 || d->seq.bit_depth != 10)) return -1;
    if (configure(d)) return -1;
    d->slices_pos = pos + 13 + d->pic.slices_offset;
    d->slices = au + d->slices_pos;
    d->off[0] = 0;
    d->idx_done = 0;
    for (int i = 0; i < d->threads; i++) d->cs[i].t[0] = d->cs[i].t[1] = d->cs[i].t[2] = 0;
    casc_start(d, &d->cs[0], 0, 0, d->seq.height);
    d->t[0] = d->t[1] = 0;
    if (out) {
        out->width = d->seq.width; out->height = d->seq.height; out->chroma = d->seq.chroma; out->bit_depth = d->seq.bit_depth;
        for (int c = 0; c < 3; c++) { out->plane[c] = d->pl[c].pix; out->stride[c] = d->pl[c].pstride; }
    }
    if (d->gpu_on && gpu_begin(d, out)) return -1;
    return 0;
}

// index the slices that are complete within the first len bytes of the access unit
static int index_slices(vc2_dec *d, size_t len)
{
    const vc2_pic *pc = &d->pic;
    const int nslices = pc->slices_x * pc->slices_y;
    size_t have = len > d->slices_pos ? len - d->slices_pos : 0;
    while (d->idx_done < nslices) {
        size_t o = d->off[d->idx_done] + pc->prefix_bytes + 1;
        int c;
        for (c = 0; c < 3 && o < have; c++) o += 1 + (size_t)d->slices[o] * pc->size_scaler;
        if (c < 3 || o > have) break;
        d->off[++d->idx_done] = o;
    }
    return d->idx_done / pc->slices_x;
}

int vc2_dec_feed(vc2_dec *d, size_t len)
{
    casc_t *k = &d->cs[0];
    int rows = index_slices(d, len);
    while (k->sr_done < rows) casc_row(d, k);
    d->t[0] = k->t[0]; d->t[1] = k->t[1];
    if (d->gpu_on) { // the lines are there when the GPU has done the whole frame
        if (casc_lines(d, k) < d->seq.height) return 0;
        gpu_progress(d, k, 1);
        if (gpu_finish(d)) return -1;
    }
    return casc_lines(d, k);
}

int vc2_dec_decode(vc2_dec *d, const uint8_t *au, size_t len, vc2_frame *out)
{
    if (vc2_dec_begin(d, au, len, out)) return -1;
    if (index_slices(d, len) != d->pic.slices_y) return -1;
    d->nchunks = d->threads < d->pic.slices_y / 4 ? d->threads : d->pic.slices_y / 4 > 0 ? d->pic.slices_y / 4 : 1;
    float wsum = 0, acc = 0;
    for (int i = 0; i < d->nchunks; i++) wsum += d->weight[i];
    for (int i = 0; i < d->nchunks; i++) { d->chunk_row[i] = (int)(d->pic.slices_y * acc / wsum + 0.5f); acc += d->weight[i]; }
    d->chunk_row[d->nchunks] = d->pic.slices_y;
    vc2_pool_run_each(d->pool, chunk_job, d, d->nchunks);
    if (d->gpu_on && gpu_finish(d)) return -1;
    for (int i = 0; i < d->threads; i++) { d->t[0] += d->cs[i].t[0]; d->t[1] += d->cs[i].t[1]; }
    return 0;
}
