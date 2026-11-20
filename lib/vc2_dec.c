// VC-2 HQ decoder: slice entropy decode + dequantisation straight into the subband buffers, then the
// inverse wavelet in strips. The deep levels run in 32 bits, the last level (3/4 of the samples) in 16 bits.
#define _GNU_SOURCE
#include "vc2_priv.h"
#include <time.h>
#ifdef VC2_PROF
#include <stdio.h>
static double prof_t[3];
#define PROF(i, t0) (prof_t[i] += now() - (t0))
#else
#define PROF(i, t0) ((void)(t0))
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

typedef struct {
    vc2_geom g;
    void *band[VC2_MAX_DEPTH + 1][4]; // [0][0] = coded LL; [l][1..3] = HL, LH, HH
    void *ll[VC2_MAX_DEPTH + 1];      // ll[l] = LL after l inverse levels (ll[0] = band[0][0])
    int *xs[VC2_MAX_DEPTH + 1], *ys[VC2_MAX_DEPTH + 1]; // slice boundaries per level
    int syn_done[VC2_MAX_DEPTH + 1];  // band rows already transformed, per level
    uint16_t *pix;
    ptrdiff_t pstride;
} plane_t;

typedef struct { int plane, level, y0, y1; } task_t;

struct vc2_dec {
    vc2_pool *pool;
    int threads;
    vc2_seq seq;
    vc2_pic pic;
    int have_seq, configured;
    int cfg_key[10];
    const vc2_wavelet *wl;
    int fin16, down, strip_rows;
    plane_t pl[3];
    void **scratch;            // per thread
    struct escratch_s *es;     // per thread: entropy decoding buffers
    int max_coefs, ref_entropy;
    const uint8_t *au;
    const uint8_t *slices;     // first slice
    size_t slices_pos;         // offset of the slice data within the access unit
    size_t *off;               // slice offsets from `slices`, nslices + 1
    int idx_done, sr_done;
    task_t *tasks;
    int ntasks, task_cap;
    int seg_sy, seg_n;         // entropy job layout of the current call
    int row0;
    double t[2];
    uint8_t *nv20_y, *nv20_c;
    ptrdiff_t nv20_pitch;
    int err;
};

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// --- entropy decoding -------------------------------------------------------------------------------------
typedef struct { const uint8_t *p, *end; uint64_t w; int n; } sbr;

static inline uint64_t load_be64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return __builtin_bswap64(v); }
// Fill the window to at least 57 bits. Bits after the end of the component read as 1 (they decode to zeros).
static inline void sbr_refill(sbr *b)
{
    ptrdiff_t rem = b->end - b->p;
    uint64_t v = rem >= 8 ? load_be64(b->p) : rem <= 0 ? ~0ull : load_be64(b->p) | (~0ull >> (rem * 8));
    b->w |= v >> b->n;
    int adv = (64 - b->n) >> 3;
    b->p += adv; b->n += adv * 8;
}
static uint8_t compact8[256]; // the 4 bits at the even positions of a byte

#define DEC_BAND(NAME, TYPE)                                                                               \
    static inline void NAME(sbr *b, TYPE *dst, ptrdiff_t stride, int w, int h, uint32_t qf, uint32_t qo)   \
    {                                                                                                      \
        int total = w * h, pos = 0, x = 0;                                                                 \
        TYPE *row = dst;                                                                                   \
        while (pos < total) {                                                                              \
            if (b->n < 44) sbr_refill(b);                                                                  \
            uint64_t v = b->w, inv = ~v;                                                                   \
            int ones = inv ? vc2_clz64(inv) : 64;                                                          \
            if (ones) {                                                                                    \
                if (ones > b->n) ones = b->n;                                                              \
                if (ones > total - pos) ones = total - pos;                                                \
                pos += ones; x += ones; b->n -= ones;                                                      \
                v = ones == 64 ? 0 : v << ones;                                                            \
                b->w = v;                                                                                  \
                if (pos >= total) break;                                                                   \
                if (b->n < 44) continue;                                                                   \
            }                                                                                              \
            uint64_t m = v & 0xAAAAAAAAAAAAAAAAull;                                                        \
            int k = m ? vc2_clz64(m) >> 1 : 21;                                                            \
            uint32_t mag;                                                                                  \
            if (__builtin_expect(k <= 4, 1)) mag = ((1u << k) | compact8[v >> (64 - 2 * k)]) - 1;          \
            else {                                                                                         \
                if (k > 21) k = 21; /* not a valid stream; keep going without overrunning the window */    \
                mag = 1;                                                                                   \
                for (int i = 0; i < k; i++) mag = (mag << 1) | ((v >> (62 - 2 * i)) & 1);                  \
                mag -= 1;                                                                                  \
            }                                                                                              \
            int neg = (v >> (62 - 2 * k)) & 1;                                                             \
            b->w = v << (2 * k + 2); b->n -= 2 * k + 2;                                                    \
            while (x >= w) { x -= w; row += stride; }                                                      \
            int32_t c = (int32_t)((mag * qf + qo) >> 2);                                                   \
            row[x] = (TYPE)(neg ? -c : c);                                                                 \
            pos++; x++;                                                                                    \
        }                                                                                                  \
    }
DEC_BAND(dec_band_32, int32_t)
DEC_BAND(dec_band_16, int16_t)

static inline int band16(const vc2_dec *d, int l) { return d->fin16 && (l ? l : 1) == d->pic.depth; }

// --- table-driven entropy decoding ---------------------------------------------------------------------------
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

// One bit stream (a slice component) being decoded into a linear buffer of quantised values.
typedef struct { const uint8_t *buf; size_t bp, endbit; int16_t *out; int i, n; int *coded; } vlc_stream;

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

// dequantise one row of quantised values into 16-bit coefficients
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
static inline void dequant_row_32(const int16_t *v, int32_t *dst, int w, uint32_t qf, uint32_t qo)
{
    for (int x = 0; x < w; x++) {
        int a = v[x], m = a < 0 ? -a : a;
        int32_t c = (int32_t)(((uint32_t)m * qf + qo) >> 2);
        dst[x] = a ? (a < 0 ? -c : c) : 0;
    }
}

typedef struct escratch_s { int16_t *coef; vlc_stream *st; int *coded, *base; uint8_t *qi; } escratch;

// reference path for one slice: bit-serial, straight into the subbands, any coefficient size
static void dec_slice_ref(vc2_dec *d, int sy, int sx)
{
    const int depth = d->pic.depth, nx = d->pic.slices_x, scaler = d->pic.size_scaler;
    const uint8_t *s = d->slices + d->off[sy * nx + sx] + d->pic.prefix_bytes;
    int qi = *s++;
    for (int c = 0; c < 3; c++) {
        plane_t *p = &d->pl[c];
        size_t len = (size_t)*s++ * scaler;
        sbr b = { s, s + len, 0, 0 };
        for (int l = 0; l <= depth; l++) {
            int x0 = p->xs[l][sx], w = p->xs[l][sx + 1] - x0, y0 = p->ys[l][sy], h = p->ys[l][sy + 1] - y0;
            ptrdiff_t st = p->g.bw[l];
            size_t esz = band16(d, l) ? 2 : 4;
            if (w <= 0 || h <= 0) continue;
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
                int q = qi - d->pic.qm[l][o];
                q = q < 0 ? 0 : q >= VC2_NQ ? VC2_NQ - 1 : q;
                for (int y = y0; y < y0 + h; y++) memset((uint8_t *)p->band[l][o] + ((size_t)y * st + x0) * esz, 0, w * esz);
                if (b.p >= b.end + 8) continue;
                if (band16(d, l)) dec_band_16(&b, (int16_t *)p->band[l][o] + (size_t)y0 * st + x0, st, w, h, vc2_qf[q], vc2_qo[q]);
                else dec_band_32(&b, (int32_t *)p->band[l][o] + (size_t)y0 * st + x0, st, w, h, vc2_qf[q], vc2_qo[q]);
            }
        }
        s += len;
    }
}

// Decode slices [sx0, sx1) of slice row sy: all their bit streams into linear buffers first, then
// dequantisation subband row by subband row, so the subband memory is written in order.
static void dec_slices(vc2_dec *d, int sy, int sx0, int sx1, escratch *es)
{
    const int depth = d->pic.depth, nx = d->pic.slices_x, scaler = d->pic.size_scaler, ns = sx1 - sx0;
    if (d->ref_entropy) {
        for (int sx = sx0; sx < sx1; sx++) dec_slice_ref(d, sy, sx);
        return;
    }
    const int per = d->max_coefs + 32; // buffer entries per slice component
    int nst = 0;
    double tp0 = now();
    for (int sx = sx0; sx < sx1; sx++) {
        const uint8_t *s = d->slices + d->off[sy * nx + sx] + d->pic.prefix_bytes;
        es->qi[sx - sx0] = *s++;
        for (int c = 0; c < 3; c++) {
            const plane_t *p = &d->pl[c];
            size_t len = (size_t)*s++ * scaler;
            int n = 0, k = (sx - sx0) * 3 + c;
            for (int l = 0; l <= depth; l++) n += (l ? 3 : 1) * (p->xs[l][sx + 1] - p->xs[l][sx]) * (p->ys[l][sy + 1] - p->ys[l][sy]);
            int16_t *out = es->coef + (size_t)k * per;
            memset(out, 0, (size_t)n * 2);
            es->st[nst++] = (vlc_stream){ s, 0, len * 8, out, 0, n, &es->coded[k] };
            s += len;
        }
    }
    PROF(0, tp0);
    double tp1 = now();
    vlc_decode_all(es->st, nst);
    PROF(1, tp1);
    double tp2 = now();
    int bad = 0;
    for (int k = 0; k < ns * 3; k++) bad |= es->coded[k] < 0;
    if (bad) { // a coefficient beyond 16 bits (very fine quantiser): take the reference path
        for (int sx = sx0; sx < sx1; sx++) dec_slice_ref(d, sy, sx);
        return;
    }
    for (int c = 0; c < 3; c++) {
        plane_t *p = &d->pl[c];
        int *base = es->base; // start of the current band in each slice's buffer
        memset(base, 0, ns * sizeof *base);
        for (int l = 0; l <= depth; l++) {
            const int *xs = p->xs[l];
            int y0 = p->ys[l][sy], h = p->ys[l][sy + 1] - y0, is16 = band16(d, l);
            ptrdiff_t st = p->g.bw[l];
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
                for (int r = 0; r < h; r++)
                    for (int sx = sx0; sx < sx1; sx++) {
                        int w = xs[sx + 1] - xs[sx], k = (sx - sx0) * 3 + c, idx = base[sx - sx0] + r * w;
                        if (w <= 0) continue;
                        int q = es->qi[sx - sx0] - d->pic.qm[l][o];
                        q = q < 0 ? 0 : q >= VC2_NQ ? VC2_NQ - 1 : q;
                        const int16_t *v = es->coef + (size_t)k * per + idx;
                        if (is16) {
                            int16_t *dst = (int16_t *)p->band[l][o] + (size_t)(y0 + r) * st + xs[sx];
                            if (idx >= es->coded[k]) memset(dst, 0, w * 2);
                            else dequant_row_16(v, dst, w, vc2_qf[q], vc2_qo[q]);
                        } else {
                            int32_t *dst = (int32_t *)p->band[l][o] + (size_t)(y0 + r) * st + xs[sx];
                            if (idx >= es->coded[k]) memset(dst, 0, w * 4);
                            else dequant_row_32(v, dst, w, vc2_qf[q], vc2_qo[q]);
                        }
                    }
                for (int sx = sx0; sx < sx1; sx++) base[sx - sx0] += h * (xs[sx + 1] - xs[sx]);
            }
        }
    }
    PROF(2, tp2);
}

static void entropy_job(void *arg, int job, int thread)
{
    vc2_dec *d = arg;
    const int nx = d->pic.slices_x, seg = d->seg_n;
    int sy = d->seg_sy + job / seg, i = job % seg;
    dec_slices(d, sy, nx * i / seg, nx * (i + 1) / seg, &d->es[thread]);
}

// --- inverse wavelet --------------------------------------------------------------------------------------
static void synth_job(void *arg, int job, int thread)
{
    vc2_dec *d = arg;
    const task_t *t = &d->tasks[job];
    plane_t *p = &d->pl[t->plane];
    const int l = t->level, depth = d->pic.depth, last = l == depth;
    const void *in_ll = l == 1 ? p->band[0][0] : p->ll[l - 1];
    if (band16(d, l)) {
        vc2_level_16 L = { d->wl, p->g.bw[l], p->g.bh[l], { in_ll, p->band[l][1], p->band[l][2], p->band[l][3] }, p->g.bw[l],
                           VC2_OUT_PIX, p->pix, p->pstride, p->g.w, p->g.h, 1 << (d->seq.bit_depth - 1), (1 << d->seq.bit_depth) - 1, 0 };
        vc2_synth_strip_16(&L, t->y0, t->y1, d->scratch[thread]);
    } else {
        int next16 = !last && band16(d, l + 1);
        vc2_level_32 L = { d->wl, p->g.bw[l], p->g.bh[l], { in_ll, p->band[l][1], p->band[l][2], p->band[l][3] }, p->g.bw[l],
                           last ? VC2_OUT_PIX : next16 ? VC2_OUT_I16 : VC2_OUT_I32, last ? (void *)p->pix : p->ll[l],
                           last ? p->pstride : 2 * p->g.bw[l], p->g.w, p->g.h, 1 << (d->seq.bit_depth - 1),
                           (1 << d->seq.bit_depth) - 1, 0 };
        vc2_synth_strip_32(&L, t->y0, t->y1, d->scratch[thread]);
    }
}

static int wl_down(const vc2_wavelet *wl)
{
    int need1[2] = { 0, 0 };
    for (int j = wl->nsteps - 1; j >= 0; j--) {
        int lo, hi, t = wl->st[j].tgt;
        vc2_step_reach(&wl->st[j], &lo, &hi);
        if (need1[t] + hi > need1[!t]) need1[!t] = need1[t] + hi;
    }
    return need1[0] > need1[1] ? need1[0] : need1[1];
}

static void add_task(vc2_dec *d, int plane, int level, int y0, int y1)
{
    if (d->ntasks == d->task_cap) {
        d->task_cap = d->task_cap ? d->task_cap * 2 : 256;
        d->tasks = realloc(d->tasks, d->task_cap * sizeof *d->tasks);
    }
    d->tasks[d->ntasks++] = (task_t){ plane, level, y0, y1 };
}

// run every level as far as the decoded slice rows allow
static void synth_advance(vc2_dec *d)
{
    const int depth = d->pic.depth, all = d->sr_done == d->pic.slices_y;
    for (int l = 1; l <= depth; l++) {
        d->ntasks = 0;
        size_t work = 0;
        for (int c = 0; c < 3; c++) {
            plane_t *p = &d->pl[c];
            int avail = p->ys[l][d->sr_done];
            int in_ll = l == 1 ? p->ys[0][d->sr_done] : 2 * p->syn_done[l - 1];
            if (in_ll < avail) avail = in_ll;
            int lim = all || avail >= p->g.bh[l] ? avail : avail - d->down;
            int rows = d->strip_rows * 960 / (p->g.bw[l] > 60 ? p->g.bw[l] : 60);
            rows = rows < 2 ? 2 : rows > 16 ? 16 : rows;
            for (int y = p->syn_done[l]; y < lim; y += rows) add_task(d, c, l, y, y + rows < lim ? y + rows : lim);
            if (lim > p->syn_done[l]) { work += (size_t)(lim - p->syn_done[l]) * p->g.bw[l]; p->syn_done[l] = lim; }
        }
        if (work < 8192) for (int j = 0; j < d->ntasks; j++) synth_job(d, j, 0);
        else vc2_pool_run(d->pool, synth_job, d, d->ntasks);
    }
}

// --- setup --------------------------------------------------------------------------------------------------
static void free_planes(vc2_dec *d)
{
    for (int c = 0; c < 3; c++) {
        plane_t *p = &d->pl[c];
        for (int l = 0; l <= VC2_MAX_DEPTH; l++) {
            for (int o = 0; o < 4; o++) free(p->band[l][o]);
            if (l) free(p->ll[l]);
            free(p->xs[l]); free(p->ys[l]);
        }
        free(p->pix);
        memset(p, 0, sizeof *p);
    }
    for (int i = 0; d->scratch && i < d->threads; i++) { free(d->scratch[i]); d->scratch[i] = 0; free(d->es[i].coef); free(d->es[i].st); free(d->es[i].coded); free(d->es[i].base); free(d->es[i].qi); memset(&d->es[i], 0, sizeof d->es[i]); }
    free(d->off); d->off = 0;
}

static int configure(vc2_dec *d)
{
    const vc2_seq *s = &d->seq;
    const vc2_pic *pc = &d->pic;
    int key[10] = { s->width, s->height, s->chroma, s->bit_depth, pc->wavelet, pc->depth, pc->slices_x, pc->slices_y, 0, 0 };
    if (d->configured && !memcmp(key, d->cfg_key, sizeof key)) return 0;
    free_planes(d);
    memcpy(d->cfg_key, key, sizeof key);
    d->wl = vc2_wavelet_get(pc->wavelet);
    d->fin16 = d->wl->ok16 && s->bit_depth <= 10 && !getenv("VC2_NO16");
    d->down = wl_down(d->wl);
    d->strip_rows = getenv("VC2_STRIP") ? atoi(getenv("VC2_STRIP")) : 8;
    int maxw = 0;
    for (int c = 0; c < 3; c++) {
        plane_t *p = &d->pl[c];
        int w = c && s->chroma ? s->width / 2 : s->width, h = c && s->chroma == 2 ? s->height / 2 : s->height;
        vc2_geom_init(&p->g, w, h, pc->depth);
        if (p->g.pw % pc->slices_x && p->g.bw[0] < pc->slices_x) return -1;
        for (int l = 0; l <= pc->depth; l++) {
            size_t n = (size_t)p->g.bw[l] * p->g.bh[l] * (band16(d, l) ? 2 : 4);
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) p->band[l][o] = vc2_alloc(n);
            if (l && l < pc->depth) p->ll[l] = vc2_alloc((size_t)4 * p->g.bw[l] * p->g.bh[l] * (band16(d, l + 1) ? 2 : 4));
            p->xs[l] = malloc((pc->slices_x + 1) * sizeof(int));
            p->ys[l] = malloc((pc->slices_y + 1) * sizeof(int));
            for (int i = 0; i <= pc->slices_x; i++) p->xs[l][i] = (int)((int64_t)p->g.bw[l] * i / pc->slices_x);
            for (int i = 0; i <= pc->slices_y; i++) p->ys[l][i] = (int)((int64_t)p->g.bh[l] * i / pc->slices_y);
        }
        p->pstride = (w + 31) & ~31;
        p->pix = vc2_alloc((size_t)p->pstride * h * 2);
        if (p->g.pw > maxw) maxw = p->g.pw;
    }
    d->max_coefs = 0;
    for (int l = 0; l <= pc->depth; l++)
        d->max_coefs += (l ? 3 : 1) * (d->pl[0].g.bw[l] / pc->slices_x + 1) * (d->pl[0].g.bh[l] / pc->slices_y + 1);
    d->ref_entropy = !!getenv("VC2_REF_ENTROPY");
    for (int i = 0; i < d->threads; i++) {
        d->scratch[i] = vc2_alloc(vc2_strip_scratch_32(maxw / 2, 16) * 4);
        d->es[i].coef = vc2_alloc((size_t)(d->max_coefs + 32) * 3 * pc->slices_x * 2);
        d->es[i].st = malloc((size_t)3 * pc->slices_x * sizeof(vlc_stream));
        d->es[i].coded = malloc((size_t)3 * pc->slices_x * sizeof(int));
        d->es[i].base = malloc((size_t)pc->slices_x * sizeof(int));
        d->es[i].qi = malloc(pc->slices_x);
    }
    d->off = malloc(((size_t)pc->slices_x * pc->slices_y + 1) * sizeof *d->off);
    d->configured = 1;
    return 0;
}

vc2_dec *vc2_dec_new(int threads, const int *cpus)
{
    vc2_tables_init();
    for (int i = 0; i < 256; i++) compact8[i] = ((i >> 6) & 1) << 3 | ((i >> 4) & 1) << 2 | ((i >> 2) & 1) << 1 | (i & 1);
    vc2_dec *d = calloc(1, sizeof *d);
    d->threads = threads < 1 ? 1 : threads > 64 ? 64 : threads;
    d->pool = vc2_pool_new(d->threads, cpus);
    d->scratch = calloc(d->threads, sizeof *d->scratch);
    d->es = calloc(d->threads, sizeof *d->es);
    vlc_lut_init();
    return d;
}
void vc2_dec_free(vc2_dec *d)
{
    if (!d) return;
#ifdef VC2_PROF
    fprintf(stderr, "entropy: setup %.3f s, vlc %.3f s, dequant %.3f s\n", prof_t[0], prof_t[1], prof_t[2]);
#endif
    vc2_pool_free(d->pool);
    free_planes(d);
    free(d->scratch); free(d->es); free(d->tasks);
    free(d);
}
const vc2_seq *vc2_dec_seq(const vc2_dec *d) { return &d->seq; }
const vc2_pic *vc2_dec_pic(const vc2_dec *d) { return &d->pic; }
void vc2_dec_times(const vc2_dec *d, double t[2]) { t[0] = d->t[0]; t[1] = d->t[1]; }
void vc2_dec_set_nv20(vc2_dec *d, uint8_t *y, uint8_t *cbcr, ptrdiff_t pitch) { d->nv20_y = y; d->nv20_c = cbcr; d->nv20_pitch = pitch; }

int vc2_dec_begin(vc2_dec *d, const uint8_t *au, size_t len, vc2_frame *out)
{
    size_t pos = 0;
    int code = -1, found = 0;
    d->err = 0;
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
    // the picture header is short; ask for a little slack so that it is complete
    if (len < pos + 13 + 8) return -1;
    size_t avail = len - pos - 13;
    if (vc2_parse_pic(au + pos + 13, avail < 256 ? avail : 256, &d->seq, &d->pic)) return -1;
    if (configure(d)) return -1;
    d->au = au;
    d->slices_pos = pos + 13 + d->pic.slices_offset;
    d->slices = au + d->slices_pos;
    d->off[0] = 0;
    d->idx_done = 0; d->sr_done = 0;
    for (int c = 0; c < 3; c++) memset(d->pl[c].syn_done, 0, sizeof d->pl[c].syn_done);
    d->t[0] = d->t[1] = 0;
    if (out) {
        out->width = d->seq.width; out->height = d->seq.height; out->chroma = d->seq.chroma; out->bit_depth = d->seq.bit_depth;
        for (int c = 0; c < 3; c++) { out->plane[c] = d->pl[c].pix; out->stride[c] = d->pl[c].pstride; }
    }
    return 0;
}

int vc2_dec_feed(vc2_dec *d, size_t len)
{
    const vc2_pic *pc = &d->pic;
    const int nslices = pc->slices_x * pc->slices_y;
    size_t have = len > d->slices_pos ? len - d->slices_pos : 0;
    while (d->idx_done < nslices) { // index the slices that are complete
        size_t o = d->off[d->idx_done] + pc->prefix_bytes + 1;
        int c;
        for (c = 0; c < 3 && o < have; c++) o += 1 + (size_t)d->slices[o] * pc->size_scaler;
        if (c < 3 || o > have) break;
        d->off[++d->idx_done] = o;
    }
    int rows = d->idx_done / pc->slices_x;
    if (rows > d->sr_done) {
        double t0 = now();
        int n = rows - d->sr_done;
        d->seg_sy = d->sr_done;
        d->seg_n = n >= d->threads ? 1 : pc->slices_x < d->threads ? pc->slices_x : d->threads;
        vc2_pool_run(d->pool, entropy_job, d, n * d->seg_n);
        d->sr_done = rows;
        double t1 = now();
        synth_advance(d);
        d->t[0] += t1 - t0; d->t[1] += now() - t1;
    }
    int lines = d->seq.height;
    for (int c = 0; c < 3; c++) {
        int l = 2 * d->pl[c].syn_done[pc->depth];
        if (c && d->seq.chroma == 2) l *= 2;
        if (l < lines) lines = l;
    }
    return lines;
}

int vc2_dec_decode(vc2_dec *d, const uint8_t *au, size_t len, vc2_frame *out)
{
    if (vc2_dec_begin(d, au, len, out)) return -1;
    return vc2_dec_feed(d, len) == d->seq.height ? 0 : -1;
}
