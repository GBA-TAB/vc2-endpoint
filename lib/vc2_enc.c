// VC-2 HQ encoder: forward wavelet, rate control with one quantiser index per rate-control unit (the whole
// frame by default) plus per-slice refinement, tail truncation, slice writer.
#define _GNU_SOURCE
#include "vc2_priv.h"
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int n_eval;
static void vlc_init(void);

#define T int32_t
#define SUF _32
#include "vc2_lift.h"
#undef T
#undef SUF

typedef struct {
    vc2_geom g;
    int32_t *img;                        // padded input, offset removed
    int32_t *band[VC2_MAX_DEPTH + 1][4];
    int32_t *ll[VC2_MAX_DEPTH + 1];
    int *xs[VC2_MAX_DEPTH + 1], *ys[VC2_MAX_DEPTH + 1];
} eplane;

typedef struct { const int32_t *p; int level; float r16; } rowref; // one band row of a slice row
typedef struct { rowref *rows; uint32_t *bits, *dd; int32_t *tmp, *gather; uint8_t *obuf; } escratch;

struct vc2_enc {
    vc2_enc_cfg c;
    vc2_pool *pool;
    const vc2_wavelet *wl;
    vc2_seq seq;
    uint8_t qm[VC2_MAX_DEPTH + 1][4];
    eplane pl[3];
    int nx, ny, nslices, max_rows;
    float r16[VC2_NQ];
    escratch *scr;
    // rate control
    uint16_t *cb[VC2_NQ];       // per quantiser: component bytes per slice (3 each), allocated on demand
    uint16_t *cut[VC2_NQ];      // ... and the rows kept per component (coding order)
    uint32_t *valid[VC2_NQ];    // frame number + 1 per slice row when cb[q] is current
    uint32_t frame;
    uint8_t *sq;                // chosen quantiser per slice
    size_t *soff;               // slice offsets in the output
    int last_q, scaler;
    uint32_t prev_unit;         // size of the last data unit written (parse info chain)
    long q_sum;
    // per-call
    const uint16_t *const *in;
    const ptrdiff_t *in_stride;
    int eval_q, eval_r0;
    const int *eval_rows;       // optional list of slice rows to count
    uint8_t *out_slices;
};

void vc2_enc_defaults(vc2_enc_cfg *c, int width, int height)
{
    memset(c, 0, sizeof *c);
    c->width = width; c->height = height; c->chroma = 1; c->bit_depth = 10;
    c->fr_num = 50; c->fr_den = 1;
    c->wavelet = VC2_WL_LEGALL5_3; c->depth = 4;
    c->slice_w = 128; c->slice_h = 16;
    c->bpp = 3; c->tail_lambda = 0.2; c->threads = 1;
}

vc2_enc *vc2_enc_new(const vc2_enc_cfg *c)
{
    vc2_tables_init();
    vlc_init();
    vc2_enc *e = calloc(1, sizeof *e);
    e->c = *c;
    e->wl = vc2_wavelet_get(c->wavelet);
    if (!e->wl || c->depth < 1 || c->depth > VC2_MAX_DEPTH) { free(e); return 0; }
    if (c->qm_custom) memcpy(e->qm, c->qm, sizeof e->qm);
    else if (vc2_default_qm(c->wavelet, c->depth > 4 ? 4 : c->depth, e->qm)) { free(e); return 0; }
    for (int l = 5; !c->qm_custom && l <= c->depth; l++) // deeper than the default tables: continue the progression
        for (int o = 1; o < 4; o++) e->qm[l][o] = e->qm[4][o] + (e->qm[4][o] - e->qm[3][o]) * (l - 4);
    e->seq = (vc2_seq){ .major = 2, .profile = 3, .width = c->width, .height = c->height, .chroma = c->chroma,
                        .fr_num = c->fr_num, .fr_den = c->fr_den, .bit_depth = c->bit_depth };
    e->pool = vc2_pool_new(c->threads, c->cpus);
    int th = vc2_pool_threads(e->pool);
    for (int i = 0; i < 3; i++) {
        eplane *p = &e->pl[i];
        int w = i && c->chroma ? c->width / 2 : c->width, h = i && c->chroma == 2 ? c->height / 2 : c->height;
        vc2_geom_init(&p->g, w, h, c->depth);
    }
    e->nx = e->pl[0].g.pw / c->slice_w; e->ny = e->pl[0].g.ph / c->slice_h;
    if (e->nx < 1) e->nx = 1;
    if (e->ny < 1) e->ny = 1;
    e->nslices = e->nx * e->ny;
    for (int i = 0; i < 3; i++) {
        eplane *p = &e->pl[i];
        p->img = vc2_alloc((size_t)p->g.pw * p->g.ph * 4);
        for (int l = 0; l <= c->depth; l++) {
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) p->band[l][o] = vc2_alloc((size_t)p->g.bw[l] * p->g.bh[l] * 4);
            if (l && l < c->depth) p->ll[l] = vc2_alloc((size_t)16 * p->g.bw[l] * p->g.bh[l]);
            p->xs[l] = malloc((e->nx + 1) * sizeof(int));
            p->ys[l] = malloc((e->ny + 1) * sizeof(int));
            for (int k = 0; k <= e->nx; k++) p->xs[l][k] = (int)((int64_t)p->g.bw[l] * k / e->nx);
            for (int k = 0; k <= e->ny; k++) p->ys[l][k] = (int)((int64_t)p->g.bh[l] * k / e->ny);
        }
    }
    // rows of one slice component in coding order, worst case
    const vc2_geom *g = &e->pl[0].g;
    e->max_rows = 0;
    for (int l = 0; l <= c->depth; l++) e->max_rows += (l ? 3 : 1) * (g->bh[l] / e->ny + 2);
    size_t gather_n = 0; // quantised values of one slice row, see write_job
    for (int i = 0; i < 3; i++)
        for (int l = 0; l <= c->depth; l++)
            gather_n += (size_t)(l ? 3 : 1) * (e->pl[i].g.bh[l] / e->ny + 1) * ((e->pl[i].g.bw[l] + e->nx - 1) / e->nx + 1) * e->nx;
    e->scr = calloc(th, sizeof *e->scr);
    for (int i = 0; i < th; i++) {
        e->scr[i].rows = malloc(e->max_rows * sizeof(rowref));
        e->scr[i].bits = malloc((size_t)e->max_rows * e->nx * sizeof(uint32_t));
        e->scr[i].dd = malloc((size_t)e->max_rows * e->nx * sizeof(uint32_t));
        e->scr[i].gather = vc2_alloc((gather_n + 64) * 4);
        e->scr[i].tmp = vc2_alloc((size_t)(g->pw + 64) * 2 * 4);
        e->scr[i].obuf = malloc(4 + 3 * 65536 + 64);
    }
    for (int q = 0; q < VC2_NQ; q++) e->r16[q] = (float)(16.0 / vc2_qf[q] * (1.0 + 1.0 / (1 << 20)));
    e->sq = malloc(e->nslices);
    e->soff = malloc((e->nslices + 1) * sizeof *e->soff);
    e->last_q = 24; e->scaler = 1;
    return e;
}

void vc2_enc_free(vc2_enc *e)
{
    if (!e) return;
    int th = vc2_pool_threads(e->pool);
    vc2_pool_free(e->pool);
    for (int i = 0; i < 3; i++) {
        eplane *p = &e->pl[i];
        free(p->img);
        for (int l = 0; l <= VC2_MAX_DEPTH; l++) {
            for (int o = 0; o < 4; o++) free(p->band[l][o]);
            free(p->ll[l]); free(p->xs[l]); free(p->ys[l]);
        }
    }
    for (int i = 0; i < th; i++) { free(e->scr[i].rows); free(e->scr[i].bits); free(e->scr[i].dd); free(e->scr[i].tmp); free(e->scr[i].gather); free(e->scr[i].obuf); }
    for (int q = 0; q < VC2_NQ; q++) { free(e->cb[q]); free(e->cut[q]); free(e->valid[q]); }
    free(e->scr); free(e->sq); free(e->soff);
    free(e);
}

size_t vc2_enc_max_size(const vc2_enc *e)
{
    return (size_t)(e->c.width * (double)e->c.height * (e->c.fixed_q > 0 ? 40 : e->c.bpp) / 8) + 4096 + VC2_INPUT_PADDING;
}
int vc2_enc_last_q(const vc2_enc *e) { return (int)(e->q_sum * 100 / e->nslices); }

// --- forward wavelet ---------------------------------------------------------------------------------------
static void analyse_level(const vc2_wavelet *wl, const int32_t *in, ptrdiff_t istride, int w, int h, int32_t *bands[4], int32_t *tmp)
{
    const int ns = wl->nsteps, sh = wl->shift, hw = (w + 2 * VC2_G + 7) & ~7;
    int32_t *lo = tmp + VC2_G, *hi = tmp + hw + VC2_G;
    for (int y = 0; y < 2 * h; y++) {
        const int32_t *r = in + (size_t)y * istride;
        for (int x = 0; x < w; x++) { lo[x] = r[2 * x] << sh; hi[x] = r[2 * x + 1] << sh; }
        for (int j = ns - 1; j >= 0; j--) vc2_lift_h_32(lo, hi, w, &wl->st[j], -wl->st[j].sign);
        memcpy(bands[(y & 1) * 2] + (size_t)(y / 2) * w, lo, w * 4);
        memcpy(bands[(y & 1) * 2 + 1] + (size_t)(y / 2) * w, hi, w * 4);
    }
#define CLAMP(i) ((i) < 0 ? 0 : (i) >= h ? h - 1 : (i))
    for (int j = ns - 1; j >= 0; j--) {
        const vc2_step *s = &wl->st[j];
        int t = s->tgt, off = t ? -1 : -2;
        for (int half = 0; half < 2; half++) {
            int32_t *tb = bands[t * 2 + half];
            const int32_t *ob = bands[!t * 2 + half];
            for (int i = 0; i < h; i++)
                vc2_lift_32(tb + (size_t)i * w, tb + (size_t)i * w, ob + (size_t)CLAMP(i + off) * w, ob + (size_t)CLAMP(i + off + 1) * w,
                            ob + (size_t)CLAMP(i + off + 2) * w, ob + (size_t)CLAMP(i + off + 3) * w, w, s, -s->sign);
        }
    }
#undef CLAMP
}

static void plane_job(void *arg, int job, int thread)
{
    vc2_enc *e = arg;
    eplane *p = &e->pl[job];
    const int w = p->g.w, h = p->g.h, pw = p->g.pw, ph = p->g.ph, off = 1 << (e->c.bit_depth - 1), depth = e->c.depth;
    for (int y = 0; y < ph; y++) {
        const uint16_t *s = e->in[job] + (size_t)(y < h ? y : h - 1) * e->in_stride[job];
        int32_t *d = p->img + (size_t)y * pw;
        for (int x = 0; x < w; x++) d[x] = (int32_t)s[x] - off;
        for (int x = w; x < pw; x++) d[x] = d[w - 1];
    }
    for (int l = depth; l >= 1; l--) {
        const int32_t *in = l == depth ? p->img : p->ll[l];
        int32_t *bands[4] = { l == 1 ? p->band[0][0] : p->ll[l - 1], p->band[l][1], p->band[l][2], p->band[l][3] };
        analyse_level(e->wl, in, 2 * p->g.bw[l], p->g.bw[l], p->g.bh[l], bands, e->scr[thread].tmp);
    }
}

// --- counting -----------------------------------------------------------------------------------------------
static inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// Bits, truncation cost (in 1/16 of a squared quantiser step) and "any nonzero" of one coefficient row.
static inline __attribute__((always_inline)) void count_row_w(const int32_t *c, int w, float r16, uint32_t *bits, uint32_t *dd, int *nz)
{
    uint32_t b = 0, d = 0, any = 0;
    for (int x = 0; x < w; x++) {
        int32_t n = (c[x] < 0 ? -c[x] : c[x]) << 2;
        int32_t u16 = (int32_t)((float)n * r16), v = u16 >> 4;
        uint32_t ex = (fbits((float)(v + 1)) >> 23) - 127;
        int32_t vv = v > 255 ? 255 : v, uu = u16 > 4096 ? 4096 : u16;
        b += 2 * ex + 1 + (v != 0);
        d += v ? (uint32_t)((2 * vv + 1) * (uu - 8 * vv - 4)) : 0;
        any |= (uint32_t)v;
    }
    *bits = b; *dd = d; *nz = any != 0;
}

// Slice rows are short; with the width known at compile time the loop becomes straight vector code.
static void count_row(const int32_t *c, int w, float r16, uint32_t *bits, uint32_t *dd, int *nz)
{
    switch (w) {
    case 8: count_row_w(c, 8, r16, bits, dd, nz); break;
    case 16: count_row_w(c, 16, r16, bits, dd, nz); break;
    case 32: count_row_w(c, 32, r16, bits, dd, nz); break;
    case 64: count_row_w(c, 64, r16, bits, dd, nz); break;
    default: count_row_w(c, w, r16, bits, dd, nz);
    }
}
// quantised signed values of one row
static inline __attribute__((always_inline)) int quant_row_w(const int32_t *c, int w, float r16, int32_t *v)
{
    int32_t any = 0;
    for (int x = 0; x < w; x++) {
        int32_t a = c[x] < 0 ? -c[x] : c[x], q = (int32_t)((float)(a << 2) * r16) >> 4;
        v[x] = c[x] < 0 ? -q : q;
        any |= q;
    }
    return any != 0;
}
// returns 0 if the whole row quantises to zero
static int quant_row(const int32_t *c, int w, float r16, int32_t *v)
{
    switch (w) {
    case 8: return quant_row_w(c, 8, r16, v);
    case 16: return quant_row_w(c, 16, r16, v);
    case 32: return quant_row_w(c, 32, r16, v);
    case 64: return quant_row_w(c, 64, r16, v);
    default: return quant_row_w(c, w, r16, v);
    }
}

// Sizes of the slices of one slice row at quantiser index qi: bytes per component, and the rows kept (coding
// order). Works band row by band row, so the coefficient memory is read in order.
static void count_slice_row(vc2_enc *e, escratch *s, int sy, int qi, uint16_t *bytes, uint16_t *cuts)
{
    const int depth = e->c.depth, nx = e->nx, explicit_zeros = e->c.no_truncate;
    const float lam = (float)(e->c.tail_lambda * 16);
    for (int c = 0; c < 3; c++) {
        const eplane *p = &e->pl[c];
        int n = 0;
        for (int l = 0; l <= depth; l++)
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
                int q = qi - e->qm[l][o];
                float r16 = e->r16[q < 0 ? 0 : q];
                for (int y = p->ys[l][sy]; y < p->ys[l][sy + 1]; y++, n++) {
                    const int32_t *row = p->band[l][o] + (size_t)y * p->g.bw[l];
                    const int *xs = p->xs[l];
                    int nz;
                    s->rows[n] = (rowref){ row, l, r16 };
                    for (int sx = 0; sx < nx; sx++)
                        count_row(row + xs[sx], xs[sx + 1] - xs[sx], r16, &s->bits[n * nx + sx], &s->dd[n * nx + sx], &nz);
                }
            }
        for (int sx = 0; sx < nx; sx++) {
            // best truncation point: the tail whose bits are worth more than the distortion of dropping it
            int best = n;
            float accr = 0, accd = 0, bestg = 0;
            for (int j = n - 1; j >= 0; j--) {
                int w = p->xs[s->rows[j].level][sx + 1] - p->xs[s->rows[j].level][sx];
                accr += s->bits[j * nx + sx] - (explicit_zeros ? w : 0); accd += s->dd[j * nx + sx];
                float g = lam * accr - accd;
                if (g >= bestg) { bestg = g; best = j; }
            }
            uint32_t bits = 0;
            for (int j = 0; j < best; j++) bits += s->bits[j * nx + sx];
            for (int j = best; explicit_zeros && j < n; j++) bits += p->xs[s->rows[j].level][sx + 1] - p->xs[s->rows[j].level][sx];
            if (best > 0 && !explicit_zeros) { // zeros at the end of the last kept row need no bits either
                const rowref *r = &s->rows[best - 1];
                int x0 = p->xs[r->level][sx], w = p->xs[r->level][sx + 1] - x0, x = w;
                while (x > 0) {
                    int32_t v = r->p[x0 + x - 1];
                    v = (int32_t)((float)((v < 0 ? -v : v) << 2) * r->r16) >> 4;
                    if (v) break;
                    x--;
                }
                bits -= w - x;
            }
            // never an empty component: ffmpeg's decoder (a common receiver) reads one byte of a zero-length one
            bytes[sx * 3 + c] = bits ? (bits + 7) >> 3 : 1;
            cuts[sx * 3 + c] = best;
        }
    }
}

static void count_job(void *arg, int job, int thread)
{
    vc2_enc *e = arg;
    int sy = e->eval_rows ? e->eval_rows[job] : e->eval_r0 + job, q = e->eval_q;
    if (e->valid[q][sy] == e->frame + 1) return;
    count_slice_row(e, &e->scr[thread], sy, q, e->cb[q] + (size_t)sy * e->nx * 3, e->cut[q] + (size_t)sy * e->nx * 3);
    e->valid[q][sy] = e->frame + 1;
}

static void eval_q(vc2_enc *e, int q, int r0, int r1, const int *rows, int nrows)
{
    if (!e->cb[q]) { e->cb[q] = malloc((size_t)e->nslices * 3 * 2); e->cut[q] = malloc((size_t)e->nslices * 3 * 2); e->valid[q] = calloc(e->ny, 4); }
    e->eval_q = q; e->eval_r0 = r0; e->eval_rows = rows;
    if (e->valid[q][rows ? rows[0] : r0] != e->frame + 1) n_eval++;
    vc2_pool_run(e->pool, count_job, e, rows ? nrows : r1 - r0);
}

static inline size_t slice_size(const uint16_t *cb, int s)
{
    return 4 + (size_t)((cb[0] + s - 1) / s + (cb[1] + s - 1) / s + (cb[2] + s - 1) / s) * s;
}
static size_t total_size(vc2_enc *e, int q, int r0, int r1, int s, int *maxcb)
{
    size_t t = 0;
    int m = 0;
    const uint16_t *cb = e->cb[q] + (size_t)r0 * e->nx * 3;
    for (int i = 0; i < (r1 - r0) * e->nx; i++, cb += 3) {
        t += slice_size(cb, s);
        if (cb[0] > m) m = cb[0];
        if (cb[1] > m) m = cb[1];
        if (cb[2] > m) m = cb[2];
    }
    if (maxcb) *maxcb = m;
    return t;
}

// does quantiser q fit the unit's budget? Sets *sc (the size scaler it needs) and *tot.
static int fits(vc2_enc *e, int q, int r0, int r1, size_t budget, int *sc, size_t *tot)
{
    int m;
    eval_q(e, q, r0, r1, 0, 0);
    total_size(e, q, r0, r1, 1, &m);
    int s = (m + 254) / 255;
    if (s < e->scaler) s = e->scaler;
    *sc = s;
    *tot = total_size(e, q, r0, r1, s, 0);
    return *tot <= budget;
}

static int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

// --- slice writer -------------------------------------------------------------------------------------------
// Bit writer: acc holds n (< 8 after each put) pending bits at its top; every put stores 8 bytes, so the
// target needs 8 spare bytes after the end.
typedef struct { uint8_t *p; uint64_t acc; int n; } sbw;
static inline void put(sbw *b, uint64_t v, int len) // len <= 56
{
    b->acc |= v << (64 - len - b->n);
    uint64_t be = __builtin_bswap64(b->acc);
    memcpy(b->p, &be, 8);
    int n = b->n + len;
    b->p += n >> 3;
    b->acc <<= n & ~7;
    b->n = n & 7;
}
static inline uint32_t spread16(uint32_t x)
{
    x = (x | x << 8) & 0x00ff00ff; x = (x | x << 4) & 0x0f0f0f0f; x = (x | x << 2) & 0x33333333; x = (x | x << 1) & 0x55555555;
    return x;
}
#define VLC_R 64
static uint16_t vlc_code[2 * VLC_R];
static uint8_t vlc_len[2 * VLC_R];
static void vlc_init(void)
{
    for (int i = 0; i < 2 * VLC_R; i++) {
        int sv = i - VLC_R;
        uint32_t v = (uint32_t)(sv < 0 ? -sv : sv) + 1;
        int k = 31 - __builtin_clz(v);
        vlc_code[i] = sv ? spread16(v - (1u << k)) << 2 | 2 | (sv < 0) : 1;
        vlc_len[i] = sv ? 2 * k + 2 : 1;
    }
}

// One slice row: quantise band row by band row into per-slice buffers in coding order, then write the slices.
static void write_job(void *arg, int sy, int thread)
{
    vc2_enc *e = arg;
    escratch *s = &e->scr[thread];
    const int depth = e->c.depth, nx = e->nx, sc = e->scaler;
    size_t comp_n[3];
    int32_t *g = s->gather;
    // gather: g[c] holds, slice after slice, the quantised rows of the slice (all rows: the cut is applied below)
    int32_t *gc[3];
    for (int c = 0; c < 3; c++) {
        const eplane *p = &e->pl[c];
        size_t per_slice = 0;
        for (int l = 0; l <= depth; l++) per_slice += (size_t)(l ? 3 : 1) * (p->ys[l][sy + 1] - p->ys[l][sy]) * ((p->g.bw[l] + nx - 1) / nx + 1);
        comp_n[c] = per_slice;
        gc[c] = g;
        g += per_slice * nx;
        size_t off = 0;
        int n = 0;
        for (int l = 0; l <= depth; l++)
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++)
                for (int y = p->ys[l][sy]; y < p->ys[l][sy + 1]; y++, n++) {
                    const int32_t *row = p->band[l][o] + (size_t)y * p->g.bw[l];
                    const int *xs = p->xs[l];
                    for (int sx = 0; sx < nx; sx++) {
                        int si = sy * nx + sx, q = e->sq[si] - e->qm[l][o];
                        if (n >= e->cut[e->sq[si]][si * 3 + c]) continue;
                        int32_t *gv = gc[c] + per_slice * sx + off;
                        // an all-zero row is marked in its spare last entry and written as a run of ones
                        gv[xs[sx + 1] - xs[sx]] = quant_row(row + xs[sx], xs[sx + 1] - xs[sx], e->r16[q < 0 ? 0 : q], gv);
                    }
                    off += (p->g.bw[l] + nx - 1) / nx + 1;
                }
    }
    for (int sx = 0; sx < nx; sx++) {
        const int si = sy * nx + sx, qi = e->sq[si];
        const uint16_t *bytes = e->cb[qi] + (size_t)si * 3, *cuts = e->cut[qi] + (size_t)si * 3;
        uint8_t *o = s->obuf;
        *o++ = qi;
        for (int c = 0; c < 3; c++) {
            const eplane *p = &e->pl[c];
            int padded = (bytes[c] + sc - 1) / sc * sc, n = 0;
            const int32_t *vq = gc[c] + comp_n[c] * sx;
            *o++ = padded / sc;
            sbw b = { o, 0, 0 };
            for (int l = 0; l <= depth && n < cuts[c]; l++) {
                int w = p->xs[l][sx + 1] - p->xs[l][sx], adv = (p->g.bw[l] + nx - 1) / nx + 1;
                for (int k = (l ? 3 : 1) * (p->ys[l][sy + 1] - p->ys[l][sy]); k > 0 && n < cuts[c]; k--, n++, vq += adv) {
                    if (!vq[w]) {
                        for (int x = w; x > 0; x -= 32) put(&b, x >= 32 ? 0xffffffffu : (1u << x) - 1, x >= 32 ? 32 : x);
                        continue;
                    }
                    for (int x = 0; x < w; x++) {
                        uint32_t i = (uint32_t)(vq[x] + VLC_R);
                        if (__builtin_expect(i < 2 * VLC_R, 1)) { put(&b, vlc_code[i], vlc_len[i]); continue; }
                        uint32_t v = (uint32_t)(vq[x] < 0 ? -vq[x] : vq[x]) + 1;
                        int kk = 31 - __builtin_clz(v);
                        for (int j = kk - 1; j >= 0; j--) put(&b, (v >> j) & 1, 2);
                        put(&b, 2 | (vq[x] < 0), 2);
                    }
                }
            }
            // the rest of the counted size, and the padding, are ones (zeros for the decoder)
            put(&b, 0x7f, 7);
            if (b.p < o + padded) memset(b.p, 0xff, o + padded - b.p);
            o += padded;
        }
        memcpy(e->out_slices + e->soff[si], s->obuf, e->soff[si + 1] - e->soff[si]);
    }
}

size_t vc2_enc_encode(vc2_enc *e, const uint16_t *const plane[3], const ptrdiff_t stride[3], uint8_t *out)
{
    double t0 = now(); n_eval = 0;
    e->in = plane; e->in_stride = stride;
    vc2_pool_run(e->pool, plane_job, e, 3);
    double t1 = now();

    const int unit = e->c.rc_rows > 0 && e->c.rc_rows < e->ny ? e->c.rc_rows : e->ny;
    const double frame_budget = e->c.width * (double)e->c.height * e->c.bpp / 8 - 64;
    uint8_t uq[4096];
    // quantiser per unit; the size scaler is common to the picture, so repeat if a unit needs a larger one
    for (e->scaler = 1;;) {
        int sc_need = e->scaler, q = e->last_q;
        for (int r0 = 0, u = 0; r0 < e->ny; r0 += unit, u++) {
            int r1 = r0 + unit < e->ny ? r0 + unit : e->ny, sc = e->scaler;
            size_t budget = (size_t)(frame_budget * (r1 - r0) / e->ny), tot = 0;
            if (e->c.fixed_q > 0) { q = e->c.fixed_q; fits(e, q, r0, r1, 0, &sc, &tot); }
            else {
                while (!fits(e, q, r0, r1, budget, &sc, &tot) && q < VC2_NQ - 1) q++;
                // one index lower is worth a full count only when a good part of the budget is left
                while (q > 0 && tot * 100 <= budget * 82 && fits(e, q - 1, r0, r1, budget, &sc, &tot)) q--;
                fits(e, q, r0, r1, budget, &sc, &tot);
            }
            if (sc > sc_need) sc_need = sc;
            uq[u] = q;
        }
        if (sc_need == e->scaler) break;
        e->scaler = sc_need;
    }
    e->last_q = uq[0];
    // refinement: slices one index lower while the unit's budget lasts. The lower index is counted only for as
    // many slice rows (spread over the unit) as the remaining budget is likely to pay for.
    for (int r0 = 0, u = 0; r0 < e->ny; r0 += unit, u++) {
        int r1 = r0 + unit < e->ny ? r0 + unit : e->ny, q = uq[u], n = r1 - r0, lim = 255 * e->scaler;
        size_t budget = (size_t)(frame_budget * n / e->ny), tot = total_size(e, q, r0, r1, e->scaler, 0);
        int step = (int)(n * 0.6180339887) | 1, order[4096];
        while (n > 1 && gcd(step, n) != 1) step += 2;
        for (int i = 0, k = 0; i < n; i++, k = (k + step) % n) order[i] = r0 + k;
        for (int i = 0; i < n * e->nx; i++) e->sq[r0 * e->nx + i] = q;
        const size_t small = tot / ((size_t)n * e->nx) / 8 + 1; // an upgrade costs about this much at least
        for (int done = 0; q > 0 && e->c.fixed_q <= 0 && done < n && tot + small < budget;) {
            int batch = (int)((budget - tot) / (tot / (double)n * 0.12)) + 2;
            if (batch > n - done) batch = n - done;
            eval_q(e, q - 1, 0, 0, order + done, batch);
            for (int i = done; i < done + batch; i++)
                for (int sx = 0; sx < e->nx; sx++) {
                    int si = order[i] * e->nx + sx;
                    const uint16_t *lo = e->cb[q - 1] + (size_t)si * 3;
                    size_t a = slice_size(e->cb[q] + (size_t)si * 3, e->scaler), b = slice_size(lo, e->scaler);
                    if (tot - a + b <= budget && lo[0] <= lim && lo[1] <= lim && lo[2] <= lim) { tot = tot - a + b; e->sq[si] = q - 1; }
                }
            done += batch;
        }
    }
    double t2 = now();
    e->soff[0] = 0;
    e->q_sum = 0;
    for (int i = 0; i < e->nslices; i++) {
        e->soff[i + 1] = e->soff[i] + slice_size(e->cb[e->sq[i]] + (size_t)i * 3, e->scaler);
        e->q_sum += e->sq[i];
    }

    // headers
    uint8_t *o = out;
    size_t n = vc2_write_seq(o + 13, &e->seq);
    vc2_write_parse_info(o, VC2_PC_SEQ_HEADER, 13 + n, e->prev_unit);
    uint32_t prev = 13 + n;
    o += prev;
    vc2_bw b = { o + 13, 0 };
    for (int i = 3; i >= 0; i--) o[13 + 3 - i] = (uint8_t)(e->frame >> (8 * i));
    b.pos = 32;
    vc2_bw_uint(&b, e->c.wavelet); vc2_bw_uint(&b, e->c.depth);
    vc2_bw_uint(&b, e->nx); vc2_bw_uint(&b, e->ny);
    vc2_bw_uint(&b, 0); vc2_bw_uint(&b, e->scaler);
    vc2_bw_bit(&b, 1);
    for (int l = 0; l <= e->c.depth; l++)
        for (int oo = l ? 1 : 0; oo < (l ? 4 : 1); oo++) vc2_bw_uint(&b, e->qm[l][oo]);
    vc2_bw_align(&b);
    size_t hdr = b.pos >> 3, pic = 13 + hdr + e->soff[e->nslices];
    vc2_write_parse_info(o, VC2_PC_PICTURE_HQ, pic, prev);
    e->out_slices = o + 13 + hdr;
    vc2_pool_run(e->pool, write_job, e, e->ny);
    if (getenv("VC2_PROF")) fprintf(stderr, "dwt %.1f rc %.1f (%d counts) write %.1f ms\n", (t1 - t0) * 1e3, (t2 - t1) * 1e3, n_eval, (now() - t2) * 1e3);
    e->frame++;
    e->prev_unit = (uint32_t)pic;
    return prev + pic;
}

size_t vc2_enc_end(vc2_enc *e, uint8_t *out)
{
    size_t n = vc2_write_parse_info(out, VC2_PC_END_SEQ, 0, e->prev_unit);
    e->prev_unit = 0;
    return n;
}
