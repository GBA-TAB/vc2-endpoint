// VC-2 HQ encoder: forward wavelet, rate control with one quantiser index per rate-control unit (the whole
// frame by default) plus per-slice refinement, tail truncation, slice writer.
#define _GNU_SOURCE
#include "vc2_priv.h"
#include "vc2_enc_gpu.h"
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
#define T int16_t
#define SUF _16
#define T16
#include "vc2_lift.h"
#undef T16
#undef T
#undef SUF

typedef struct {
    vc2_geom g;
    int32_t *img;                        // padded input, offset removed
    int32_t *band[VC2_MAX_DEPTH + 1][4];
    int32_t *ll[VC2_MAX_DEPTH + 1];
    // 16-bit storage for the finest levels (l >= l16, see vc2_enc.l16): their bands, the input picture and the
    // LL between two 16-bit levels; the last 16-bit level's LL goes to ll[] in 32 bits
    int16_t *img16, *band16[VC2_MAX_DEPTH + 1][4], *ll16[VC2_MAX_DEPTH + 1];
    int16_t *own16[VC2_MAX_DEPTH + 1][4]; // band16 as allocated (with the GPU front end band16 points into its maps)
    const int16_t *gll;                   // the GPU's LL of level 3 (this frame)
    int *xs[VC2_MAX_DEPTH + 1], *ys[VC2_MAX_DEPTH + 1];
} eplane;

typedef struct { const void *p; int level, is16; float r16; } rowref; // one band row of a slice row
// The counter's constants for 16-bit rows, per quantiser index: v = floor(n / qf) = mulhi(n, m) >> s and
// f = floor(16 r / qf) = mulhi(r << k, m2) >> s2 with r = n - v qf, both checked exhaustively at start-up for every
// n = 4|c| with |c| <= 12813 (the bound of the 16-bit levels) and every r < qf. mode 0: no such constants, the
// generic row counter; 1: the vector counter; 2: qf > 51252, every value of such a row is 0.
typedef struct { uint16_t m, m2, qf; uint8_t s, s2, k, mode, av, af; } cnt16; // av/af: the 17-bit form (see c16_find)
static const cnt16 c16_off = { 0 };
static int getenv_no16c = -1; // VC2_NO16C=1: the 16-bit rows through the generic counter
static int getenv_nogc;       // VC2_NO_GPU_COUNT=1: with the GPU front end, count on the CPU
typedef struct { rowref *rows; uint32_t *bits, *dd; int32_t *tmp, *gather; uint8_t *obuf; } escratch;

struct vc2_enc {
    vc2_enc_cfg c;
    vc2_pool *pool;
    const vc2_wavelet *wl;
    vc2_seq seq;
    uint8_t qm[VC2_MAX_DEPTH + 1][4];
    eplane pl[3];
    int nx, ny, nslices, max_rows;
    // levels l >= l16 are transformed and stored in 16 bits: LeGall 5,3, depth 4, 10-bit, where the bounds are
    // proven (every band and every sum inside a lifting step of the two finest levels below 32768); else depth + 1
    int l16;
    vc2_encgpu *gpu;            // the GPU front end, when set (vc2_enc_set_gpu)
    double gpu_ms;
    int gpu_frame;              // this frame's 16-bit bands came from the GPU
    const uint32_t *gcount;     // the GPU's counts for the quantiser being evaluated, or 0
    float r16[VC2_NQ];
    cnt16 c16[VC2_NQ];
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

// Constants for floor(x * mul / qf), x = 0, step, .. <= xmax, with y = x << pre < 65536, in 16-bit lanes:
//   plain: mulhi(y, m) >> s, i.e. floor(y m / 2^(16+s)), m < 2^16;
//   17-bit (*a = 1): ((y + mulhi(y, m)) >> 1) >> (s - 1) by a halving add, i.e. floor(y (2^16 + m) / 2^(16+s)), s >= 1.
// Checked for every x; the first form that holds is taken.
static int c16_find(uint32_t qf, uint32_t step, uint32_t xmax, uint32_t mul, uint32_t pre, uint16_t *m, uint8_t *s, uint8_t *a)
{
    if ((xmax << pre) > 65535) return 0;
    for (uint32_t add = 0; add < 2; add++)
        for (uint32_t ss = add; ss < 16; ss++) {
            uint64_t t = ((uint64_t)mul << (16 + ss)) / qf >> pre;
            for (uint64_t mm = t; mm <= t + 2; mm++) {
                uint64_t mf = mm; // the full multiplier, 2^16 + m in the 17-bit form
                if (add ? (mf < 65536 || mf > 131071) : (mf < 1 || mf > 65535)) continue;
                int ok = 1;
                for (uint32_t x = 0; x <= xmax && ok; x += step)
                    ok = (uint32_t)(((uint64_t)(x << pre) * mf) >> (16 + ss)) == (uint32_t)((uint64_t)x * mul / qf);
                if (ok) { *m = (uint16_t)(mf & 0xffff); *s = (uint8_t)ss; *a = (uint8_t)add; return 1; }
            }
        }
    return 0;
}
static void c16_init(cnt16 *t, uint32_t qf, float r16)
{
    memset(t, 0, sizeof *t);
    // the float counter must equal the integer floors on the whole range, else stay with it
    for (uint32_t c = 0; c <= 12813; c++)
        if ((uint32_t)(int32_t)((float)(c << 2) * r16) != (uint32_t)((uint64_t)64 * c / qf)) return;
    if (qf > 51252) { t->mode = 2; return; }
    t->qf = (uint16_t)qf; t->k = 16 * qf <= 65536 ? 4 : 0;
    if (c16_find(qf, 4, 51252, 1, 0, &t->m, &t->s, &t->av) && c16_find(qf, 1, qf - 1, 16, t->k, &t->m2, &t->s2, &t->af)) t->mode = 1;
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
    e->l16 = c->wavelet == VC2_WL_LEGALL5_3 && c->depth == 4 && c->bit_depth <= 10 && !getenv("VC2_NO16") ? c->depth - 1 : c->depth + 1;
    for (int i = 0; i < 3; i++) {
        eplane *p = &e->pl[i];
        if (e->l16 <= c->depth) p->img16 = vc2_alloc((size_t)p->g.pw * p->g.ph * 2);
        else p->img = vc2_alloc((size_t)p->g.pw * p->g.ph * 4);
        for (int l = 0; l <= c->depth; l++) {
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
                if (l >= e->l16 && o) p->band16[l][o] = p->own16[l][o] = vc2_alloc((size_t)p->g.bw[l] * p->g.bh[l] * 2);
                else p->band[l][o] = vc2_alloc((size_t)p->g.bw[l] * p->g.bh[l] * 4);
            }
            if (l && l < c->depth) p->ll[l] = vc2_alloc((size_t)16 * p->g.bw[l] * p->g.bh[l]);
            if (l && l < c->depth && l >= e->l16 - 1) p->ll16[l] = vc2_alloc((size_t)8 * p->g.bw[l] * p->g.bh[l]);
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
    for (int q = 0; q < VC2_NQ; q++) c16_init(&e->c16[q], vc2_qf[q], e->r16[q]);
    if (getenv_no16c < 0) getenv_no16c = getenv("VC2_NO16C") != 0;
    getenv_nogc = getenv("VC2_NO_GPU_COUNT") != 0;
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
    vc2_encgpu_free(e->gpu);
    for (int i = 0; i < 3; i++) {
        eplane *p = &e->pl[i];
        free(p->img); free(p->img16);
        for (int l = 0; l <= VC2_MAX_DEPTH; l++) {
            for (int o = 0; o < 4; o++) { free(p->band[l][o]); free(p->own16[l][o]); }
            free(p->ll[l]); free(p->ll16[l]); free(p->xs[l]); free(p->ys[l]);
        }
    }
    for (int i = 0; i < th; i++) { free(e->scr[i].rows); free(e->scr[i].bits); free(e->scr[i].dd); free(e->scr[i].tmp); free(e->scr[i].gather); free(e->scr[i].obuf); }
    for (int q = 0; q < VC2_NQ; q++) { free(e->cb[q]); free(e->cut[q]); free(e->valid[q]); }
    free(e->scr); free(e->sq); free(e->soff);
    free(e);
}

int vc2_enc_set_gpu(vc2_enc *e, int backend)
{
    vc2_encgpu_free(e->gpu);
    e->gpu = 0;
    if (backend == VC2_GPU_OFF) return 0;
    if (backend != VC2_GPU_OPENCL || e->l16 != 3 || e->c.depth != 4) return -1; // the kernel is LeGall 5,3, 16-bit
    int pw[3] = { e->pl[0].g.pw, e->pl[1].g.pw, e->pl[2].g.pw };
    if (e->pl[1].g.ph != e->pl[0].g.ph) return -1; // 4:2:2 / 4:4:4 only
    e->gpu = vc2_encgpu_new(pw, e->pl[0].g.ph, e->nx);
    return e->gpu ? 0 : -1;
}
double vc2_enc_gpu_ms(const vc2_enc *e) { return e->gpu ? e->gpu_ms : -1; }

size_t vc2_enc_max_size(const vc2_enc *e)
{
    // The budget holds unless nothing smaller exists: a fixed quantiser, or explicit zeros (--no-truncate: at
    // least one bit per coefficient, so even the coarsest quantiser can exceed a low budget), plus the slice
    // headers and the per-component padding to the size scaler.
    double bpp = e->c.bpp;
    if (e->c.fixed_q > 0) bpp = 40;
    else if (e->c.no_truncate) {
        double spp = e->c.chroma == 0 ? 3 : e->c.chroma == 1 ? 2 : 1.5; // coefficients per pixel
        if (bpp < spp + 1) bpp = spp + 1;
    }
    return (size_t)(e->c.width * (double)e->c.height * bpp / 8) + (size_t)e->nslices * (4 + 3 * 256) + 4096 + VC2_INPUT_PADDING;
}
int vc2_enc_last_q(const vc2_enc *e) { return (int)(e->q_sum * 100 / e->nslices); }

// --- forward wavelet ---------------------------------------------------------------------------------------
// One level, line by line: after each input row pair is lifted horizontally, every vertical lifting step runs on
// the band rows whose inputs are final, so the steps work on rows still in cache instead of sweeping the whole
// bands once per step (the wavelet is memory-bound on small boards). The arithmetic and its order per row are
// those of the whole-band version; the steps of every VC-2 wavelet alternate between the two row types.
static void analyse_level(const vc2_wavelet *wl, const int32_t *in, ptrdiff_t istride, int w, int h, int32_t *bands[4], int32_t *tmp)
{
    const int ns = wl->nsteps, sh = wl->shift, hw = (w + 2 * VC2_G + 7) & ~7;
    int32_t *lo = tmp + VC2_G, *hi = tmp + hw + VC2_G;
    int done[5] = { 0 }, rlo[5], rhi[5], tg[5];
    for (int k = 1; k <= ns; k++) { const vc2_step *st = &wl->st[ns - k]; tg[k] = st->tgt; vc2_step_reach(st, &rlo[k], &rhi[k]); }
#define CLAMP(i) ((i) < 0 ? 0 : (i) >= h ? h - 1 : (i))
    for (int p = 0; p <= h; p++) {
        if (p < h) {
            for (int y = 2 * p; y < 2 * p + 2; y++) {
                const int32_t *r = in + (size_t)y * istride;
                for (int x = 0; x < w; x++) { lo[x] = r[2 * x] << sh; hi[x] = r[2 * x + 1] << sh; }
                for (int j = ns - 1; j >= 0; j--) vc2_lift_h_32(lo, hi, w, &wl->st[j], -wl->st[j].sign);
                memcpy(bands[(y & 1) * 2] + (size_t)(y / 2) * w, lo, w * 4);
                memcpy(bands[(y & 1) * 2 + 1] + (size_t)(y / 2) * w, hi, w * 4);
            }
            done[0] = p + 1;
        }
        for (int k = 1; k <= ns; k++) {
            const vc2_step *st = &wl->st[ns - k];
            const int t = tg[k], off = t ? -1 : -2;
            for (;;) {
                int i = done[k];
                if (i >= h) break;
                // the other row type, as the previous step (or the horizontal pass) left it
                int need = i + rhi[k] > h - 1 ? h - 1 : i + rhi[k];
                if (done[k - 1] <= need) break;
                // the previous step read this row type: all its rows that read row i must be done
                if (k >= 2) {
                    int last = i - rlo[k - 1];
                    if ((last >= h - 1 || i == h - 1) ? done[k - 1] < h : done[k - 1] <= last) break;
                }
                for (int half = 0; half < 2; half++) {
                    int32_t *tb = bands[t * 2 + half];
                    const int32_t *ob = bands[!t * 2 + half];
                    vc2_lift_32(tb + (size_t)i * w, tb + (size_t)i * w, ob + (size_t)CLAMP(i + off) * w, ob + (size_t)CLAMP(i + off + 1) * w,
                                ob + (size_t)CLAMP(i + off + 2) * w, ob + (size_t)CLAMP(i + off + 3) * w, w, st, -st->sign);
                }
                done[k] = i + 1;
            }
        }
    }
#undef CLAMP
}

// The same in 16-bit lanes, for the levels whose bounds allow it (vc2_enc.l16)
static void analyse_level16(const vc2_wavelet *wl, const int16_t *in, ptrdiff_t istride, int w, int h, int16_t *bands[4], int16_t *tmp)
{
    const int ns = wl->nsteps, sh = wl->shift, hw = (w + 2 * VC2_G + 7) & ~7;
    int16_t *lo = tmp + VC2_G, *hi = tmp + hw + VC2_G;
    int done[5] = { 0 }, rlo[5], rhi[5], tg[5];
    for (int k = 1; k <= ns; k++) { const vc2_step *st = &wl->st[ns - k]; tg[k] = st->tgt; vc2_step_reach(st, &rlo[k], &rhi[k]); }
#define CLAMP(i) ((i) < 0 ? 0 : (i) >= h ? h - 1 : (i))
    for (int p = 0; p <= h; p++) {
        if (p < h) {
            for (int y = 2 * p; y < 2 * p + 2; y++) {
                const int16_t *r = in + (size_t)y * istride;
                for (int x = 0; x < w; x++) { lo[x] = (int16_t)(r[2 * x] << sh); hi[x] = (int16_t)(r[2 * x + 1] << sh); }
                for (int j = ns - 1; j >= 0; j--) vc2_lift_h_16(lo, hi, w, &wl->st[j], -wl->st[j].sign);
                memcpy(bands[(y & 1) * 2] + (size_t)(y / 2) * w, lo, w * 2);
                memcpy(bands[(y & 1) * 2 + 1] + (size_t)(y / 2) * w, hi, w * 2);
            }
            done[0] = p + 1;
        }
        for (int k = 1; k <= ns; k++) {
            const vc2_step *st = &wl->st[ns - k];
            const int t = tg[k], off = t ? -1 : -2;
            for (;;) {
                int i = done[k];
                if (i >= h) break;
                // the other row type, as the previous step (or the horizontal pass) left it
                int need = i + rhi[k] > h - 1 ? h - 1 : i + rhi[k];
                if (done[k - 1] <= need) break;
                // the previous step read this row type: all its rows that read row i must be done
                if (k >= 2) {
                    int last = i - rlo[k - 1];
                    if ((last >= h - 1 || i == h - 1) ? done[k - 1] < h : done[k - 1] <= last) break;
                }
                for (int half = 0; half < 2; half++) {
                    int16_t *tb = bands[t * 2 + half];
                    const int16_t *ob = bands[!t * 2 + half];
                    vc2_lift_16(tb + (size_t)i * w, tb + (size_t)i * w, ob + (size_t)CLAMP(i + off) * w, ob + (size_t)CLAMP(i + off + 1) * w,
                                ob + (size_t)CLAMP(i + off + 2) * w, ob + (size_t)CLAMP(i + off + 3) * w, w, st, -st->sign);
                }
                done[k] = i + 1;
            }
        }
    }
#undef CLAMP
}

// the GPU front end: the input planes into its buffer, then (after the launch) the coarse levels on the CPU
static void gpu_in_job(void *arg, int job, int thread)
{
    (void)thread;
    vc2_enc *e = arg;
    eplane *p = &e->pl[job];
    const int w = p->g.w, h = p->g.h, pw = p->g.pw, ph = p->g.ph, off = 1 << (e->c.bit_depth - 1), max = (1 << e->c.bit_depth) - 1;
    int16_t *dst = vc2_encgpu_input(e->gpu, job);
    for (int y = 0; y < ph; y++) {
        const uint16_t *s = e->in[job] + (size_t)(y < h ? y : h - 1) * e->in_stride[job];
        int16_t *d = dst + (size_t)y * pw;
        for (int x = 0; x < w; x++) d[x] = (int16_t)((s[x] > max ? max : (int32_t)s[x]) - off);
        for (int x = w; x < pw; x++) d[x] = d[w - 1];
    }
}
static void gpu_coarse_job(void *arg, int job, int thread)
{
    vc2_enc *e = arg;
    eplane *p = &e->pl[job];
    for (size_t i = 0, n = (size_t)p->g.bw[3] * p->g.bh[3]; i < n; i++) p->ll[2][i] = p->gll[i]; // the GPU's last LL, widened
    for (int l = e->l16 - 1; l >= 1; l--) {
        const int32_t *in = p->ll[l];
        int32_t *bands[4] = { l == 1 ? p->band[0][0] : p->ll[l - 1], p->band[l][1], p->band[l][2], p->band[l][3] };
        analyse_level(e->wl, in, 2 * p->g.bw[l], p->g.bw[l], p->g.bh[l], bands, e->scr[thread].tmp);
    }
}
static int gpu_wavelet(vc2_enc *e)
{
    for (int c = 0; c < 3; c++) if (!vc2_encgpu_input(e->gpu, c)) return -1;
    vc2_pool_run(e->pool, gpu_in_job, e, 3);
    int16_t *b4[3][4], *b3[3][4];
    if (vc2_encgpu_run(e->gpu, b4, b3, &e->gpu_ms)) return -1;
    for (int c = 0; c < 3; c++) {
        eplane *p = &e->pl[c];
        for (int o = 1; o < 4; o++) { p->band16[4][o] = b4[c][o]; p->band16[3][o] = b3[c][o]; }
        p->gll = b3[c][0];
    }
    vc2_pool_run(e->pool, gpu_coarse_job, e, 3);
    return 0;
}

static void plane_job(void *arg, int job, int thread)
{
    vc2_enc *e = arg;
    eplane *p = &e->pl[job];
    const int w = p->g.w, h = p->g.h, pw = p->g.pw, ph = p->g.ph, off = 1 << (e->c.bit_depth - 1), depth = e->c.depth;
    for (int y = 0; y < ph; y++) {
        const uint16_t *s = e->in[job] + (size_t)(y < h ? y : h - 1) * e->in_stride[job];
        if (p->img16) {
            int16_t *d = p->img16 + (size_t)y * pw;
            // the 16-bit bounds hold for legal samples only: clamp what a file may carry beyond the bit depth
            const int max = (1 << e->c.bit_depth) - 1;
            for (int x = 0; x < w; x++) d[x] = (int16_t)((s[x] > max ? max : (int32_t)s[x]) - off);
            for (int x = w; x < pw; x++) d[x] = d[w - 1];
        } else {
            int32_t *d = p->img + (size_t)y * pw;
            for (int x = 0; x < w; x++) d[x] = (int32_t)s[x] - off;
            for (int x = w; x < pw; x++) d[x] = d[w - 1];
        }
    }
    for (int l = depth; l >= 1; l--) {
        if (l >= e->l16) {
            const int16_t *in = l == depth ? p->img16 : p->ll16[l];
            // the last 16-bit level writes its LL into a 16-bit buffer first and widens it for the 32-bit levels
            int16_t *bands[4] = { p->ll16[l - 1], p->band16[l][1], p->band16[l][2], p->band16[l][3] };
            analyse_level16(e->wl, in, 2 * p->g.bw[l], p->g.bw[l], p->g.bh[l], bands, (int16_t *)e->scr[thread].tmp);
            if (l == e->l16) {
                int32_t *ll = l == 1 ? p->band[0][0] : p->ll[l - 1];
                for (size_t i = 0, n = (size_t)p->g.bw[l] * p->g.bh[l]; i < n; i++) ll[i] = p->ll16[l - 1][i];
            }
        } else {
            const int32_t *in = l == depth ? p->img : p->ll[l];
            int32_t *bands[4] = { l == 1 ? p->band[0][0] : p->ll[l - 1], p->band[l][1], p->band[l][2], p->band[l][3] };
            analyse_level(e->wl, in, 2 * p->g.bw[l], p->g.bw[l], p->g.bh[l], bands, e->scr[thread].tmp);
        }
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

// The same for rows stored in 16 bits (widened on load; the arithmetic is the same)
// Bits, truncation cost (in 1/16 of a squared quantiser step) and "any nonzero" of one coefficient row.
static inline __attribute__((always_inline)) void count_row_w16(const int16_t *c, int w, float r16, uint32_t *bits, uint32_t *dd, int *nz)
{
    uint32_t b = 0, d = 0, any = 0;
    for (int x = 0; x < w; x++) {
        int32_t n = (int32_t)(c[x] < 0 ? -c[x] : c[x]) << 2;
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
static void count_row16(const int16_t *c, int w, float r16, uint32_t *bits, uint32_t *dd, int *nz)
{
    switch (w) {
    case 8: count_row_w16(c, 8, r16, bits, dd, nz); break;
    case 16: count_row_w16(c, 16, r16, bits, dd, nz); break;
    case 32: count_row_w16(c, 32, r16, bits, dd, nz); break;
    case 64: count_row_w16(c, 64, r16, bits, dd, nz); break;
    default: count_row_w16(c, w, r16, bits, dd, nz);
    }
}
// quantised signed values of one row
static inline __attribute__((always_inline)) int quant_row_w16(const int16_t *c, int w, float r16, int32_t *v)
{
    int32_t any = 0;
    for (int x = 0; x < w; x++) {
        int32_t a = c[x] < 0 ? -(int32_t)c[x] : c[x], q = (int32_t)((float)(a << 2) * r16) >> 4;
        v[x] = c[x] < 0 ? -q : q;
        any |= q;
    }
    return any != 0;
}
// returns 0 if the whole row quantises to zero
static int quant_row16(const int16_t *c, int w, float r16, int32_t *v)
{
    switch (w) {
    case 8: return quant_row_w16(c, 8, r16, v);
    case 16: return quant_row_w16(c, 16, r16, v);
    case 32: return quant_row_w16(c, 32, r16, v);
    case 64: return quant_row_w16(c, 64, r16, v);
    default: return quant_row_w16(c, w, r16, v);
    }
}

// The 16-bit rows' counter: the arithmetic of count_row_w in exact integers, eight lanes of 16 bits (NEON).
// bits = 2 J + 1 + [v > 0] with J = floor(log2(v + 1)); dd = (2 vv + 1)(uu - 8 vv - 4) for v > 0, where vv = min(v, 255)
// and uu = min(16 v + f, 4096): for v <= 255 that is (2v + 1)(8v + f - 4), above it the constant 511 * 2052.
#ifdef __aarch64__
#include <arm_neon.h>
static inline uint16x8_t mulhi16(uint16x8_t a, uint16x8_t b)
{
    return vuzp2q_u16(vreinterpretq_u16_u32(vmull_u16(vget_low_u16(a), vget_low_u16(b))), vreinterpretq_u16_u32(vmull_high_u16(a, b)));
}
#endif
static inline __attribute__((always_inline)) void count_row16v(const int16_t *c, int w, const cnt16 *t, float r16, uint32_t *bits, uint32_t *dd, int *nz)
{
    if (t->mode == 2) { *bits = (uint32_t)w; *dd = 0; *nz = 0; return; }
#ifdef __aarch64__
    if (t->mode == 1 && w >= 8) {
        const uint16x8_t m = vdupq_n_u16(t->m), m2 = vdupq_n_u16(t->m2), qf = vdupq_n_u16(t->qf), lim = vdupq_n_u16(255);
        const int16x8_t sh = vdupq_n_s16(-(int16_t)(t->s - t->av)), sh2 = vdupq_n_s16(-(int16_t)(t->s2 - t->af)), k = vdupq_n_s16((int16_t)t->k);
        const int av = t->av, af = t->af;
        uint16x8_t accb = vdupq_n_u16(0), any = vdupq_n_u16(0);
        uint32x4_t accd = vdupq_n_u32(0);
        int x = 0;
        for (; x + 8 <= w; x += 8) {
            uint16x8_t n = vshlq_n_u16(vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(c + x))), 2);
            uint16x8_t hv = mulhi16(n, m);
            uint16x8_t v = vshlq_u16(av ? vhaddq_u16(n, hv) : hv, sh);
            uint16x8_t r = vmlsq_u16(n, v, qf), rk = vshlq_u16(r, k), hf = mulhi16(rk, m2);
            uint16x8_t f = vshlq_u16(af ? vhaddq_u16(rk, hf) : hf, sh2);
            uint16x8_t nzm = vtstq_u16(v, v);
            uint16x8_t j = vsubq_u16(vdupq_n_u16(15), vclzq_u16(vaddq_u16(v, vdupq_n_u16(1))));
            accb = vaddq_u16(accb, vsubq_u16(vaddq_u16(vshlq_n_u16(j, 1), vdupq_n_u16(1)), nzm));
            any = vorrq_u16(any, v);
            uint16x8_t vv = vminq_u16(v, lim);
            int16x8_t a = vreinterpretq_s16_u16(vandq_u16(vaddq_u16(vshlq_n_u16(vv, 1), vdupq_n_u16(1)), nzm));
            uint16x8_t bt = vsubq_u16(vaddq_u16(vshlq_n_u16(vv, 3), f), vdupq_n_u16(4));
            bt = vbslq_u16(vcgtq_u16(v, lim), vdupq_n_u16(2052), bt);
            int16x8_t b = vreinterpretq_s16_u16(bt);
            accd = vreinterpretq_u32_s32(vmlal_s16(vreinterpretq_s32_u32(accd), vget_low_s16(a), vget_low_s16(b)));
            accd = vreinterpretq_u32_s32(vmlal_high_s16(vreinterpretq_s32_u32(accd), a, b));
        }
        uint32_t bsum = vaddlvq_u16(accb), dsum = vaddvq_u32(accd), anys = vmaxvq_u16(any);
        if (x < w) { uint32_t b2, d2; int nz2; count_row16(c + x, w - x, r16, &b2, &d2, &nz2); bsum += b2; dsum += d2; anys |= (uint32_t)nz2; }
        *bits = bsum; *dd = dsum; *nz = anys != 0;
        return;
    }
#endif
    count_row16(c, w, r16, bits, dd, nz);
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
                const cnt16 *ct = getenv_no16c ? &c16_off : &e->c16[q < 0 ? 0 : q];
                for (int y = p->ys[l][sy]; y < p->ys[l][sy + 1]; y++, n++) {
                    const int *xs = p->xs[l];
                    int nz;
                    if (p->band16[l][o]) {
                        const int16_t *row = p->band16[l][o] + (size_t)y * p->g.bw[l];
                        s->rows[n] = (rowref){ row, l, 1, r16 };
                        if (e->gcount) {
                            const uint32_t *gc = e->gcount + 2 * ((size_t)vc2_encgpu_base(e->gpu, c, l, o) + (size_t)y * nx);
                            for (int sx = 0; sx < nx; sx++) { s->bits[n * nx + sx] = gc[2 * sx]; s->dd[n * nx + sx] = gc[2 * sx + 1]; }
                        } else
                            for (int sx = 0; sx < nx; sx++)
                                count_row16v(row + xs[sx], xs[sx + 1] - xs[sx], ct, r16, &s->bits[n * nx + sx], &s->dd[n * nx + sx], &nz);
                    } else {
                        const int32_t *row = p->band[l][o] + (size_t)y * p->g.bw[l];
                        s->rows[n] = (rowref){ row, l, 0, r16 };
                        for (int sx = 0; sx < nx; sx++)
                            count_row(row + xs[sx], xs[sx + 1] - xs[sx], r16, &s->bits[n * nx + sx], &s->dd[n * nx + sx], &nz);
                    }
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
                    int32_t v = r->is16 ? ((const int16_t *)r->p)[x0 + x - 1] : ((const int32_t *)r->p)[x0 + x - 1];
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
    // the 16-bit bands' counts from the GPU when this frame's bands are there and the evaluation is not a few
    // single slice rows (the slice-size cap's corrections), for which the CPU is quicker than a launch
    e->gcount = 0;
    if (e->gpu_frame && !getenv_nogc && (rows ? nrows : r1 - r0) * 8 >= e->ny) {
        float r[18];
        int i = 0;
        for (int c = 0; c < 3; c++)
            for (int l = 4; l >= 3; l--)
                for (int o = 1; o < 4; o++) { int qq = q - e->qm[l][o]; r[i++] = e->r16[qq < 0 ? 0 : qq]; }
        e->gcount = vc2_encgpu_count(e->gpu, q, r, &e->gpu_ms);
    }
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
// two values at once, |v| <= 7 each: the two codes back to back (at most 8 + 8 bits)
#define VLC_P 7
static uint16_t vlc2_code[(2 * VLC_P + 1) * (2 * VLC_P + 1)];
static uint8_t vlc2_len[(2 * VLC_P + 1) * (2 * VLC_P + 1)];
static void vlc_init(void)
{
    for (int i = 0; i < 2 * VLC_R; i++) {
        int sv = i - VLC_R;
        uint32_t v = (uint32_t)(sv < 0 ? -sv : sv) + 1;
        int k = 31 - __builtin_clz(v);
        vlc_code[i] = sv ? spread16(v - (1u << k)) << 2 | 2 | (sv < 0) : 1;
        vlc_len[i] = sv ? 2 * k + 2 : 1;
    }
    for (int a = -VLC_P; a <= VLC_P; a++)
        for (int b = -VLC_P; b <= VLC_P; b++) {
            int i = (a + VLC_P) * (2 * VLC_P + 1) + b + VLC_P, ia = a + VLC_R, ib = b + VLC_R;
            vlc2_code[i] = (uint16_t)(vlc_code[ia] << vlc_len[ib] | vlc_code[ib]);
            vlc2_len[i] = vlc_len[ia] + vlc_len[ib];
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
                    const int32_t *row = p->band[l][o] ? p->band[l][o] + (size_t)y * p->g.bw[l] : 0;
                    const int16_t *row16 = p->band16[l][o] ? p->band16[l][o] + (size_t)y * p->g.bw[l] : 0;
                    const int *xs = p->xs[l];
                    for (int sx = 0; sx < nx; sx++) {
                        int si = sy * nx + sx, q = e->sq[si] - e->qm[l][o];
                        if (n >= e->cut[e->sq[si]][si * 3 + c]) continue;
                        int32_t *gv = gc[c] + per_slice * sx + off;
                        // an all-zero row is marked in its spare last entry and written as a run of ones
                        int wd = xs[sx + 1] - xs[sx];
                        gv[wd] = row16 ? quant_row16(row16 + xs[sx], wd, e->r16[q < 0 ? 0 : q], gv) : quant_row(row + xs[sx], wd, e->r16[q < 0 ? 0 : q], gv);
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
                    // short codes are collected in a local word (at most 16 bits per step, stored once it holds over 40)
                    uint64_t la = 0;
                    int ln = 0;
                    for (int x = 0; x < w; x++) {
                        uint32_t pa = (uint32_t)(vq[x] + VLC_P), pb = (uint32_t)(vq[x + 1] + VLC_P);
                        if (x + 1 < w && pa <= 2 * VLC_P && pb <= 2 * VLC_P) { // the common case: two small values
                            int j = pa * (2 * VLC_P + 1) + pb;
                            la = la << vlc2_len[j] | vlc2_code[j];
                            ln += vlc2_len[j];
                            if (ln > 40) { put(&b, la, ln); la = 0; ln = 0; }
                            x++;
                            continue;
                        }
                        uint32_t i = (uint32_t)(vq[x] + VLC_R);
                        if (__builtin_expect(i < 2 * VLC_R, 1)) {
                            la = la << vlc_len[i] | vlc_code[i];
                            ln += vlc_len[i];
                            if (ln > 40) { put(&b, la, ln); la = 0; ln = 0; }
                            continue;
                        }
                        if (ln) { put(&b, la, ln); la = 0; ln = 0; }
                        uint32_t v = (uint32_t)(vq[x] < 0 ? -vq[x] : vq[x]) + 1;
                        int kk = 31 - __builtin_clz(v);
                        for (int j = kk - 1; j >= 0; j--) put(&b, (v >> j) & 1, 2);
                        put(&b, 2 | (vq[x] < 0), 2);
                    }
                    if (ln) put(&b, la, ln);
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
    e->gpu_frame = 0;
    e->gcount = 0;
    if (e->gpu && !gpu_wavelet(e)) e->gpu_frame = 1;
    else if (e->gpu) {
        fprintf(stderr, "vc2: the GPU front end failed, on the CPU from now on\n");
        vc2_encgpu_free(e->gpu);
        e->gpu = 0;
        for (int c = 0; c < 3; c++) for (int l = 0; l <= VC2_MAX_DEPTH; l++) for (int o = 0; o < 4; o++) e->pl[c].band16[l][o] = e->pl[c].own16[l][o];
    }
    if (!e->gpu) vc2_pool_run(e->pool, plane_job, e, 3);
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
    // a slice larger than the cap gets a coarser quantiser until it fits
    for (int i = 0; e->c.max_slice_bytes > 0 && i < e->nslices; i++) {
        int row = i / e->nx;
        while (e->sq[i] < VC2_NQ - 1 && slice_size(e->cb[e->sq[i]] + (size_t)i * 3, e->scaler) > (size_t)e->c.max_slice_bytes) {
            e->sq[i]++;
            eval_q(e, e->sq[i], 0, 0, &row, 1);
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
