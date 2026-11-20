// Tables, wavelet definitions, stream headers, thread pool.
#define _GNU_SOURCE
#include "vc2_priv.h"
#include <sched.h>
#include <stdatomic.h>

uint32_t vc2_qf[VC2_NQ], vc2_qo[VC2_NQ];
static pthread_once_t tables_once = PTHREAD_ONCE_INIT;

// quantisation factor = round(4 * 2^(q/4)), found exactly: the largest n with (2n-1)^4 <= 2^(q+12)
static void tables_make(void)
{
    for (int q = 0; q < VC2_NQ; q++) {
        uint64_t lo = 1, hi = (uint64_t)1 << 31;
        while (lo < hi) {
            uint64_t n = (lo + hi + 1) / 2, m = 2 * n - 1;
            unsigned __int128 m2 = (unsigned __int128)m * m;
            int le;
            if (q + 12 >= 128) le = 1;
            else if (m2 >> 64) le = 0;
            else { unsigned __int128 m4 = m2 * m2; le = m4 <= ((unsigned __int128)1 << (q + 12)); }
            if (le) lo = n; else hi = n - 1;
        }
        vc2_qf[q] = (uint32_t)lo;
        vc2_qo[q] = (q == 0 ? 1 : q == 1 ? 2 : (vc2_qf[q] + 1) / 2) + 2;
    }
}
void vc2_tables_init(void) { pthread_once(&tables_once, tables_make); }

// Default quantisation matrices, by wavelet index: LL, then HL/LH/HH for levels 1..4.
static const uint8_t def_qm[7][13] = {
    { 5, 3, 3, 0, 4, 4, 1, 5, 5, 2, 6, 6, 3 },
    { 4, 2, 2, 0, 4, 4, 2, 5, 5, 3, 7, 7, 5 },
    { 5, 3, 3, 0, 4, 4, 1, 5, 5, 2, 6, 6, 3 },
    { 8, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0 },
    { 8, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0 },
    { 0, 4, 4, 8, 8, 8, 12, 13, 13, 17, 17, 17, 21 },
    { 3, 1, 1, 0, 4, 4, 2, 6, 6, 5, 9, 9, 7 },
};
int vc2_default_qm(int wavelet, int depth, uint8_t qm[VC2_MAX_DEPTH + 1][4])
{
    if (wavelet < 0 || wavelet > 6 || depth < 1 || depth > 4) return -1;
    memset(qm, 0, (VC2_MAX_DEPTH + 1) * 4);
    for (int l = 0; l <= depth; l++)
        for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
            int v = l ? def_qm[wavelet][(l - 1) * 3 + o] : def_qm[wavelet][0];
            if (wavelet == VC2_WL_HAAR0) v += 4 * (depth - (l ? l : 1)); // without the shift the gain grows per level
            qm[l][o] = v;
        }
    return 0;
}

static const vc2_wavelet wavelets[7] = {
    [VC2_WL_DD9_7] = { 2, { { 0, VC2_K_AVG2, -1, 2, 1 }, { 1, VC2_K_DD4, 1, 4, 1 } }, 1, 1 },
    [VC2_WL_LEGALL5_3] = { 2, { { 0, VC2_K_AVG2, -1, 2, 1 }, { 1, VC2_K_AVG2, 1, 1, 1 } }, 1, 1 },
    [VC2_WL_DD13_7] = { 2, { { 0, VC2_K_DD4, -1, 5, 1 }, { 1, VC2_K_DD4, 1, 4, 1 } }, 1, 1 },
    [VC2_WL_HAAR0] = { 2, { { 0, VC2_K_ONE, -1, 1, 1 }, { 1, VC2_K_ONE, 1, 0, 1 } }, 0, 1 },
    [VC2_WL_HAAR1] = { 2, { { 0, VC2_K_ONE, -1, 1, 1 }, { 1, VC2_K_ONE, 1, 0, 1 } }, 1, 1 },
    [VC2_WL_DAUB9_7] = { 4, { { 0, VC2_K_MUL2, -1, 12, 1817 }, { 1, VC2_K_MUL2, -1, 12, 3616 },
                              { 0, VC2_K_MUL2, 1, 12, 217 }, { 1, VC2_K_MUL2, 1, 12, 6497 } }, 1, 0 },
};
const vc2_wavelet *vc2_wavelet_get(int idx)
{
    if (idx < 0 || idx > 6 || !wavelets[idx].nsteps) return 0;
    return &wavelets[idx];
}

void vc2_geom_init(vc2_geom *g, int w, int h, int depth)
{
    int m = (1 << depth) - 1;
    g->w = w; g->h = h;
    g->pw = (w + m) & ~m; g->ph = (h + m) & ~m;
    for (int l = 0; l <= depth; l++) {
        int s = l ? depth - l + 1 : depth;
        g->bw[l] = g->pw >> s; g->bh[l] = g->ph >> s;
    }
}

// --- header bit i/o ---------------------------------------------------------------------------------------
typedef struct { const uint8_t *p; size_t len, pos; } br_t;
static int br_bit(br_t *b)
{
    if (b->pos >= b->len * 8) return 1;
    int v = (b->p[b->pos >> 3] >> (7 - (b->pos & 7))) & 1;
    b->pos++;
    return v;
}
static uint32_t br_uint(br_t *b)
{
    uint32_t v = 1;
    for (int i = 0; i < 32 && !br_bit(b); i++) v = (v << 1) | br_bit(b);
    return v - 1;
}
static void br_align(br_t *b) { b->pos = (b->pos + 7) & ~(size_t)7; }
void vc2_bw_bit(vc2_bw *b, int v)
{
    if (!(b->pos & 7)) b->p[b->pos >> 3] = 0;
    if (v) b->p[b->pos >> 3] |= 0x80 >> (b->pos & 7);
    b->pos++;
}
void vc2_bw_uint(vc2_bw *b, uint32_t v)
{
    v++;
    int k = 31 - __builtin_clz(v);
    for (int i = k - 1; i >= 0; i--) { vc2_bw_bit(b, 0); vc2_bw_bit(b, (v >> i) & 1); }
    vc2_bw_bit(b, 1);
}
void vc2_bw_align(vc2_bw *b) { while (b->pos & 7) vc2_bw_bit(b, 0); }

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }
static void wr32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }

size_t vc2_write_parse_info(uint8_t *p, int code, uint32_t next, uint32_t prev)
{
    memcpy(p, "BBCD", 4);
    p[4] = code;
    wr32(p + 5, next); wr32(p + 9, prev);
    return 13;
}

size_t vc2_unit(const uint8_t *buf, size_t len, int *code)
{
    if (len < 13 || memcmp(buf, "BBCD", 4)) return 0;
    *code = buf[4];
    uint32_t next = rd32(buf + 5);
    if (next == 0) { // last unit of a stream: an end of sequence is just the header
        if (*code == VC2_PC_END_SEQ) return 13;
        // otherwise look for the next parse info
        for (size_t i = 13; i + 13 <= len; i++)
            if (!memcmp(buf + i, "BBCD", 4)) return i;
        return len;
    }
    return next <= len ? next : 0;
}

size_t vc2_access_unit(const uint8_t *buf, size_t len)
{
    size_t pos = 0;
    int code, pic = 0;
    while (pos < len) {
        size_t n = vc2_unit(buf + pos, len - pos, &code);
        if (!n) break;
        if (pic && code != VC2_PC_END_SEQ) break;
        pos += n;
        if (pic) break;
        if (code == VC2_PC_PICTURE_HQ) pic = 1;
    }
    return pic ? pos : 0;
}

// base video formats: width, height, chroma, interlaced, top field first, frame rate index, signal range index
static const uint16_t base_fmt[21][7] = {
    { 640, 480, 2, 0, 0, 1, 1 }, { 176, 120, 2, 0, 0, 9, 1 }, { 176, 144, 2, 0, 1, 10, 1 }, { 352, 240, 2, 0, 0, 9, 1 },
    { 352, 288, 2, 0, 1, 10, 1 }, { 704, 480, 2, 0, 0, 9, 1 }, { 704, 576, 2, 0, 1, 10, 1 }, { 720, 480, 1, 1, 0, 4, 3 },
    { 720, 576, 1, 1, 1, 3, 3 }, { 1280, 720, 1, 0, 1, 7, 3 }, { 1280, 720, 1, 0, 1, 6, 3 }, { 1920, 1080, 1, 1, 1, 4, 3 },
    { 1920, 1080, 1, 1, 1, 3, 3 }, { 1920, 1080, 1, 0, 1, 7, 3 }, { 1920, 1080, 1, 0, 1, 6, 3 }, { 2048, 1080, 0, 0, 1, 2, 4 },
    { 4096, 2160, 0, 0, 1, 2, 4 }, { 3840, 2160, 1, 0, 1, 7, 3 }, { 3840, 2160, 1, 0, 1, 6, 3 }, { 7680, 4320, 1, 0, 1, 7, 3 },
    { 7680, 4320, 1, 0, 1, 6, 3 },
};
static const int frame_rates[11][2] = { { 0, 1 }, { 24000, 1001 }, { 24, 1 }, { 25, 1 }, { 30000, 1001 }, { 30, 1 },
                                        { 50, 1 }, { 60000, 1001 }, { 60, 1 }, { 15000, 1001 }, { 25, 2 } };
static const int range_depth[5] = { 0, 8, 8, 10, 12 };

int vc2_parse_seq(const uint8_t *payload, size_t len, vc2_seq *s)
{
    br_t b = { payload, len, 0 };
    memset(s, 0, sizeof *s);
    s->major = br_uint(&b); s->minor = br_uint(&b); s->profile = br_uint(&b); s->level = br_uint(&b);
    s->base_format = br_uint(&b);
    if (s->base_format > 20) return -1;
    const uint16_t *f = base_fmt[s->base_format];
    s->width = f[0]; s->height = f[1]; s->chroma = f[2]; s->interlaced = f[3]; s->top_field_first = f[4];
    int fr = f[5], range = f[6];
    if (br_bit(&b)) { s->width = br_uint(&b); s->height = br_uint(&b); }
    if (br_bit(&b)) s->chroma = br_uint(&b);
    if (br_bit(&b)) s->interlaced = br_uint(&b);
    s->fr_num = frame_rates[fr][0]; s->fr_den = frame_rates[fr][1];
    if (br_bit(&b)) {
        fr = br_uint(&b);
        if (fr == 0) { s->fr_num = br_uint(&b); s->fr_den = br_uint(&b); }
        else if (fr <= 10) { s->fr_num = frame_rates[fr][0]; s->fr_den = frame_rates[fr][1]; }
        else { s->fr_num = 0; s->fr_den = 1; }
    }
    if (br_bit(&b)) { if (br_uint(&b) == 0) { br_uint(&b); br_uint(&b); } }           // pixel aspect ratio
    if (br_bit(&b)) { br_uint(&b); br_uint(&b); br_uint(&b); br_uint(&b); }          // clean area
    s->bit_depth = range_depth[range];
    if (br_bit(&b)) {
        range = br_uint(&b);
        if (range == 0) {
            br_uint(&b);
            uint32_t exc = br_uint(&b);
            br_uint(&b); br_uint(&b);
            s->bit_depth = 32 - __builtin_clz(exc | 1);
        } else if (range <= 4) s->bit_depth = range_depth[range];
        else return -1;
    }
    if (br_bit(&b)) {                                                                 // colour specification
        if (br_uint(&b) == 0) {
            if (br_bit(&b)) br_uint(&b);
            if (br_bit(&b)) br_uint(&b);
            if (br_bit(&b)) br_uint(&b);
        }
    }
    s->picture_coding_mode = br_uint(&b);
    if (s->chroma > 2 || s->width < 1 || s->height < 1 || s->width > 16384 || s->height > 16384) return -1;
    return 0;
}

size_t vc2_write_seq(uint8_t *p, const vc2_seq *s)
{
    vc2_bw b = { p, 0 };
    int base = 0;
    for (int i = 0; i < 21; i++)
        if (base_fmt[i][0] == s->width && base_fmt[i][1] == s->height && !base_fmt[i][3] &&
            frame_rates[base_fmt[i][5]][0] == s->fr_num && frame_rates[base_fmt[i][5]][1] == s->fr_den) base = i;
    vc2_bw_uint(&b, 2); vc2_bw_uint(&b, 0); vc2_bw_uint(&b, 3); vc2_bw_uint(&b, 0); // version 2.0, HQ profile, level 0
    vc2_bw_uint(&b, base);
    vc2_bw_bit(&b, 1); vc2_bw_uint(&b, s->width); vc2_bw_uint(&b, s->height);
    vc2_bw_bit(&b, 1); vc2_bw_uint(&b, s->chroma);
    vc2_bw_bit(&b, 1); vc2_bw_uint(&b, 0);                                           // progressive
    int fr = 0;
    for (int i = 1; i <= 10; i++)
        if (frame_rates[i][0] == s->fr_num && frame_rates[i][1] == s->fr_den) fr = i;
    vc2_bw_bit(&b, 1); vc2_bw_uint(&b, fr);
    if (!fr) { vc2_bw_uint(&b, s->fr_num); vc2_bw_uint(&b, s->fr_den); }
    vc2_bw_bit(&b, 0);                                                               // pixel aspect ratio: base
    vc2_bw_bit(&b, 0);                                                               // clean area: base
    vc2_bw_bit(&b, 1); vc2_bw_uint(&b, s->bit_depth == 8 ? 2 : s->bit_depth == 10 ? 3 : 4);
    vc2_bw_bit(&b, 0);                                                               // colour: base
    vc2_bw_uint(&b, 0);                                                              // pictures are frames
    vc2_bw_align(&b);
    return b.pos >> 3;
}

int vc2_parse_pic(const uint8_t *payload, size_t len, const vc2_seq *s, vc2_pic *p)
{
    if (len < 8) return -1;
    br_t b = { payload, len, 32 };
    memset(p, 0, sizeof *p);
    p->picture_number = rd32(payload);
    p->wavelet = br_uint(&b); p->depth = br_uint(&b);
    if (s->major >= 3) { // asymmetric transform parameters: not supported, must be off
        if (br_bit(&b)) br_uint(&b);
        if (br_bit(&b)) return -1;
    }
    p->slices_x = br_uint(&b); p->slices_y = br_uint(&b);
    p->prefix_bytes = br_uint(&b); p->size_scaler = br_uint(&b);
    if (!vc2_wavelet_get(p->wavelet) || p->depth < 1 || p->depth > VC2_MAX_DEPTH || p->slices_x < 1 || p->slices_y < 1 ||
        p->slices_x > 4096 || p->slices_y > 4096 || p->size_scaler < 1 || p->prefix_bytes > 1024) return -1;
    if (br_bit(&b)) {
        for (int l = 0; l <= p->depth; l++)
            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) p->qm[l][o] = br_uint(&b);
    } else if (vc2_default_qm(p->wavelet, p->depth, p->qm)) return -1;
    br_align(&b);
    p->slices_offset = b.pos >> 3;
    return p->slices_offset <= len ? 0 : -1;
}

// --- thread pool ----------------------------------------------------------------------------------------
struct vc2_pool {
    int n, quit;
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t cv, done_cv;
    vc2_job_fn fn;
    void *arg;
    int njobs, gen, running, each;
    atomic_int next;
    const int *cpus;
    int cpu_list[64];
};
struct worker_arg { vc2_pool *p; int id; };

static void work(vc2_pool *p, int id)
{
    if (p->each) { // one job per thread: job number = thread number
        if (id < p->njobs) p->fn(p->arg, id, id);
        return;
    }
    for (;;) {
        int j = atomic_fetch_add(&p->next, 1);
        if (j >= p->njobs) break;
        p->fn(p->arg, j, id);
    }
}
static void *worker(void *v)
{
    struct worker_arg *wa = v;
    vc2_pool *p = wa->p;
    int id = wa->id, gen = 0;
    free(wa);
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->quit && p->gen == gen) pthread_cond_wait(&p->cv, &p->mu);
        if (p->quit) break;
        gen = p->gen;
        pthread_mutex_unlock(&p->mu);
        work(p, id);
        pthread_mutex_lock(&p->mu);
        if (--p->running == 0) pthread_cond_signal(&p->done_cv);
    }
    pthread_mutex_unlock(&p->mu);
    return 0;
}
vc2_pool *vc2_pool_new(int threads, const int *cpus)
{
    vc2_pool *p = calloc(1, sizeof *p);
    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;
    p->n = threads;
    pthread_mutex_init(&p->mu, 0);
    pthread_cond_init(&p->cv, 0);
    pthread_cond_init(&p->done_cv, 0);
    p->th = calloc(threads, sizeof *p->th);
    for (int i = 1; i < threads; i++) {
        struct worker_arg *wa = malloc(sizeof *wa);
        wa->p = p; wa->id = i;
        pthread_create(&p->th[i], 0, worker, wa);
        if (cpus) {
            cpu_set_t set;
            CPU_ZERO(&set); CPU_SET(cpus[i], &set);
            pthread_setaffinity_np(p->th[i], sizeof set, &set);
        }
    }
    return p;
}
void vc2_pool_free(vc2_pool *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->quit = 1;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->th[i], 0);
    free(p->th);
    free(p);
}
int vc2_pool_threads(const vc2_pool *p) { return p->n; }
// run job i on thread i (n <= threads): for work shares sized per core
void vc2_pool_run_each(vc2_pool *p, vc2_job_fn fn, void *arg, int n)
{
    p->each = 1;
    vc2_pool_run(p, fn, arg, n > p->n ? p->n : n);
    p->each = 0;
}
void vc2_pool_run(vc2_pool *p, vc2_job_fn fn, void *arg, int n)
{
    if (n <= 0) return;
    if (p->n == 1 || (n == 1 && !p->each)) {
        for (int j = 0; j < n; j++) fn(arg, j, 0);
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->fn = fn; p->arg = arg; p->njobs = n;
    atomic_store(&p->next, 0);
    p->running = p->n - 1;
    p->gen++;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    work(p, 0);
    pthread_mutex_lock(&p->mu);
    while (p->running) pthread_cond_wait(&p->done_cv, &p->mu);
    pthread_mutex_unlock(&p->mu);
}
