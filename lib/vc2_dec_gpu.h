// The decoder's side of the GPU path; included by vc2_dec.c (it uses the decoder's private types).
//
// With the GPU path on, the cascades do not run the last wavelet level. Its inputs (the LL coming out of the
// level above and the dequantised HL / LH / HH) go into whole-frame planes shared with the GPU instead of
// ring buffers, and the transform is launched in blocks of band rows while the CPU decodes on:
//  - a thread owns the LL rows of its band of slice rows (the rows above come out of its warm-up and are
//    wrong, so it does not write them); HL / LH / HH rows are right whoever writes them;
//  - a thread launches the blocks that read only rows it has written itself; the few blocks across the
//    border of two threads' bands are launched when all threads are done.
#include "vc2_gpu.h"
#include <limits.h>
#include <stdio.h>

struct dec_gpu {
    int req;                     // VC2_GPU_*
    const vc2_gpu_api *api;
    vc2_gpu *g;
    vc2_gpu_bufs b;
    int batch;                   // blocks per launch
    int depth;                   // of the configured stream
    int nunits, nblk[3];
    uint8_t *launched[3];        // per unit and block
    int nv20;                    // this frame's output
    int frame_open;
    int fd;                      // NV20 target as a dma-buf (-1 = none)
    size_t y_off, c_off;
    ptrdiff_t pitch;
    int cpus[64], ncpus;
    double t_gpu;
    int warned;
};

static int gpu_pow2ceil(int v) { int r = 1; while (r < v) r <<= 1; return r; }

// the LL plane of the last level, as the cascade's pointer and the ring size of the geometry
static void **gpu_ll_slot(vc2_dec *d, casc_t *k, int c) { return d->pic.depth > 1 ? &k->ll[c][d->pic.depth - 1] : &k->band[c][0][0]; }
static int *gpu_ll_rows(vc2_dec *d, int c) { return d->pic.depth > 1 ? &d->pl[c].lrows[d->pic.depth - 1] : &d->pl[c].brows[0]; }

static void gpu_unconfigure(vc2_dec *d)
{
    struct dec_gpu *G = d->gpu;
    if (!G || !d->gpu_on) return;
    for (int i = 0; d->cs && i < d->threads; i++) // the shared planes are the backend's
        for (int c = 0; c < 3; c++) {
            casc_t *k = &d->cs[i];
            if (G->depth > 1) k->ll[c][G->depth - 1] = 0; else k->band[c][0][0] = 0;
            for (int o = 1; o < 4; o++) k->band[c][G->depth][o] = 0;
        }
    G->api->destroy(G->g);
    G->g = 0;
    for (int u = 0; u < 3; u++) { free(G->launched[u]); G->launched[u] = 0; }
    d->gpu_on = 0;
}

// at the end of configure(): the stream's geometry is known and the cascades have their buffers
static void gpu_configure(vc2_dec *d)
{
    struct dec_gpu *G = d->gpu;
    const int depth = d->pic.depth;
    if (!G || !G->req || !G->api) return;
    const char *why = 0;
    if (!d->fin16) why = "needs the 16-bit last level (10-bit video, wavelets 0-4)";
    else if (d->pic.wavelet != VC2_WL_LEGALL5_3 && d->pic.wavelet != VC2_WL_DD9_7) why = "only LeGall 5,3 and Deslauriers-Dubuc 9,7";
    if (!why) {
        vc2_gpu_cfg cfg = { d->pic.wavelet, d->seq.bit_depth, 3, { 0 }, { 0 }, { 0 }, { 0 }, d->threads, G->cpus, G->ncpus };
        for (int c = 0; c < 3; c++) {
            cfg.bw[c] = d->pl[c].g.bw[depth]; cfg.bh[c] = d->pl[c].g.bh[depth];
            cfg.w[c] = d->pl[c].g.w; cfg.h[c] = d->pl[c].g.h;
        }
        G->g = G->api->create(&cfg, &G->b);
        if (!G->g) why = "the backend could not start";
    }
    if (why) {
        if (!G->warned) fprintf(stderr, "vc2: GPU path (%s) not used, %s; decoding on the CPU\n", G->api->name, why);
        G->warned = 1;
        return;
    }
    for (int c = 0; c < 3; c++) {
        comp_t *p = &d->pl[c];
        for (int i = 0; i < d->threads; i++) {
            casc_t *k = &d->cs[i];
            void **ll = gpu_ll_slot(d, k, c);
            free(*ll); *ll = G->b.band[c][0];
            for (int o = 1; o < 4; o++) { free(k->band[c][depth][o]); k->band[c][depth][o] = G->b.band[c][o]; }
        }
        *gpu_ll_rows(d, c) = p->brows[depth] = gpu_pow2ceil(p->g.bh[depth]); // whole planes: the ring never wraps
        G->nblk[c] = (p->g.bh[depth] + G->b.rb - 1) / G->b.rb;
        G->launched[c] = calloc(G->nblk[c], 1);
    }
    G->batch = getenv("VC2_GPU_BATCH") ? atoi(getenv("VC2_GPU_BATCH")) : 4;
    if (G->batch < 1) G->batch = 1;
    G->depth = depth;
    d->gpu_on = 1;
}

// NV20 through the GPU needs whole tiles of 16 luma samples and Cb / Cr of the same size
static int gpu_nv20_ok(const vc2_dec *d)
{
    const comp_t *p = d->pl;
    return d->seq.chroma == 1 && d->seq.bit_depth == 10 && p[0].g.w == p[0].g.pw && p[0].g.w % 16 == 0;
}

static void gpu_casc_start(vc2_dec *d, casc_t *k, int r0, int r1);
static int gpu_begin(vc2_dec *d, vc2_frame *out)
{
    struct dec_gpu *G = d->gpu;
    G->nv20 = d->nv20_y || G->fd >= 0;
    if (G->nv20 && !gpu_nv20_ok(d)) return -1;
    vc2_gpu_target t = { G->nv20, G->fd, G->y_off, G->c_off, d->nv20_y, d->nv20_c, G->fd >= 0 ? G->pitch : d->nv20_pitch };
    if (G->api->begin(G->g, &t)) return -1;
    G->nunits = G->nv20 ? 2 : 3;
    for (int u = 0; u < G->nunits; u++) memset(G->launched[u], 0, G->nblk[u]);
    G->frame_open = 1;
    gpu_casc_start(d, &d->cs[0], 0, d->pic.slices_y);
    for (int c = 0; out && c < 3; c++) { out->plane[c] = G->b.pix[c]; out->stride[c] = G->b.pstride[c]; }
    return 0;
}

static void gpu_wrote_rows(vc2_dec *d, int c, int r0, int r1)
{
    struct dec_gpu *G = d->gpu;
    const int bw = d->pl[c].g.bw[d->pic.depth];
    if (r1 <= r0) return;
    for (int o = 0; o < 4; o++) G->api->wrote(G->g, G->b.band[c][o] + (size_t)r0 * bw, (size_t)(r1 - r0) * bw * 2);
}

// a cascade starts: which LL rows it owns, and the first block it can launch by itself
static void gpu_casc_start(vc2_dec *d, casc_t *k, int r0, int r1)
{
    struct dec_gpu *G = d->gpu;
    const int depth = d->pic.depth, rb = G->b.rb;
    for (int c = 0; c < 3; c++) {
        const comp_t *p = &d->pl[c];
        k->ll0[c] = p->ys[depth][r0];
        k->ll1[c] = r1 >= d->pic.slices_y ? INT_MAX : p->ys[depth][r1];
        int first = k->ll0[c] ? (k->ll0[c] + d->up + rb - 1) / rb : 0;
        k->gblk[c] = first;
        int row = first * rb - d->up, end = k->ll1[c] < p->g.bh[depth] ? k->ll1[c] : p->g.bh[depth];
        row = row < k->ll0[c] ? k->ll0[c] : row > end ? end : row;
        k->gcl0[c] = k->gcl1[c] = row;
    }
}

// after the cascade has advanced: launch the blocks whose inputs it has written (final: all it will ever have)
static void gpu_progress(vc2_dec *d, casc_t *k, int final)
{
    struct dec_gpu *G = d->gpu;
    const int depth = d->pic.depth, rb = G->b.rb, thread = (int)(k - d->cs);
    if (!G->frame_open) return;
    for (int u = 0; u < G->nunits; u++) {
        const int c0 = G->nv20 && u ? 1 : u, c1 = G->nv20 && u ? 2 : u, bh = d->pl[c0].g.bh[depth];
        int rows = k->syn_done[c0][depth] < k->syn_done[c1][depth] ? k->syn_done[c0][depth] : k->syn_done[c1][depth];
        if (k->ll1[c0] != INT_MAX && rows > k->ll1[c0] - d->down) rows = k->ll1[c0] - d->down; // below: the next thread's rows
        int g1 = rows >= bh ? G->nblk[u] : rows / rb, g0 = k->gblk[u];
        if (g1 - g0 < (final ? 1 : G->batch)) continue;
        int need = g1 * rb + d->down < bh ? g1 * rb + d->down : bh;
        for (int c = c0; c <= c1; c++)
            if (need > k->gcl1[c]) { gpu_wrote_rows(d, c, k->gcl1[c], need); k->gcl1[c] = need; }
        if (G->api->launch(G->g, thread, u, g0, g1)) continue; // left for gpu_finish, which reports the error
        memset(G->launched[u] + g0, 1, g1 - g0);
        k->gblk[u] = g1;
    }
    if (!final) return;
    for (int c = 0; c < 3; c++) { // the rest of the rows this cascade owns
        int end = k->ll1[c] < d->pl[c].g.bh[depth] ? k->ll1[c] : d->pl[c].g.bh[depth];
        gpu_wrote_rows(d, c, k->ll0[c], k->gcl0[c]);
        gpu_wrote_rows(d, c, k->gcl1[c], end);
        k->gcl0[c] = k->ll0[c]; k->gcl1[c] = end > k->gcl1[c] ? end : k->gcl1[c];
    }
}

// all cascades are done: launch what is left (the blocks across the threads' borders) and wait
static int gpu_finish(vc2_dec *d)
{
    struct dec_gpu *G = d->gpu;
    int rc = 0;
    if (!G->frame_open) return 0;
    G->frame_open = 0;
    for (int u = 0; u < G->nunits; u++)
        for (int g = 0; g < G->nblk[u];) {
            if (G->launched[u][g]) { g++; continue; }
            int e = g;
            while (e < G->nblk[u] && !G->launched[u][e]) e++;
            rc |= G->api->launch(G->g, 0, u, g, e);
            g = e;
        }
    rc |= G->api->finish(G->g, &G->t_gpu);
    return rc ? -1 : 0;
}

int vc2_dec_set_gpu(vc2_dec *d, int backend, const int *cpus, int ncpus)
{
    const vc2_gpu_api *api = 0;
#ifdef VC2_OPENCL
    if (backend == VC2_GPU_OPENCL) api = &vc2_cl_api;
#endif
#ifdef VC2_VULKAN
    if (backend == VC2_GPU_VULKAN) api = &vc2_vk_api;
#endif
    if (backend != VC2_GPU_OFF && !api) return -1;
    if (!d->gpu) { d->gpu = calloc(1, sizeof *d->gpu); d->gpu->fd = -1; }
    struct dec_gpu *G = d->gpu;
    G->req = backend; G->api = api ? api : G->api;
    G->ncpus = cpus ? (ncpus > 64 ? 64 : ncpus) : 0;
    for (int i = 0; i < G->ncpus; i++) G->cpus[i] = cpus[i];
    G->warned = 0;
    d->configured = 0; // the buffers change: set up again at the next frame
    return 0;
}
void vc2_dec_set_nv20_dmabuf(vc2_dec *d, int fd, size_t y_offset, size_t cbcr_offset, ptrdiff_t pitch)
{
    if (!d->gpu) { d->gpu = calloc(1, sizeof *d->gpu); d->gpu->fd = -1; }
    d->gpu->fd = fd; d->gpu->y_off = y_offset; d->gpu->c_off = cbcr_offset; d->gpu->pitch = pitch;
}
int vc2_dec_gpu_active(const vc2_dec *d) { return d->gpu_on ? d->gpu->req : VC2_GPU_OFF; }
double vc2_dec_gpu_time(const vc2_dec *d) { return d->gpu_on ? d->gpu->t_gpu : 0; }
