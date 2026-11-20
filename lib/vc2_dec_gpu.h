// The decoder's side of the GPU path; included by vc2_dec.c (it uses the decoder's private types).
//
// With the GPU path on, the cascades do not run the last wavelet level. Its inputs (the LL coming out of the
// level above and the dequantised HL / LH / HH) go into whole-frame planes shared with the GPU instead of
// ring buffers, and the transform is launched in blocks of band rows while the CPU decodes on:
//  - a thread owns the LL rows of its band of slice rows (the rows above come out of its warm-up and are
//    wrong, so it does not write them); HL / LH / HH rows are right whoever writes them;
//  - a thread launches the blocks that read only rows it has written itself; when it is done it also
//    launches the few blocks across the border to the next thread's band, if that thread has written the rows
//    they read (it does so at its start); what is left is launched when all threads are done.
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
    int nblk;
    uint8_t *launched;           // per block
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
    free(G->launched); G->launched = 0;
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
    else if (d->seq.chroma == 2) why = "not for 4:2:0";
    if (!why) {
        vc2_gpu_cfg cfg = { d->pic.wavelet, d->seq.bit_depth, 3, { 0 }, d->pl[0].g.bh[depth], { 0 }, d->pl[0].g.h, d->threads, G->cpus, G->ncpus };
        for (int c = 0; c < 3; c++) { cfg.bw[c] = d->pl[c].g.bw[depth]; cfg.w[c] = d->pl[c].g.w; }
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
    }
    G->nblk = (d->pl[0].g.bh[depth] + G->b.rb - 1) / G->b.rb;
    G->launched = calloc(G->nblk, 1);
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
    memset(G->launched, 0, G->nblk);
    G->frame_open = 1;
    for (int i = 0; i < d->threads; i++) __atomic_store_n(&d->cs[i].gpub, 0, __ATOMIC_RELEASE);
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

// the first block a cascade starting its own rows at band row ll0 can launch by itself
static int gpu_first_block(const vc2_dec *d, int ll0) { return ll0 ? (ll0 + d->up + d->gpu->b.rb - 1) / d->gpu->b.rb : 0; }

// a cascade starts: which LL rows it owns, and where it starts launching
static void gpu_casc_start(vc2_dec *d, casc_t *k, int r0, int r1)
{
    const int depth = d->pic.depth;
    for (int c = 0; c < 3; c++) { // the same rows for every component: the band heights are equal
        k->ll0[c] = d->pl[c].ys[depth][r0];
        k->ll1[c] = r1 >= d->pic.slices_y ? INT_MAX : d->pl[c].ys[depth][r1];
    }
    k->gblk = gpu_first_block(d, k->ll0[0]);
    k->gcl1 = k->ll0[0];
    __atomic_store_n(&k->gpub, 0, __ATOMIC_RELEASE);
}

static void gpu_launch(vc2_dec *d, casc_t *k, int g1, int rows_to)
{
    struct dec_gpu *G = d->gpu;
    if (rows_to > k->gcl1) { for (int c = 0; c < 3; c++) gpu_wrote_rows(d, c, k->gcl1, rows_to); k->gcl1 = rows_to; }
    if (!G->api->launch(G->g, (int)(k - d->cs), k->gblk, g1)) { // on an error the blocks are left for gpu_finish, which reports it
        memset(G->launched + k->gblk, 1, g1 - k->gblk);
        k->gblk = g1;
    }
}

// after the cascade has advanced: launch the blocks whose inputs it has written (final: all it will ever have)
static void gpu_progress(vc2_dec *d, casc_t *k, int final)
{
    struct dec_gpu *G = d->gpu;
    const int depth = d->pic.depth, rb = G->b.rb, bh = d->pl[0].g.bh[depth];
    if (!G->frame_open) return;
    int rows = k->syn_done[0][depth];
    for (int c = 1; c < 3; c++) if (k->syn_done[c][depth] < rows) rows = k->syn_done[c][depth];
    __atomic_store_n(&k->gpub, rows, __ATOMIC_RELEASE);
    if (k->ll1[0] != INT_MAX && rows > k->ll1[0] - d->down) rows = k->ll1[0] - d->down; // below: the next thread's rows
    const int g1 = rows >= bh ? G->nblk : rows / rb;
    if (g1 - k->gblk >= (final ? 1 : G->batch)) gpu_launch(d, k, g1, g1 * rb + d->down < bh ? g1 * rb + d->down : bh);
    if (!final) return;
    const int end = k->ll1[0] < bh ? k->ll1[0] : bh; // the rest of the rows this cascade owns
    if (end > k->gcl1) { for (int c = 0; c < 3; c++) gpu_wrote_rows(d, c, k->gcl1, end); k->gcl1 = end; }
    if (k->ll1[0] < bh && k + 1 < d->cs + d->nchunks) { // the blocks across the border, up to the next thread's first
        const int g2 = gpu_first_block(d, k->ll1[0]), to = g2 * rb + d->down < bh ? g2 * rb + d->down : bh;
        if (g2 > k->gblk && __atomic_load_n(&k[1].gpub, __ATOMIC_ACQUIRE) >= (g2 * rb < bh ? g2 * rb : bh)) gpu_launch(d, k, g2, to);
    }
}

// all cascades are done: launch what is left (the blocks across the threads' borders) and wait
static int gpu_finish(vc2_dec *d)
{
    struct dec_gpu *G = d->gpu;
    int rc = 0;
    if (!G->frame_open) return 0;
    G->frame_open = 0;
    for (int g = 0; g < G->nblk;) {
        if (G->launched[g]) { g++; continue; }
        int e = g;
        while (e < G->nblk && !G->launched[e]) e++;
        rc |= G->api->launch(G->g, 0, g, e);
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
