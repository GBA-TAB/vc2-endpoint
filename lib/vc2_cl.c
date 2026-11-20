// OpenCL backend of the GPU path, for Mali with the vendor driver (libmali): the kernel of vc2_idwt.cl on
// buffers from the dma-heap, imported as dma-bufs (cl_arm_import_memory_dma_buf). The driver does no cache
// maintenance for those, so nothing walks the buffers inside the queue: the CPU threads clean the cache
// lines they have written (DC CVAC) before they launch, and the planar output is dropped from the CPU's
// caches (DC CIVAC) before it is read. An NV20 target that is a dma-buf (a scanout buffer) is written
// directly; one given as host pointers is filled through an internal buffer and a copy.
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include "vc2_gpu.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// the kernel source, from the file next to this one
__asm__(".pushsection .rodata\n.global vc2_idwt_cl_src\nvc2_idwt_cl_src:\n.incbin \"" VC2_IDWT_CL "\"\n.byte 0\n.popsection\n");
extern const char vc2_idwt_cl_src[];

#define MAX_EV 512
#define MAX_IMPORT 8
#define LANES 8
#define TW 4       // column pairs per work-item, as in the kernel for LANES = 8
#define RB 16      // band rows per work-item

typedef cl_mem (*import_fn)(cl_context, cl_mem_flags, const cl_import_properties_arm *, void *, size_t, cl_int *);
typedef struct { int fd; void *map; size_t size; cl_mem mem; } dbuf;

struct vc2_gpu {
    vc2_gpu_cfg cfg;
    cl_context ctx;
    cl_device_id dev;
    cl_command_queue q;
    import_fn import;
    cl_program prog[3];              // planar, NV20 Y, NV20 CbCr
    cl_kernel (*kern)[3];            // per thread
    cl_event (*ev)[MAX_EV];          // per thread: this frame's launches
    int *nev;
    dbuf coef, pix, nv20;            // nv20: only for host-pointer targets
    int off[3][4];                   // band planes within coef, in int16
    int ooff[3], ostride[3];         // planar output within pix, in uint16
    struct { int fd; cl_mem mem; } imp[MAX_IMPORT];
    int nimp;
    vc2_gpu_target t;
    cl_mem out;                      // this frame's target
    int y_off, c_off, pitch;
    int failed;
};

static int heap_buf(vc2_gpu *g, dbuf *b, size_t size)
{
    int h = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (h < 0) { perror("vc2: /dev/dma_heap/system"); return -1; }
    struct dma_heap_allocation_data a = { .len = size, .fd_flags = O_RDWR | O_CLOEXEC };
    int rc = ioctl(h, DMA_HEAP_IOCTL_ALLOC, &a);
    close(h);
    if (rc) { perror("vc2: dma-heap allocation"); return -1; }
    b->fd = a.fd; b->size = size;
    b->map = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
    if (b->map == MAP_FAILED) { b->map = 0; return -1; }
    memset(b->map, 0, size);
    cl_int err;
    const cl_import_properties_arm props[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0 };
    b->mem = g->import(g->ctx, CL_MEM_READ_WRITE, props, &b->fd, size, &err);
    return err ? -1 : 0;
}
static void heap_free(dbuf *b)
{
    if (b->mem) clReleaseMemObject(b->mem);
    if (b->map) munmap(b->map, b->size);
    if (b->map || b->mem) close(b->fd);
    memset(b, 0, sizeof *b);
}

// cache maintenance from user space: write the lines back (clean), or write back and drop them (flush)
static void cache_range(const void *p, size_t n, int drop)
{
#ifdef __aarch64__
    const char *q = (const char *)((uintptr_t)p & ~(uintptr_t)63), *e = (const char *)p + n;
    if (drop) for (; q < e; q += 64) __asm__ volatile("dc civac, %0" : : "r"(q));
    else for (; q < e; q += 64) __asm__ volatile("dc cvac, %0" : : "r"(q));
    __asm__ volatile("dsb ish");
#else
    (void)p; (void)n; (void)drop;
#endif
}

static void cl_destroy(vc2_gpu *g)
{
    if (!g) return;
    if (g->q) clFinish(g->q);
    for (int i = 0; g->kern && i < g->cfg.threads; i++)
        for (int o = 0; o < 3; o++) if (g->kern[i][o]) clReleaseKernel(g->kern[i][o]);
    for (int o = 0; o < 3; o++) if (g->prog[o]) clReleaseProgram(g->prog[o]);
    for (int i = 0; i < g->nimp; i++) clReleaseMemObject(g->imp[i].mem);
    heap_free(&g->coef); heap_free(&g->pix); heap_free(&g->nv20);
    if (g->q) clReleaseCommandQueue(g->q);
    if (g->ctx) clReleaseContext(g->ctx);
    free(g->kern); free(g->ev); free(g->nev);
    free(g);
}

static vc2_gpu *cl_create(const vc2_gpu_cfg *cfg, vc2_gpu_bufs *bufs)
{
    vc2_gpu *g = calloc(1, sizeof *g);
    cl_platform_id plat;
    cl_int err;
    cpu_set_t old, set;
    g->cfg = *cfg;
    // the driver starts its threads from here: let them inherit the cores they are meant for
    sched_getaffinity(0, sizeof old, &old);
    if (cfg->ncpus) {
        CPU_ZERO(&set);
        for (int i = 0; i < cfg->ncpus; i++) CPU_SET(cfg->cpus[i], &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    if (clGetPlatformIDs(1, &plat, 0) || clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &g->dev, 0)) goto fail;
    g->import = (import_fn)clGetExtensionFunctionAddressForPlatform(plat, "clImportMemoryARM");
    if (!g->import) { fprintf(stderr, "vc2: this OpenCL has no clImportMemoryARM\n"); goto fail; }
    g->ctx = clCreateContext(0, 1, &g->dev, 0, 0, &err);
    if (err) goto fail;
    const cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    g->q = clCreateCommandQueueWithProperties(g->ctx, g->dev, qp, &err);
    if (err) goto fail;

    // buffers: every band plane with room before and after for the kernel's lane loads
    size_t pos = 2048;
    for (int c = 0; c < cfg->ncomp; c++)
        for (int o = 0; o < 4; o++) { g->off[c][o] = (int)pos; pos += ((size_t)cfg->bw[c] * cfg->bh[c] + 64 + 31) & ~(size_t)31; }
    if (heap_buf(g, &g->coef, (pos * 2 + 8191) & ~(size_t)4095)) goto fail;
    pos = 0;
    for (int c = 0; c < cfg->ncomp; c++) {
        g->ostride[c] = (2 * cfg->bw[c] + 15) & ~15;
        g->ooff[c] = (int)pos;
        pos += (size_t)g->ostride[c] * 2 * cfg->bh[c] + 32;
    }
    if (heap_buf(g, &g->pix, (pos * 2 + 8191) & ~(size_t)4095)) goto fail;
    for (int c = 0; c < cfg->ncomp; c++) {
        for (int o = 0; o < 4; o++) bufs->band[c][o] = (int16_t *)g->coef.map + g->off[c][o];
        bufs->pix[c] = (uint16_t *)g->pix.map + g->ooff[c];
        bufs->pstride[c] = g->ostride[c];
    }
    bufs->rb = RB;

    for (int o = 0; o < 3; o++) {
        char opt[200];
        const char *src = vc2_idwt_cl_src;
        if (o && cfg->bit_depth != 10) break; // NV20 is 10-bit
        snprintf(opt, sizeof opt, "-DWL=%d -DLANES=%d -DRB=%d -DOUT=%d -DPIX_OFF=%d -DPIX_MAX=%d", cfg->wavelet == VC2_WL_LEGALL5_3, LANES, RB, o,
                 1 << (cfg->bit_depth - 1), (1 << cfg->bit_depth) - 1);
        g->prog[o] = clCreateProgramWithSource(g->ctx, 1, &src, 0, &err);
        if (err || clBuildProgram(g->prog[o], 1, &g->dev, opt, 0, 0)) {
            static char log[16384];
            log[0] = 0;
            if (g->prog[o]) clGetProgramBuildInfo(g->prog[o], g->dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, 0);
            fprintf(stderr, "vc2: OpenCL build failed (%s):\n%s\n", opt, log);
            goto fail;
        }
    }
    g->kern = calloc(cfg->threads, sizeof *g->kern);
    g->ev = calloc(cfg->threads, sizeof *g->ev);
    g->nev = calloc(cfg->threads, sizeof *g->nev);
    for (int i = 0; i < cfg->threads; i++)
        for (int o = 0; o < 3 && g->prog[o]; o++) {
            g->kern[i][o] = clCreateKernel(g->prog[o], "idwt", &err);
            if (err) goto fail;
        }
    { // one small launch, so that whatever the driver starts lazily starts now
        cl_kernel k = g->kern[0][0];
        cl_int o4[4] = { g->off[0][0], g->off[0][1], g->off[0][2], g->off[0][3] }, v[5] = { cfg->bw[0], cfg->bh[0], g->ooff[0], g->ostride[0], 2 * cfg->bh[0] };
        size_t gws[2] = { 1, 1 };
        clSetKernelArg(k, 0, sizeof(cl_mem), &g->coef.mem); clSetKernelArg(k, 1, sizeof(cl_mem), &g->pix.mem);
        clSetKernelArg(k, 2, sizeof o4, o4); clSetKernelArg(k, 3, sizeof o4, o4);
        for (int i = 0; i < 5; i++) clSetKernelArg(k, 4 + i, sizeof(cl_int), &v[i]);
        if (clEnqueueNDRangeKernel(g->q, k, 2, 0, gws, 0, 0, 0, 0) || clFinish(g->q)) goto fail;
    }
    sched_setaffinity(0, sizeof old, &old);
    return g;
fail:
    sched_setaffinity(0, sizeof old, &old);
    cl_destroy(g);
    return 0;
}

static cl_mem import_fd(vc2_gpu *g, int fd)
{
    for (int i = 0; i < g->nimp; i++) if (g->imp[i].fd == fd) return g->imp[i].mem;
    off_t size = lseek(fd, 0, SEEK_END);
    if (size <= 0) return 0;
    cl_int err;
    const cl_import_properties_arm props[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0 };
    cl_mem m = g->import(g->ctx, CL_MEM_READ_WRITE, props, &fd, (size_t)size, &err);
    if (err) return 0;
    if (g->nimp == MAX_IMPORT) { // forget the oldest
        clReleaseMemObject(g->imp[0].mem);
        memmove(g->imp, g->imp + 1, (MAX_IMPORT - 1) * sizeof g->imp[0]);
        g->nimp--;
    }
    g->imp[g->nimp].fd = fd; g->imp[g->nimp].mem = m; g->nimp++;
    return m;
}

static int cl_begin(vc2_gpu *g, const vc2_gpu_target *t)
{
    const int h = g->cfg.h[0];
    g->t = *t;
    g->failed = 0;
    if (!t->nv20) { g->out = g->pix.mem; return 0; }
    if (!g->prog[1]) return -1;
    g->pitch = (int)t->pitch;
    if (t->fd >= 0) {
        g->out = import_fd(g, t->fd);
        g->y_off = (int)t->y_off; g->c_off = (int)t->c_off;
        return g->out ? 0 : -1;
    }
    size_t need = ((size_t)2 * t->pitch * h + 8191) & ~(size_t)4095;
    if (g->nv20.size < need) {
        heap_free(&g->nv20);
        if (heap_buf(g, &g->nv20, need)) return -1;
    }
    g->out = g->nv20.mem;
    g->y_off = 0; g->c_off = (int)t->pitch * h;
    return 0;
}

static void cl_wrote(vc2_gpu *g, const void *p, size_t bytes) { (void)g; cache_range(p, bytes, 0); }

static int cl_launch(vc2_gpu *g, int thread, int unit, int blk0, int blk1)
{
    const int nv20 = g->t.nv20, c = unit, kind = nv20 ? 1 + unit : 0;
    cl_kernel k = g->kern[thread][kind];
    cl_int o0[4], o1[4];
    for (int o = 0; o < 4; o++) { o0[o] = g->off[c][o]; o1[o] = g->off[nv20 && unit ? 2 : c][o]; }
    cl_int v[5] = { g->cfg.bw[c], g->cfg.bh[c], nv20 ? (unit ? g->c_off : g->y_off) : g->ooff[c], nv20 ? g->pitch : g->ostride[c], g->cfg.h[c] };
    cl_int err = clSetKernelArg(k, 0, sizeof(cl_mem), &g->coef.mem);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &g->out);
    err |= clSetKernelArg(k, 2, sizeof o0, o0);
    err |= clSetKernelArg(k, 3, sizeof o1, o1);
    for (int i = 0; i < 5; i++) err |= clSetKernelArg(k, 4 + i, sizeof(cl_int), &v[i]);
    const size_t gwo[2] = { 0, blk0 }, gws[2] = { (g->cfg.bw[c] + TW - 1) / TW, blk1 - blk0 };
    if (err || g->nev[thread] == MAX_EV || clEnqueueNDRangeKernel(g->q, k, 2, gwo, gws, 0, 0, 0, &g->ev[thread][g->nev[thread]])) {
        g->failed = 1;
        return -1;
    }
    g->nev[thread]++;
    clFlush(g->q);
    return 0;
}

static int cl_finish(vc2_gpu *g, double *gpu_seconds)
{
    int rc = g->failed ? -1 : 0;
    cl_ulong ns = 0;
    for (int i = 0; i < g->cfg.threads; i++) {
        if (g->nev[i] && clWaitForEvents(g->nev[i], g->ev[i])) rc = -1;
        for (int j = 0; j < g->nev[i]; j++) {
            cl_ulong s = 0, e = 0;
            clGetEventProfilingInfo(g->ev[i][j], CL_PROFILING_COMMAND_START, sizeof s, &s, 0);
            clGetEventProfilingInfo(g->ev[i][j], CL_PROFILING_COMMAND_END, sizeof e, &e, 0);
            ns += e - s;
            clReleaseEvent(g->ev[i][j]);
        }
        g->nev[i] = 0;
    }
    if (gpu_seconds) *gpu_seconds = ns * 1e-9;
    if (!g->t.nv20) cache_range(g->pix.map, g->pix.size, 1); // the CPU may hold lines of the last frame
    else if (g->t.fd < 0) {
        const size_t n = (size_t)g->t.pitch * g->cfg.h[0];
        cache_range(g->nv20.map, 2 * n, 1);
        memcpy(g->t.y, g->nv20.map, n);
        memcpy(g->t.c, (uint8_t *)g->nv20.map + n, n);
    }
    return rc;
}

const vc2_gpu_api vc2_cl_api = { "OpenCL", cl_create, cl_destroy, cl_begin, cl_wrote, cl_launch, cl_finish };
