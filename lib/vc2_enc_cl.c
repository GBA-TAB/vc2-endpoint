// The encoder's GPU front end in OpenCL (see vc2_enc_gpu.h): the kernels of vc2_fwd.cl on buffers the driver
// allocates (CL_MEM_ALLOC_HOST_PTR), mapped for the CPU around each frame: the input for writing before the
// launch, the bands for reading after it (the writer reads them in place). All 24 bands of the two levels lie in
// one buffer (sub-buffers for the wavelet kernel), so one launch counts all of them for a quantiser.
#define CL_TARGET_OPENCL_VERSION 300
#include "vc2_enc_gpu.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__asm__(".pushsection .rodata\n.global vc2_fwd_cl_src\nvc2_fwd_cl_src:\n.incbin \"" VC2_FWD_CL "\"\n.byte 0\n.popsection\n");
extern const char vc2_fwd_cl_src[];

#define RB 16    // band rows per work-item of the wavelet (Mali-G610: 8 1.56 ms, 16 1.39 ms, 32 2.36 ms per 1080p frame)
#define NSLOT 8  // counting results kept per frame (the rate control asks for a few quantisers)
#define ALIGN 4096

typedef struct { int q; cl_mem mem; uint32_t *map; } slot;

typedef cl_mem (*import_fn)(cl_context, cl_mem_flags, const cl_import_properties_arm *, void *, size_t, cl_int *);
struct vc2_encgpu {
    import_fn import;
    cl_mem raw;                            // the caller's picture, imported (cl_arm_import_memory_host)
    const void *raw_base;
    size_t raw_size;
    int raw_off[3], raw_stride[3], raw_h, raw_max, raw_offset, use_raw;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k, kc;
    cl_mem in[3], bands, sub4[3][4], sub3[3][4], desc, r16;
    int16_t *min[3], *mbands;
    size_t off4[3][4], off3[3][4], nbands; // in shorts
    int pw[3], ph, nx, mapped, nseg_max;
    int base[3][5][4];                     // first counting output index of each 16-bit band
    size_t nout;
    slot sl[NSLOT];
    int nsl;
    double ms;
};

static void unmap_slots(vc2_encgpu *g)
{
    for (int i = 0; i < g->nsl; i++) if (g->sl[i].map) { clEnqueueUnmapMemObject(g->q, g->sl[i].mem, g->sl[i].map, 0, 0, 0); g->sl[i].map = 0; }
    for (int i = 0; i < NSLOT; i++) g->sl[i].q = -1;
    g->nsl = 0;
}

void vc2_encgpu_release(vc2_encgpu *g)
{
    if (!g) return;
    unmap_slots(g);
    if (g->mapped) { clEnqueueUnmapMemObject(g->q, g->bands, g->mbands, 0, 0, 0); g->mapped = 0; }
}

void vc2_encgpu_free(vc2_encgpu *g)
{
    if (!g) return;
    vc2_encgpu_release(g);
    for (int c = 0; c < 3; c++) {
        if (g->min[c]) clEnqueueUnmapMemObject(g->q, g->in[c], g->min[c], 0, 0, 0);
        if (g->in[c]) clReleaseMemObject(g->in[c]);
        for (int b = 0; b < 4; b++) { if (g->sub4[c][b]) clReleaseMemObject(g->sub4[c][b]); if (g->sub3[c][b]) clReleaseMemObject(g->sub3[c][b]); }
    }
    for (int i = 0; i < NSLOT; i++) if (g->sl[i].mem) clReleaseMemObject(g->sl[i].mem);
    if (g->bands) clReleaseMemObject(g->bands);
    if (g->raw) clReleaseMemObject(g->raw);
    if (g->desc) clReleaseMemObject(g->desc);
    if (g->r16) clReleaseMemObject(g->r16);
    if (g->q) clFinish(g->q);
    if (g->k) clReleaseKernel(g->k);
    if (g->kc) clReleaseKernel(g->kc);
    if (g->prog) clReleaseProgram(g->prog);
    if (g->q) clReleaseCommandQueue(g->q);
    if (g->ctx) clReleaseContext(g->ctx);
    free(g);
}

static cl_mem sub(vc2_encgpu *g, size_t off, size_t n, cl_int *err)
{
    cl_buffer_region r = { off * 2, n * 2 + 64 };
    return clCreateSubBuffer(g->bands, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, err);
}

vc2_encgpu *vc2_encgpu_new(const int pw[3], int ph, int nx)
{
    for (int c = 0; c < 3; c++) if (pw[c] % 64 || pw[c] < 64) return 0;
    if (ph % 16 || nx < 1) return 0;
    vc2_encgpu *g = calloc(1, sizeof *g);
    for (int i = 0; i < NSLOT; i++) g->sl[i].q = -1;
    cl_platform_id pl; cl_device_id dev; cl_int err;
    if (clGetPlatformIDs(1, &pl, 0) || clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, 0)) goto fail;
    g->ctx = clCreateContext(0, 1, &dev, 0, 0, &err);
    if (err) goto fail;
    g->q = clCreateCommandQueueWithProperties(g->ctx, dev, (cl_queue_properties[]){ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 }, &err);
    if (err) goto fail;
    const char *src = vc2_fwd_cl_src;
    g->prog = clCreateProgramWithSource(g->ctx, 1, &src, 0, &err);
    if (err || clBuildProgram(g->prog, 1, &dev, "-cl-std=CL2.0", 0, 0)) {
        char log[8192] = "";
        if (g->prog) clGetProgramBuildInfo(g->prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, 0);
        fprintf(stderr, "vc2: GPU front end: kernel build failed\n%s\n", log);
        goto fail;
    }
    g->import = (import_fn)clGetExtensionFunctionAddressForPlatform(pl, "clImportMemoryARM");
    g->k = clCreateKernel(g->prog, "fwd53", &err);
    if (err) goto fail;
    g->kc = clCreateKernel(g->prog, "count16", &err);
    if (err) goto fail;
    g->ph = ph; g->nx = nx;
    // the band layout: level 4 (index 4) and level 3 of each component, every band at an aligned offset
    size_t off = 0;
    cl_int desc[18][4];
    int nd = 0;
    g->nout = 0;
    for (int c = 0; c < 3; c++) {
        g->pw[c] = pw[c];
        size_t n4 = (size_t)(pw[c] / 2) * (ph / 2), n3 = n4 / 4;
        for (int b = 0; b < 4; b++) {
            g->off4[c][b] = off; off += (n4 + 64 + ALIGN / 2 - 1) / (ALIGN / 2) * (ALIGN / 2);
            g->off3[c][b] = off; off += (n3 + 64 + ALIGN / 2 - 1) / (ALIGN / 2) * (ALIGN / 2);
        }
        for (int lv = 4; lv >= 3; lv--)
            for (int b = 1; b < 4; b++) {
                int bw = lv == 4 ? pw[c] / 2 : pw[c] / 4, bh = lv == 4 ? ph / 2 : ph / 4;
                desc[nd][0] = (cl_int)(lv == 4 ? g->off4[c][b] : g->off3[c][b]);
                desc[nd][1] = bw; desc[nd][2] = bh; desc[nd][3] = (cl_int)g->nout;
                g->base[c][lv][b] = (int)g->nout;
                g->nout += (size_t)bh * nx;
                if (bh * nx > g->nseg_max) g->nseg_max = bh * nx;
                nd++;
            }
    }
    g->nbands = off;
    g->bands = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, off * 2, 0, &err);
    if (err) goto fail;
    g->desc = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof desc, desc, &err);
    if (err) goto fail;
    g->r16 = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY, 18 * sizeof(float), 0, &err);
    if (err) goto fail;
    for (int c = 0; c < 3; c++) {
        size_t n4 = (size_t)(pw[c] / 2) * (ph / 2), n3 = n4 / 4;
        g->in[c] = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)pw[c] * ph * 2, 0, &err);
        if (err) goto fail;
        for (int b = 0; b < 4; b++) {
            g->sub4[c][b] = sub(g, g->off4[c][b], n4, &err);
            if (err) goto fail;
            g->sub3[c][b] = sub(g, g->off3[c][b], n3, &err);
            if (err) goto fail;
        }
    }
    return g;
fail:
    vc2_encgpu_free(g);
    return 0;
}

int vc2_encgpu_use_picture(vc2_encgpu *g, const uint16_t *const plane[3], const ptrdiff_t stride[3], int h, int bit_depth)
{
    g->use_raw = 0;
    if (!g->import || getenv("VC2_GPU_NO_IMPORT")) return -1;
    // one import covering the three planes, page-aligned (the pages around the picture belong to the caller too)
    uintptr_t lo = (uintptr_t)-1, hi = 0;
    for (int c = 0; c < 3; c++) {
        if (stride[c] < g->pw[c]) return -1; // the kernel reads whole padded rows: the picture must be that wide
        uintptr_t a = (uintptr_t)plane[c], b = a + ((size_t)(h - 1) * stride[c] + g->pw[c]) * 2;
        if (a & 1) return -1;
        lo = a < lo ? a : lo; hi = b > hi ? b : hi;
    }
    uintptr_t base = lo & ~(uintptr_t)4095, end = (hi + 4095) & ~(uintptr_t)4095;
    if ((const void *)base != g->raw_base || end - base != g->raw_size) {
        if (g->raw) { clReleaseMemObject(g->raw); g->raw = 0; }
        const cl_import_properties_arm props[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_HOST_ARM, 0 };
        cl_int err;
        g->raw = g->import(g->ctx, CL_MEM_READ_ONLY, props, (void *)base, end - base, &err);
        if (err || !g->raw) { g->raw = 0; g->raw_base = 0; return -1; }
        g->raw_base = (const void *)base; g->raw_size = end - base;
    }
    for (int c = 0; c < 3; c++) { g->raw_off[c] = (int)(((uintptr_t)plane[c] - base) / 2); g->raw_stride[c] = (int)stride[c]; }
    g->raw_h = h; g->raw_max = (1 << bit_depth) - 1; g->raw_offset = 1 << (bit_depth - 1);
    g->use_raw = 1;
    return 0;
}

int16_t *vc2_encgpu_input(vc2_encgpu *g, int c)
{
    cl_int err;
    if (!g->min[c]) g->min[c] = clEnqueueMapBuffer(g->q, g->in[c], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, (size_t)g->pw[c] * g->ph * 2, 0, 0, 0, &err);
    return g->min[c];
}

static double ev_ms(cl_event e)
{
    cl_ulong a = 0, b = 0;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, 8, &a, 0);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, 8, &b, 0);
    clReleaseEvent(e);
    return (double)(b - a) * 1e-6;
}

int vc2_encgpu_run(vc2_encgpu *g, int16_t *b4[3][4], int16_t *b3[3][4], double *gpu_ms)
{
    cl_int err = 0;
    cl_event ev[6];
    vc2_encgpu_release(g);
    for (int c = 0; c < 3; c++) if (g->min[c]) { clEnqueueUnmapMemObject(g->q, g->in[c], g->min[c], 0, 0, 0); g->min[c] = 0; }
    const cl_int rb = RB;
    for (int lv = 0; lv < 2; lv++)
        for (int c = 0; c < 3; c++) {
            cl_int w = lv ? g->pw[c] / 4 : g->pw[c] / 2, h = lv ? g->ph / 4 : g->ph / 2, is = 2 * w;
            cl_mem in = lv ? g->sub4[c][0] : g->in[c], *o = lv ? g->sub3[c] : g->sub4[c];
            cl_int raw = !lv && g->use_raw, hin = raw ? g->raw_h : 2 * h, maxv = g->raw_max, off = g->raw_offset, inoff = raw ? g->raw_off[c] : 0;
            if (raw) { in = g->raw; is = g->raw_stride[c]; }
            err |= clSetKernelArg(g->k, 0, sizeof in, &in);
            err |= clSetKernelArg(g->k, 1, 4, &is);
            err |= clSetKernelArg(g->k, 2, 4, &w);
            err |= clSetKernelArg(g->k, 3, 4, &h);
            err |= clSetKernelArg(g->k, 4, 4, &rb);
            for (int b = 0; b < 4; b++) err |= clSetKernelArg(g->k, 5 + b, sizeof o[b], &o[b]);
            err |= clSetKernelArg(g->k, 9, 4, &raw);
            err |= clSetKernelArg(g->k, 10, 4, &hin);
            err |= clSetKernelArg(g->k, 11, 4, &maxv);
            err |= clSetKernelArg(g->k, 12, 4, &off);
            err |= clSetKernelArg(g->k, 13, 4, &inoff);
            size_t gs[2] = { (size_t)w / 8, (size_t)(h + RB - 1) / RB };
            err |= clEnqueueNDRangeKernel(g->q, g->k, 2, 0, gs, 0, 0, 0, &ev[lv * 3 + c]);
            if (err) return -1;
        }
    g->mbands = clEnqueueMapBuffer(g->q, g->bands, CL_FALSE, CL_MAP_READ, 0, g->nbands * 2, 0, 0, 0, &err);
    if (err || clFinish(g->q)) return -1;
    g->mapped = 1;
    double t = 0;
    for (int i = 0; i < 6; i++) t += ev_ms(ev[i]);
    g->ms = t;
    if (gpu_ms) *gpu_ms = t;
    for (int c = 0; c < 3; c++) for (int b = 0; b < 4; b++) { b4[c][b] = g->mbands + g->off4[c][b]; b3[c][b] = g->mbands + g->off3[c][b]; }
    return 0;
}

int vc2_encgpu_base(const vc2_encgpu *g, int c, int level, int o) { return g->base[c][level][o]; }

const uint32_t *vc2_encgpu_count(vc2_encgpu *g, int q, const float r16[18], double *gpu_ms)
{
    for (int i = 0; i < g->nsl; i++) if (g->sl[i].q == q) return g->sl[i].map;
    if (g->nsl == NSLOT) return 0;
    slot *s = &g->sl[g->nsl];
    cl_int err = 0;
    if (!s->mem) {
        s->mem = clCreateBuffer(g->ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, g->nout * 8, 0, &err);
        if (err) { s->mem = 0; return 0; }
    }
    cl_event ev;
    const cl_int nx = g->nx;
    err |= clEnqueueWriteBuffer(g->q, g->r16, CL_FALSE, 0, 18 * sizeof(float), r16, 0, 0, 0);
    err |= clSetKernelArg(g->kc, 0, sizeof g->bands, &g->bands);
    err |= clSetKernelArg(g->kc, 1, sizeof g->desc, &g->desc);
    err |= clSetKernelArg(g->kc, 2, sizeof g->r16, &g->r16);
    err |= clSetKernelArg(g->kc, 3, 4, &nx);
    err |= clSetKernelArg(g->kc, 4, sizeof s->mem, &s->mem);
    size_t gs[2] = { (size_t)(g->nseg_max + 63) / 64 * 64, 18 };
    err |= clEnqueueNDRangeKernel(g->q, g->kc, 2, 0, gs, 0, 0, 0, &ev);
    if (err) return 0;
    s->map = clEnqueueMapBuffer(g->q, s->mem, CL_TRUE, CL_MAP_READ, 0, g->nout * 8, 0, 0, 0, &err);
    if (err) { s->map = 0; return 0; }
    double t = ev_ms(ev);
    g->ms += t;
    if (gpu_ms) *gpu_ms += t;
    s->q = q;
    g->nsl++;
    return s->map;
}
