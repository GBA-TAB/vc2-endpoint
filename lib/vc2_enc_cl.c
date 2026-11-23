// The encoder's GPU front end in OpenCL (see vc2_enc_gpu.h): the kernel of vc2_fwd.cl on buffers the driver
// allocates (CL_MEM_ALLOC_HOST_PTR), mapped for the CPU around each frame: the input for writing before the
// launch, the bands for reading after it (the counter and the writer read them in place).
#define CL_TARGET_OPENCL_VERSION 300
#include "vc2_enc_gpu.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__asm__(".pushsection .rodata\n.global vc2_fwd_cl_src\nvc2_fwd_cl_src:\n.incbin \"" VC2_FWD_CL "\"\n.byte 0\n.popsection\n");
extern const char vc2_fwd_cl_src[];

#define RB 16 // band rows per work-item (best on the Mali-G610: 8 1.56 ms, 16 1.39 ms, 32 2.36 ms per 1080p frame)

struct vc2_encgpu {
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k;
    cl_mem in[3], b4[3][4], b3[3][4];
    int16_t *min[3], *m4[3][4], *m3[3][4];
    int pw[3], ph, mapped;
};

void vc2_encgpu_free(vc2_encgpu *g)
{
    if (!g) return;
    vc2_encgpu_release(g);
    for (int c = 0; c < 3; c++) {
        if (g->min[c]) clEnqueueUnmapMemObject(g->q, g->in[c], g->min[c], 0, 0, 0);
        if (g->in[c]) clReleaseMemObject(g->in[c]);
        for (int b = 0; b < 4; b++) { if (g->b4[c][b]) clReleaseMemObject(g->b4[c][b]); if (g->b3[c][b]) clReleaseMemObject(g->b3[c][b]); }
    }
    if (g->q) clFinish(g->q);
    if (g->k) clReleaseKernel(g->k);
    if (g->prog) clReleaseProgram(g->prog);
    if (g->q) clReleaseCommandQueue(g->q);
    if (g->ctx) clReleaseContext(g->ctx);
    free(g);
}

vc2_encgpu *vc2_encgpu_new(const int pw[3], int ph)
{
    for (int c = 0; c < 3; c++) if (pw[c] % 64 || pw[c] < 64) return 0;
    if (ph % 16) return 0;
    vc2_encgpu *g = calloc(1, sizeof *g);
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
    g->k = clCreateKernel(g->prog, "fwd53", &err);
    if (err) goto fail;
    g->ph = ph;
    for (int c = 0; c < 3; c++) {
        size_t w4 = pw[c] / 2, h4 = ph / 2, n4 = w4 * h4, n3 = n4 / 4;
        g->pw[c] = pw[c];
        g->in[c] = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)pw[c] * ph * 2, 0, &err);
        if (err) goto fail;
        for (int b = 0; b < 4; b++) {
            g->b4[c][b] = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n4 * 2 + 64, 0, &err);
            if (err) goto fail;
            g->b3[c][b] = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n3 * 2 + 64, 0, &err);
            if (err) goto fail;
        }
    }
    return g;
fail:
    vc2_encgpu_free(g);
    return 0;
}

int16_t *vc2_encgpu_input(vc2_encgpu *g, int c)
{
    cl_int err;
    if (!g->min[c]) g->min[c] = clEnqueueMapBuffer(g->q, g->in[c], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, (size_t)g->pw[c] * g->ph * 2, 0, 0, 0, &err);
    return g->min[c];
}

void vc2_encgpu_release(vc2_encgpu *g)
{
    if (!g || !g->mapped) return;
    for (int c = 0; c < 3; c++)
        for (int b = 0; b < 4; b++) {
            clEnqueueUnmapMemObject(g->q, g->b4[c][b], g->m4[c][b], 0, 0, 0);
            clEnqueueUnmapMemObject(g->q, g->b3[c][b], g->m3[c][b], 0, 0, 0);
        }
    g->mapped = 0;
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
            cl_mem in = lv ? g->b4[c][0] : g->in[c], *o = lv ? g->b3[c] : g->b4[c];
            err |= clSetKernelArg(g->k, 0, sizeof in, &in);
            err |= clSetKernelArg(g->k, 1, 4, &is);
            err |= clSetKernelArg(g->k, 2, 4, &w);
            err |= clSetKernelArg(g->k, 3, 4, &h);
            err |= clSetKernelArg(g->k, 4, 4, &rb);
            for (int b = 0; b < 4; b++) err |= clSetKernelArg(g->k, 5 + b, sizeof o[b], &o[b]);
            size_t gs[2] = { (size_t)w / 8, (size_t)(h + RB - 1) / RB };
            err |= clEnqueueNDRangeKernel(g->q, g->k, 2, 0, gs, 0, 0, 0, &ev[lv * 3 + c]);
            if (err) return -1;
        }
    for (int c = 0; c < 3 && !err; c++)
        for (int b = 0; b < 4 && !err; b++) {
            size_t n4 = (size_t)(g->pw[c] / 2) * (g->ph / 2);
            g->m4[c][b] = clEnqueueMapBuffer(g->q, g->b4[c][b], CL_FALSE, CL_MAP_READ, 0, n4 * 2, 0, 0, 0, &err);
            if (!err) g->m3[c][b] = clEnqueueMapBuffer(g->q, g->b3[c][b], CL_FALSE, CL_MAP_READ, 0, n4 / 2, 0, 0, 0, &err);
        }
    if (err || clFinish(g->q)) return -1;
    g->mapped = 1;
    double t = 0;
    for (int i = 0; i < 6; i++) {
        cl_ulong a = 0, b = 0;
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, 8, &a, 0);
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, 8, &b, 0);
        t += (double)(b - a) * 1e-6;
        clReleaseEvent(ev[i]);
    }
    if (gpu_ms) *gpu_ms = t;
    for (int c = 0; c < 3; c++) for (int b = 0; b < 4; b++) { b4[c][b] = g->m4[c][b]; b3[c][b] = g->m3[c][b]; }
    return 0;
}
