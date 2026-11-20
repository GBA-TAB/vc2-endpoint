// vc2bench: decodes a VC-2 stream held in memory, in a loop, and reports speed, CPU and latency.
//   --frames N     frames to decode (default 300)        --threads N   decoder threads (default 1)
//   --cpus a,b,..  pin the threads (first = caller)      --fps F       pace at F frames/s (default: flat out)
//   --subframe K   feed each frame in K equal parts; with --fps the parts arrive spread over the frame period
//                  (constant bit rate), and the latency is counted from the arrival of the last byte
//   --out f.yuv    write the decoded frames of the first pass
//   --weights a,b  relative speed of each thread's core (sizes the bands of slice rows)
//   --nv20         decode into the display's packed NV20 layout (checked against the planar output first)
//   --gpu cl|vk    the last wavelet level on the GPU (OpenCL / Vulkan, if built with GPU=...); with --nv20 the
//                  target is a dma-buf from the dma-heap, as a scanout buffer would be (--nv20-copy: plain memory)
//   --gpu-cpus a,b cores for the GPU driver's own threads
#define _GNU_SOURCE
#include "../lib/vc2.h"
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#ifdef __linux__
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void sleep_until(double t)
{
    struct timespec ts = { (time_t)t, (long)((t - (time_t)t) * 1e9) };
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, 0);
}

int main(int argc, char **argv)
{
    int frames = 300, threads = 1, sub = 0, cpus[64], ncpu = 0, nv20 = 0, gpu = VC2_GPU_OFF, gcpus[64], ngcpu = 0, nv20_copy = 0;
    double fps = 0;
    float weights[64];
    int nw = 0;
    for (int i = 0; i < 64; i++) weights[i] = 1;
    const char *in = 0, *out = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--frames")) { frames = atoi(v); i++; }
        else if (!strcmp(a, "--threads")) { threads = atoi(v); i++; }
        else if (!strcmp(a, "--fps")) { fps = atof(v); i++; }
        else if (!strcmp(a, "--subframe")) { sub = atoi(v); i++; }
        else if (!strcmp(a, "--out")) { out = v; i++; }
        else if (!strcmp(a, "--nv20")) nv20 = 1;
        else if (!strcmp(a, "--nv20-copy")) nv20 = nv20_copy = 1;
        else if (!strcmp(a, "--gpu")) { gpu = !strcmp(v, "vk") ? VC2_GPU_VULKAN : VC2_GPU_OPENCL; i++; }
        else if (!strcmp(a, "--gpu-cpus")) {
            char *s = strdup(v);
            for (char *t = strtok(s, ","); t && ngcpu < 64; t = strtok(0, ",")) gcpus[ngcpu++] = atoi(t);
            i++;
        }
        else if (!strcmp(a, "--weights")) {
            char *s = strdup(v);
            for (char *t = strtok(s, ","); t && nw < 64; t = strtok(0, ",")) weights[nw++] = (float)atof(t);
            i++;
        }
        else if (!strcmp(a, "--cpus")) {
            char *s = strdup(v);
            for (char *t = strtok(s, ","); t && ncpu < 64; t = strtok(0, ",")) cpus[ncpu++] = atoi(t);
            i++;
        } else in = a;
    }
    if (!in) { fprintf(stderr, "usage: vc2bench [--frames N] [--threads N] [--cpus list] [--fps F] [--subframe K] [--out f.yuv] in.vc2\n"); return 2; }
    FILE *fi = fopen(in, "rb");
    if (!fi) { perror(in); return 1; }
    fseek(fi, 0, SEEK_END);
    size_t len = ftell(fi);
    rewind(fi);
    uint8_t *buf = calloc(1, len + VC2_INPUT_PADDING);
    if (fread(buf, 1, len, fi) != len) return 1;
    size_t au_off[4096], au_len[4096], pos = 0, n;
    int naus = 0;
    while (naus < 4096 && (n = vc2_access_unit(buf + pos, len - pos))) { au_off[naus] = pos; au_len[naus++] = n; pos += n; }
    if (!naus) { fprintf(stderr, "no pictures in %s\n", in); return 1; }
    if (ncpu) {
        if (ncpu < threads) { fprintf(stderr, "--cpus needs %d entries\n", threads); return 2; }
        cpu_set_t set;
        CPU_ZERO(&set); CPU_SET(cpus[0], &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    vc2_dec *d = vc2_dec_new(threads, ncpu ? cpus : 0);
    if (nw) vc2_dec_set_weights(d, weights);
    if (gpu && vc2_dec_set_gpu(d, gpu, ngcpu ? gcpus : 0, ngcpu)) { fprintf(stderr, "this build has no such GPU backend (make GPU=cl|vk)\n"); return 2; }
    FILE *fo = out ? fopen(out, "wb") : 0;
    vc2_frame f;
    for (int i = 0; i < naus && i < 8; i++) // warm up: buffers, page faults
        if (vc2_dec_decode(d, buf + au_off[i], au_len[i], &f)) { fprintf(stderr, "decode error\n"); return 1; }
    if (nv20) { // decode into the display layout instead; check the first frame against the planar output
        size_t pitch = (size_t)f.width * 10 / 8, sz = pitch * f.height;
        uint8_t *y = 0, *ref = calloc(1, 2 * sz + 8);
        int fd = -1;
#ifdef __linux__
        if (gpu && !nv20_copy) { // a dma-buf, as the display's buffers are
            int h = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
            struct dma_heap_allocation_data a = { .len = (2 * sz + 8191) & ~(size_t)4095, .fd_flags = O_RDWR | O_CLOEXEC };
            if (h >= 0 && !ioctl(h, DMA_HEAP_IOCTL_ALLOC, &a)) {
                y = mmap(0, a.len, PROT_READ | PROT_WRITE, MAP_SHARED, a.fd, 0);
                if (y == MAP_FAILED) y = 0; else fd = a.fd;
            }
            if (h >= 0) close(h);
        }
#endif
        if (!y) y = malloc(2 * sz + 8);
        if (vc2_dec_decode(d, buf, au_len[0], &f)) return 1;
        for (int r = 0; r < f.height; r++)
            for (int x = 0; x < f.width; x += 4) {
                const uint16_t *yp = f.plane[0] + (size_t)r * f.stride[0] + x, *u = f.plane[1] + (size_t)r * f.stride[1] + x / 2,
                               *v = f.plane[2] + (size_t)r * f.stride[2] + x / 2;
                uint64_t a = (uint64_t)yp[0] | (uint64_t)yp[1] << 10 | (uint64_t)yp[2] << 20 | (uint64_t)yp[3] << 30;
                uint64_t c = (uint64_t)u[0] | (uint64_t)v[0] << 10 | (uint64_t)u[1] << 20 | (uint64_t)v[1] << 30;
                memcpy(ref + r * pitch + x / 4 * 5, &a, 5);
                memcpy(ref + sz + r * pitch + x / 4 * 5, &c, 5);
            }
        vc2_dec_set_nv20(d, y, y + sz, pitch);
        if (fd >= 0) vc2_dec_set_nv20_dmabuf(d, fd, 0, sz, pitch);
        if (vc2_dec_decode(d, buf, au_len[0], &f)) return 1;
        printf("NV20 output %s\n", memcmp(y, ref, 2 * sz) ? "DIFFERS from the planar output" : "matches the planar output");
        free(ref);
    }
    double t_start = now(), c_start = cpu(), lat_sum = 0, lat_max = 0, tp[2] = { 0, 0 }, first_sum = 0, dec_sum = 0, gpu_sum = 0;
    int late = 0;
    for (int i = 0; i < frames; i++) {
        const uint8_t *au = buf + au_off[i % naus];
        size_t al = au_len[i % naus];
        double t_arrive = fps > 0 ? t_start + (i + 1) / fps : 0; // the last byte of frame i arrives here
        double t0, t_first = 0, t_done;
        if (sub > 1) {
            int lines = 0, begun = 0;
            t0 = now();
            for (int k = 1; k <= sub; k++) {
                size_t have = k == sub ? al : al * k / sub;
                if (fps > 0) sleep_until(t_arrive - (sub - k) / fps / sub);
                if (k == 1) t0 = now();
                if (!begun) { if (vc2_dec_begin(d, au, have, &f)) continue; begun = 1; }
                lines = vc2_dec_feed(d, have);
                if (lines > 0 && !t_first) t_first = now();
            }
            if (lines != f.height) { fprintf(stderr, "decode error\n"); return 1; }
        } else {
            if (fps > 0) sleep_until(t_arrive);
            t0 = now();
            if (vc2_dec_decode(d, au, al, &f)) { fprintf(stderr, "decode error\n"); return 1; }
        }
        t_done = now();
        double lat = fps > 0 ? t_done - t_arrive : t_done - t0, t[2];
        if (fps > 0 && lat > 1 / fps) late++;
        lat_sum += lat; if (lat > lat_max) lat_max = lat;
        dec_sum += t_done - t0;
        if (t_first) first_sum += t_first - t0;
        vc2_dec_times(d, t);
        tp[0] += t[0]; tp[1] += t[1];
        gpu_sum += vc2_dec_gpu_time(d);
        if (fo && i < naus && !nv20)
            for (int c = 0; c < 3; c++) {
                int w = c && f.chroma ? f.width / 2 : f.width, h = c && f.chroma == 2 ? f.height / 2 : f.height;
                for (int y = 0; y < h; y++) fwrite(f.plane[c] + (size_t)y * f.stride[c], 2, w, fo);
            }
    }
    double wall = now() - t_start, c = cpu() - c_start;
    const vc2_seq *s = vc2_dec_seq(d);
    const vc2_pic *p = vc2_dec_pic(d);
    printf("%dx%d wavelet %d depth %d slices %dx%d, %.2f bpp, %d threads%s\n", s->width, s->height, p->wavelet, p->depth, p->slices_x,
           p->slices_y, au_len[0] * 8.0 / s->width / s->height, threads, sub > 1 ? ", sub-frame" : "");
    printf("frames %d  fps %.1f  cpu %.2f ms/frame  decode %.2f ms (entropy %.2f + wavelet %.2f)", frames, frames / wall, c / frames * 1e3,
           dec_sum / frames * 1e3, tp[0] / frames * 1e3, tp[1] / frames * 1e3);
    if (vc2_dec_gpu_active(d)) printf("  gpu %.2f ms", gpu_sum / frames * 1e3);
    if (fps > 0) printf("  latency after the last byte: avg %.2f max %.2f ms, late %d", lat_sum / frames * 1e3, lat_max * 1e3, late);
    else printf("  max %.2f ms", lat_max * 1e3);
    if (sub > 1) printf("  first lines after %.2f ms", first_sum / frames * 1e3);
    printf("\n");
    if (fo) fclose(fo);
    vc2_dec_free(d);
    free(buf);
    return 0;
}
