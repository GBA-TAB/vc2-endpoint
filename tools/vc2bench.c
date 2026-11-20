// vc2bench: decodes a VC-2 stream held in memory, in a loop, and reports speed, CPU and latency.
//   --frames N     frames to decode (default 300)        --threads N   decoder threads (default 1)
//   --cpus a,b,..  pin the threads (first = caller)      --fps F       pace at F frames/s (default: flat out)
//   --subframe K   feed each frame in K equal parts; with --fps the parts arrive spread over the frame period
//                  (constant bit rate), and the latency is counted from the arrival of the last byte
//   --out f.yuv    write the decoded frames of the first pass
#define _GNU_SOURCE
#include "../lib/vc2.h"
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void sleep_until(double t)
{
    struct timespec ts = { (time_t)t, (long)((t - (time_t)t) * 1e9) };
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, 0);
}

int main(int argc, char **argv)
{
    int frames = 300, threads = 1, sub = 0, cpus[64], ncpu = 0;
    double fps = 0;
    const char *in = 0, *out = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--frames")) { frames = atoi(v); i++; }
        else if (!strcmp(a, "--threads")) { threads = atoi(v); i++; }
        else if (!strcmp(a, "--fps")) { fps = atof(v); i++; }
        else if (!strcmp(a, "--subframe")) { sub = atoi(v); i++; }
        else if (!strcmp(a, "--out")) { out = v; i++; }
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
    FILE *fo = out ? fopen(out, "wb") : 0;
    vc2_frame f;
    for (int i = 0; i < naus && i < 8; i++) // warm up: buffers, page faults
        if (vc2_dec_decode(d, buf + au_off[i], au_len[i], &f)) { fprintf(stderr, "decode error\n"); return 1; }
    double t_start = now(), c_start = cpu(), lat_sum = 0, lat_max = 0, tp[2] = { 0, 0 }, first_sum = 0, dec_sum = 0;
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
        if (fo && i < naus)
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
    if (fps > 0) printf("  latency after the last byte: avg %.2f max %.2f ms, late %d", lat_sum / frames * 1e3, lat_max * 1e3, late);
    else printf("  max %.2f ms", lat_max * 1e3);
    if (sub > 1) printf("  first lines after %.2f ms", first_sum / frames * 1e3);
    printf("\n");
    if (fo) fclose(fo);
    vc2_dec_free(d);
    free(buf);
    return 0;
}
