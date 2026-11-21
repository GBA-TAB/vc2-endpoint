// vc2xbench: a VC-2 stream through the decoder card (or its model) via vc2_xdma: checks the pictures against the CPU
// decoder, and measures throughput and latency.
//   vc2xbench [options] in.vc2
//   --dev D         /dev/xdma0 (the card) or model:[threads=N,cpus=a/b,queue=N] (default model:)
//   --frames N      pictures to send (default: the stream once); the stream repeats
//   --fps F         pace the sending at F pictures/s (default: flat out)
//   --piece N       receive in pieces of N line pairs (default 16 = one slice row at 1080p), to see the first lines
//   --send-cpu C, --recv-cpu C   pin the two threads (the receiver best on an A76, CPUs 4-7 on the RK3588)
//   --no-check      don't compare with the CPU decoder (else every distinct picture of the first pass is compared)
//   --out f.nv20    write the received pictures of the first pass (interleaved NV20, as received)
// With ERR_GEOM set the card's pictures have undefined content, so mismatches are expected there.
// Prints per picture only on a mismatch; at the end: pictures, mismatches, MB/s, pictures/s, latency from the start
// of the send to the first and to the last line, CPU per picture, and the card's status flags.
#define _GNU_SOURCE
#include "../lib/vc2_fpga.h"
#include "../lib/vc2_xdma.h"
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void sleep_until(double t)
{
    struct timespec ts = { (time_t)t, (long)((t - (time_t)t) * 1e9) };
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, 0) == EINTR) {}
}
static void pin(int c)
{
    if (c < 0) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(c, &s);
    if (pthread_setaffinity_np(pthread_self(), sizeof s, &s)) fprintf(stderr, "pin to cpu %d failed\n", c);
}

typedef struct { uint8_t *h2c; size_t len; int w, h; } unit;
typedef struct { vc2x *x; unit *u; int nu, frames, cpu, err; double fps, t0, *sent; } sender;

static void *send_main(void *arg)
{
    sender *s = arg;
    pin(s->cpu);
    for (int i = 0; i < s->frames; i++) {
        if (s->fps > 0) sleep_until(s->t0 + i / s->fps);
        s->sent[i] = now();
        if (vc2x_send(s->x, s->u[i % s->nu].h2c, s->u[i % s->nu].len)) { s->err = errno; break; }
    }
    return 0;
}

// interleaved NV20 -> planar 16-bit, as vc2dec writes it
static void unpack(const uint8_t *nv, int w, int h, uint16_t *y, uint16_t *cb, uint16_t *cr)
{
    size_t line = (size_t)w * 10 / 8;
    for (int r = 0; r < h; r++) {
        const uint8_t *py = nv + 2 * line * r, *pc = py + line;
        for (int x = 0; x < w; x += 4) {
            uint64_t v = 0, c = 0;
            memcpy(&v, py + x / 4 * 5, 5); memcpy(&c, pc + x / 4 * 5, 5);
            for (int k = 0; k < 4; k++) y[(size_t)r * w + x + k] = (v >> (10 * k)) & 0x3ff;
            cb[(size_t)r * w / 2 + x / 2] = c & 0x3ff; cr[(size_t)r * w / 2 + x / 2] = (c >> 10) & 0x3ff;
            cb[(size_t)r * w / 2 + x / 2 + 1] = (c >> 20) & 0x3ff; cr[(size_t)r * w / 2 + x / 2 + 1] = (c >> 30) & 0x3ff;
        }
    }
}

static int cmp_frame(const vc2_frame *f, const uint16_t *pl[3])
{
    for (int c = 0; c < 3; c++) {
        int w = c ? f->width / 2 : f->width;
        for (int y = 0; y < f->height; y++)
            if (memcmp(f->plane[c] + (size_t)y * f->stride[c], pl[c] + (size_t)y * w, (size_t)w * 2)) return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    const char *dev = "model:", *in = 0, *out = 0;
    int frames = 0, piece = 16, scpu = -1, rcpu = -1, check = 1;
    double fps = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--dev")) { dev = v; i++; }
        else if (!strcmp(a, "--frames")) { frames = atoi(v); i++; }
        else if (!strcmp(a, "--fps")) { fps = atof(v); i++; }
        else if (!strcmp(a, "--piece")) { piece = atoi(v); i++; }
        else if (!strcmp(a, "--send-cpu")) { scpu = atoi(v); i++; }
        else if (!strcmp(a, "--recv-cpu")) { rcpu = atoi(v); i++; }
        else if (!strcmp(a, "--no-check")) check = 0;
        else if (!strcmp(a, "--out")) { out = v; i++; }
        else in = a;
    }
    if (!in || piece < 1) { fprintf(stderr, "usage: vc2xbench [--dev D] [--frames N] [--fps F] [--piece N] [--send-cpu C] [--recv-cpu C] [--no-check] [--out f] in.vc2\n"); return 2; }
    FILE *fi = fopen(in, "rb");
    if (!fi) { perror(in); return 1; }
    fseek(fi, 0, SEEK_END);
    size_t len = ftell(fi);
    rewind(fi);
    uint8_t *buf = calloc(1, len + VC2_INPUT_PADDING);
    if (fread(buf, 1, len, fi) != len) return 1;
    fclose(fi);

    // the H2C bytes of every access unit the card takes, and the CPU decoder's pictures for the check
    unit *u = 0;
    vc2_frame *ref = 0;
    int nu = 0;
    vc2_fpga_tx tx = { 0 };
    vc2_dec *d = vc2_dec_new(1, 0);
    for (size_t pos = 0, n; (n = vc2_access_unit(buf + pos, len - pos)); pos += n) {
        const char *why;
        size_t cap = n + 256;
        uint8_t *h = 0;
        if (posix_memalign((void **)&h, 4096, cap)) return 1;
        size_t m = vc2_fpga_pack(&tx, buf + pos, n, h, cap, &why);
        if (!m) { fprintf(stderr, "access unit %d is not for the card: %s\n", nu, why); free(h); continue; }
        u = realloc(u, (nu + 1) * sizeof *u);
        vc2_fpga_desc ds;
        vc2_fpga_desc_read(h, m, &ds);
        u[nu] = (unit){ h, m, ds.width, ds.height };
        if (check) {
            vc2_frame f;
            if (vc2_dec_decode(d, buf + pos, n, &f)) { fprintf(stderr, "CPU decode failed at %d\n", nu); return 1; }
            ref = realloc(ref, (nu + 1) * sizeof *ref);
            ref[nu] = f;
            for (int c = 0; c < 3; c++) { // keep a copy: the decoder reuses its buffers
                int w = c ? f.width / 2 : f.width;
                uint16_t *p = malloc((size_t)w * f.height * 2);
                for (int y = 0; y < f.height; y++) memcpy(p + (size_t)y * w, f.plane[c] + (size_t)y * f.stride[c], (size_t)w * 2);
                ref[nu].plane[c] = p; ref[nu].stride[c] = w;
            }
        }
        nu++;
    }
    if (!nu) { fprintf(stderr, "nothing for the card in %s\n", in); return 1; }
    if (!frames) frames = nu;
    for (int i = 1; i < nu; i++)
        if (u[i].w != u[0].w || u[i].h != u[0].h) { fprintf(stderr, "the picture size changes: one size per run\n"); return 2; }
    const int W = u[0].w, H = u[0].h;
    const size_t pair = 2 * vc2x_line_bytes(W), pic = vc2x_picture_bytes(W, H);
    if (piece > H) piece = H;

    vc2x *x = vc2x_open(dev);
    if (!x) { fprintf(stderr, "vc2x_open %s: %s\n", dev, strerror(errno)); return 1; }
    if (vc2x_check(x)) { fprintf(stderr, "%s: no decoder (%s)\n", dev, strerror(errno)); return 1; }
    if (vc2x_reset(x)) { fprintf(stderr, "%s: reset failed (%s)\n", dev, strerror(errno)); return 1; }

    uint8_t *rx = 0;
    if (posix_memalign((void **)&rx, 4096, pic)) return 1;
    uint16_t *pl[3] = { malloc((size_t)W * H * 2), malloc((size_t)W * H), malloc((size_t)W * H) };
    FILE *fo = out ? fopen(out, "wb") : 0;
    double *sent = calloc(frames, sizeof *sent), *first = calloc(frames, sizeof *first), *last = calloc(frames, sizeof *last);
    sender s = { x, u, nu, frames, scpu, 0, fps, 0, sent };
    pin(rcpu);
    double t0 = now(), c0 = cpu();
    s.t0 = t0;
    pthread_t th;
    pthread_create(&th, 0, send_main, &s);
    int bad = 0, got = 0;
    for (int i = 0; i < frames; i++) {
        for (int r = 0; r < H; r += piece) {
            int n = H - r < piece ? H - r : piece;
            if (vc2x_recv(x, rx + pair * r, pair * n)) { fprintf(stderr, "recv picture %d: %s\n", i, strerror(errno)); goto done; }
            if (!r) first[i] = now();
        }
        last[i] = now();
        got++;
        if (i < nu) {
            if (fo) fwrite(rx, 1, pic, fo);
            if (check) {
                unpack(rx, W, H, pl[0], pl[1], pl[2]);
                if (!cmp_frame(&ref[i], (const uint16_t **)pl)) { printf("picture %d: MISMATCH\n", i); bad++; }
            }
        }
    }
done:
    pthread_join(th, 0);
    double t1 = now(), c1 = cpu(), lf = 0, ll = 0, lfmax = 0, llmax = 0;
    for (int i = 0; i < got; i++) {
        double a = first[i] - sent[i], b = last[i] - sent[i];
        lf += a; ll += b;
        if (a > lfmax) lfmax = a;
        if (b > llmax) llmax = b;
    }
    struct vc2x_status st;
    vc2x_status(x, &st);
    printf("%s: %d of %d pictures %dx%d, %d mismatches%s, %.1f MB/s out, %.2f pictures/s\n", dev, got, frames, W, H, bad,
           check ? "" : " (not checked)", got * (double)pic / (t1 - t0) / 1e6, got / (t1 - t0));
    if (got)
        printf("  latency from send start: first lines %.2f ms (max %.2f), last line %.2f ms (max %.2f); CPU %.2f ms/picture\n",
               lf / got * 1e3, lfmax * 1e3, ll / got * 1e3, llmax * 1e3, (c1 - c0) / got * 1e3);
    printf("  status: id %08x version %08x flags %#x%s%s%s%s%s pictures_done %u\n", st.id, st.version, st.flags,
           st.flags & VC2X_ERR_MAGIC ? " ERR_MAGIC" : "", st.flags & VC2X_ERR_FORMAT ? " ERR_FORMAT" : "",
           st.flags & VC2X_ERR_OVERRUN ? " ERR_OVERRUN" : "", st.flags & VC2X_ERR_GEOM ? " ERR_GEOM" : "",
           st.flags & VC2X_SATURATED ? " SATURATED" : "", st.pictures_done);
    if (s.err) printf("  send failed: %s\n", strerror(s.err));
    if (fo) fclose(fo);
    vc2x_close(x);
    return bad || got < frames || s.err || (st.flags & VC2X_ERRORS) ? 1 : 0;
}
