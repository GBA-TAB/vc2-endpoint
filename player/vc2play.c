// vc2play: decodes a VC-2 HQ stream (4:2:2 10-bit) and shows it on a DRM plane as NV20. The decoder writes
// the packed NV20 lines straight into the scanout buffers (no repack pass); the display controller scales
// to the screen. Needs DRM master (desktop stopped). The DRM part follows our jxsplay (jxs-endpoint/player).
//
//   vc2play [--frames N] [--threads N] [--cpus a,b,..] [--conn id --crtc id] [--plane id] [--gpu cl|vk] [--gpu-cpus a,b,..] file.vc2
// --gpu: the last wavelet level on the GPU, which writes the NV20 lines into the scanout buffers itself (they are
// handed to it as dma-bufs); --gpu-cpus: cores for the GPU driver's threads.
// Defaults 224 / 93 = the Firefly's HDMI-A-2; plane 0 = the CRTC's own primary plane, found after the modeset.
// Mode: the picture's size at 50 Hz if the screen has it (then paced by vblank), else its preferred mode
// (paced by a timer). VC2_MODE=WxH@Hz picks another.
#define _GNU_SOURCE
#include "../lib/vc2.h"
#include <drm_fourcc.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#ifndef DRM_FORMAT_NV20
#define DRM_FORMAT_NV20 fourcc_code('N', 'V', '2', '0')
#endif
#define FPS 50
#define NBUF 3

struct buf { uint32_t handle, pitch, fb; uint8_t *map; uint64_t size; int dmabuf; };
static volatile int stop;
static void on_sig(int s) { (void)s; stop = 1; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int dumb(int fd, uint32_t w, uint32_t h, uint32_t bpp, struct buf *b)
{
    struct drm_mode_create_dumb c = { .width = w, .height = h, .bpp = bpp };
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &c)) return -1;
    struct drm_mode_map_dumb m = { .handle = c.handle };
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &m)) return -1;
    b->map = mmap(0, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
    if (b->map == MAP_FAILED) return -1;
    b->handle = c.handle; b->pitch = c.pitch; b->size = c.size;
    memset(b->map, 0, c.size);
    return 0;
}

int main(int argc, char **argv)
{
    int frames = 3000, threads = 3, cpus[64] = { 4, 5, 6 }, ncpu = 3, gpu = VC2_GPU_OFF, gcpus[64], ngcpu = 0;
    uint32_t conn_id = 224, crtc_id = 93, plane_id = 0;
    const char *in = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--frames")) { frames = atoi(v); i++; }
        else if (!strcmp(a, "--threads")) { threads = atoi(v); i++; }
        else if (!strcmp(a, "--conn")) { conn_id = atoi(v); i++; }
        else if (!strcmp(a, "--crtc")) { crtc_id = atoi(v); i++; }
        else if (!strcmp(a, "--plane")) { plane_id = atoi(v); i++; }
        else if (!strcmp(a, "--gpu")) { gpu = !strcmp(v, "vk") ? VC2_GPU_VULKAN : VC2_GPU_OPENCL; i++; }
        else if (!strcmp(a, "--gpu-cpus")) {
            char *s = strdup(v);
            for (char *t = strtok(s, ","); t && ngcpu < 64; t = strtok(0, ",")) gcpus[ngcpu++] = atoi(t);
            i++;
        }
        else if (!strcmp(a, "--cpus")) {
            char *s = strdup(v);
            ncpu = 0;
            for (char *t = strtok(s, ","); t && ncpu < 64; t = strtok(0, ",")) cpus[ncpu++] = atoi(t);
            i++;
        } else in = a;
    }
    if (!in) { fprintf(stderr, "usage: vc2play [--frames N] [--threads N] [--cpus list] [--conn id --crtc id] [--plane id] file.vc2\n"); return 2; }
    if (threads > ncpu) threads = ncpu;
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);

    FILE *fi = fopen(in, "rb");
    if (!fi) { perror(in); return 1; }
    fseek(fi, 0, SEEK_END);
    size_t len = ftell(fi), pos = 0, n;
    rewind(fi);
    uint8_t *clip = calloc(1, len + VC2_INPUT_PADDING);
    if (fread(clip, 1, len, fi) != len) return 1;
    size_t au_off[4096], au_len[4096];
    int naus = 0;
    while (naus < 4096 && (n = vc2_access_unit(clip + pos, len - pos))) { au_off[naus] = pos; au_len[naus++] = n; pos += n; }
    if (!naus) { fprintf(stderr, "no pictures\n"); return 1; }

    cpu_set_t set;
    CPU_ZERO(&set); CPU_SET(cpus[0], &set);
    sched_setaffinity(0, sizeof set, &set);
    vc2_dec *dec = vc2_dec_new(threads, cpus);
    if (gpu && vc2_dec_set_gpu(dec, gpu, ngcpu ? gcpus : 0, ngcpu)) { fprintf(stderr, "this build has no such GPU backend (make GPU=cl|vk)\n"); return 2; }
    vc2_frame f;
    if (vc2_dec_begin(dec, clip, au_len[0], &f) || f.chroma != 1 || f.bit_depth != 10) { fprintf(stderr, "need a 4:2:2 10-bit stream\n"); return 1; }
    const int W = f.width, H = f.height;

    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0 || drmSetMaster(fd)) { perror("drm master (desktop still running?)"); return 1; }
    drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    drmModeConnector *conn = drmModeGetConnector(fd, conn_id);
    if (!conn || conn->connection != DRM_MODE_CONNECTED || !conn->count_modes) { fprintf(stderr, "connector %u\n", conn_id); return 1; }
    int want_w = W, want_h = H, want_hz = FPS;
    if (getenv("VC2_MODE")) sscanf(getenv("VC2_MODE"), "%dx%d@%d", &want_w, &want_h, &want_hz);
    drmModeModeInfo mode = conn->modes[0];
    for (int i = 0; i < conn->count_modes; i++)
        if (conn->modes[i].hdisplay == want_w && conn->modes[i].vdisplay == want_h && (int)conn->modes[i].vrefresh == want_hz) {
            mode = conn->modes[i];
            break;
        }
    int vsync = mode.vrefresh == FPS, crtc_index = -1; // vblank pacing when the screen runs at the clip's rate
    drmModeRes *res = drmModeGetResources(fd);
    for (int i = 0; res && i < res->count_crtcs; i++)
        if (res->crtcs[i] == crtc_id) crtc_index = i;
    if (crtc_index < 0) vsync = 0;

    struct buf bg; // black primary at the screen's mode, then the video on that plane
    if (dumb(fd, mode.hdisplay, mode.vdisplay, 32, &bg) ||
        drmModeAddFB(fd, mode.hdisplay, mode.vdisplay, 24, 32, bg.pitch, bg.handle, &bg.fb) ||
        drmModeSetCrtc(fd, crtc_id, bg.fb, 0, 0, &conn_id, 1, &mode)) { perror("set crtc"); return 1; }
    if (!plane_id) {
        drmModePlaneRes *pr = drmModeGetPlaneResources(fd);
        for (uint32_t i = 0; pr && i < pr->count_planes && !plane_id; i++) {
            drmModePlane *pl = drmModeGetPlane(fd, pr->planes[i]);
            if (pl && pl->crtc_id == crtc_id && pl->fb_id == bg.fb) plane_id = pl->plane_id;
            drmModeFreePlane(pl);
        }
        if (!plane_id) { fprintf(stderr, "no primary plane on crtc %u\n", crtc_id); return 1; }
    }
    struct buf b[NBUF];
    for (int i = 0; i < NBUF; i++) {
        if (dumb(fd, W * 10 / 8, H * 2 + 1, 8, &b[i])) { perror("dumb"); return 1; } // 10 bits per sample, Y and CbCr lines alike
        uint32_t h[4] = { b[i].handle, b[i].handle }, p[4] = { b[i].pitch, b[i].pitch }, o[4] = { 0, b[i].pitch * H };
        if (drmModeAddFB2(fd, W, H, DRM_FORMAT_NV20, h, p, o, &b[i].fb, 0)) { perror("addfb2 NV20"); return 1; }
        b[i].dmabuf = -1;
        if (gpu && drmPrimeHandleToFD(fd, b[i].handle, DRM_CLOEXEC | DRM_RDWR, &b[i].dmabuf)) { perror("prime export"); return 1; }
    }
    uint32_t dw = mode.hdisplay, dh = (uint32_t)((uint64_t)dw * H / W); // fit into the screen, letterboxed
    if (dh > mode.vdisplay) { dh = mode.vdisplay; dw = (uint32_t)((uint64_t)dh * W / H); }
    int32_t dx = (mode.hdisplay - dw) / 2, dy = (mode.vdisplay - dh) / 2;
    printf("%dx%d, %d threads; mode %ux%u@%u, plane %u -> %ux%u+%d+%d, paced by %s\n", W, H, threads, mode.hdisplay, mode.vdisplay,
           mode.vrefresh, plane_id, dw, dh, dx, dy, vsync ? "vblank" : "timer");

    // GPU path: a second of decoding before the show starts, into every buffer once at least, so that the GPU's
    // clock governor is up (it idles at 300 MHz) and the buffers are imported
    for (int i = 0; gpu && !stop && i < FPS; i++) {
        struct buf *cur = &b[i % NBUF];
        vc2_dec_set_nv20(dec, cur->map, cur->map + (size_t)cur->pitch * H, cur->pitch);
        vc2_dec_set_nv20_dmabuf(dec, cur->dmabuf, 0, (size_t)cur->pitch * H, cur->pitch);
        if (vc2_dec_decode(dec, clip + au_off[i % naus], au_len[i % naus], &f)) { fprintf(stderr, "decode error\n"); return 1; }
    }
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    double dec_sum = 0, dec_max = 0, gpu_sum = 0, t_begin = now(), c_begin = cpu();
    int shown = 0, late = 0;
    unsigned last_seq = 0;
    for (int i = 0; !stop && i < frames; i++) {
        struct buf *cur = &b[i % NBUF];
        double t0 = now();
        vc2_dec_set_nv20(dec, cur->map, cur->map + (size_t)cur->pitch * H, cur->pitch);
        if (gpu) vc2_dec_set_nv20_dmabuf(dec, cur->dmabuf, 0, (size_t)cur->pitch * H, cur->pitch);
        if (vc2_dec_decode(dec, clip + au_off[i % naus], au_len[i % naus], &f)) { fprintf(stderr, "decode error\n"); break; }
        double dt = now() - t0;
        dec_sum += dt; if (dt > dec_max) dec_max = dt;
        gpu_sum += vc2_dec_gpu_time(dec);
        next.tv_nsec += 1000000000 / FPS;
        if (next.tv_nsec >= 1000000000) { next.tv_sec++; next.tv_nsec -= 1000000000; }
        if (vsync) {
            drmVBlank vb = { .request = { .type = DRM_VBLANK_RELATIVE | (crtc_index << DRM_VBLANK_HIGH_CRTC_SHIFT), .sequence = 1 } };
            if (drmWaitVBlank(fd, &vb)) { perror("vblank"); break; }
            if (last_seq && vb.reply.sequence != last_seq + 1) late += vb.reply.sequence - last_seq - 1; // refreshes without a new frame
            last_seq = vb.reply.sequence;
        } else {
            struct timespec t;
            clock_gettime(CLOCK_MONOTONIC, &t);
            if (t.tv_sec > next.tv_sec || (t.tv_sec == next.tv_sec && t.tv_nsec > next.tv_nsec)) late++;
            else clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, 0);
        }
        if (drmModeSetPlane(fd, plane_id, crtc_id, cur->fb, 0, dx, dy, dw, dh, 0, 0, W << 16, H << 16)) { perror("setplane"); break; }
        if (++shown % 500 == 0) {
            printf("frames %d missed %d decode avg %.2f ms max %.2f ms\n", shown, late, dec_sum / shown * 1e3, dec_max * 1e3);
            fflush(stdout);
        }
    }
    double wall = now() - t_begin;
    printf("done: frames %d in %.1f s (%.2f fps), missed vblanks %d, decode avg %.2f ms max %.2f ms, cpu %.2f ms/frame (%.2f cores)\n", shown, wall,
           shown / wall, late, dec_sum / (shown ? shown : 1) * 1e3, dec_max * 1e3, (cpu() - c_begin) / (shown ? shown : 1) * 1e3, (cpu() - c_begin) / wall);
    if (vc2_dec_gpu_active(dec)) printf("gpu path %s, gpu busy %.2f ms/frame\n", vc2_dec_gpu_active(dec) == VC2_GPU_VULKAN ? "Vulkan" : "OpenCL", gpu_sum / (shown ? shown : 1) * 1e3);
    drmModeSetPlane(fd, plane_id, crtc_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    drmDropMaster(fd);
    return 0;
}
