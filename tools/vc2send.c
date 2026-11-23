// vc2send: sends a VC-2 file as RTP (RFC 8450) over UDP, paced.
//   vc2send [--dst 127.0.0.1] [--port 5004] [--fps 50] [--frames 500] [--mtu 1200] [--spread 1.0] [--loss permille]
//           [--reorder permille] [--cpu N] [--sdp] file.vc2
//   vc2send --encode WxH [--bpp 3] [--threads N] [--gpu cl] [same options] file.yuv
// --encode: the input is planar 16-bit 4:2:2 (as vc2enc reads it), looped; every frame is encoded live (slices
// capped to the MTU) and sent in its slot, and the encoding time and frames that miss their slot are reported.
// --dst: loopback by default. A unicast address outside 127/8 needs --network as well (media on a real network is
// a decision for the network's owner); multicast, broadcast and the lab media network 172.30.3.0/24 (ST 2110,
// PTP) are always refused.
// --spread: the fraction of the frame period over which a frame's packets are sent (1 = constant rate).
// --loss / --reorder: drop packets, or swap a packet with the next, at random (seeded, repeatable).
#define _GNU_SOURCE
#include "../lib/vc2.h"
#include <arpa/inet.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void sleep_until(double t)
{
    struct timespec ts = { (time_t)t, (long)((t - (time_t)t) * 1e9) };
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, 0);
}
struct pk { uint8_t d[9216]; size_t n; };
static struct pk *q;
static int nq;
static int collect(void *ctx, const uint8_t *p, size_t n) { (void)ctx; memcpy(q[nq].d, p, n); q[nq++].n = n; return 0; }

// Live: every frame is encoded at its slot, then handed to the sending thread, which paces its packets over
// `spread` of the period while the next frame is encoded. A frame whose encoding ends after the next slot is late.
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct pk *buf[2];
    int n[2], full[2], stop, fd;
    double spread, fps;
    const struct sockaddr_in *to;
    long sent;
} sender;
static void *send_main(void *arg)
{
    sender *s = arg;
    for (int k = 0;; k ^= 1) {
        pthread_mutex_lock(&s->mu);
        while (!s->full[k] && !s->stop) pthread_cond_wait(&s->cv, &s->mu);
        if (!s->full[k]) { pthread_mutex_unlock(&s->mu); break; }
        pthread_mutex_unlock(&s->mu);
        double ts = now();
        for (int i = 0; i < s->n[k]; i++) {
            if (i % 8 == 0) sleep_until(ts + s->spread * i / s->n[k] / s->fps);
            sendto(s->fd, s->buf[k][i].d, s->buf[k][i].n, 0, (const struct sockaddr *)s->to, sizeof *s->to);
            s->sent++;
        }
        pthread_mutex_lock(&s->mu);
        s->full[k] = 0;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
    }
    return 0;
}
static int send_live(const char *in, int w, int h, double bpp, int threads, int gpu, double fps, int frames, int mtu,
                     double spread, const struct sockaddr_in *to)
{
    const int cw = w / 2;
    const size_t fsz = ((size_t)w * h + 2 * (size_t)cw * h) * 2;
    FILE *fi = fopen(in, "rb");
    if (!fi) { perror(in); return 1; }
    fseek(fi, 0, SEEK_END);
    int nf = (int)(ftell(fi) / fsz);
    rewind(fi);
    if (nf < 1) { fprintf(stderr, "no frame of %dx%d in %s\n", w, h, in); return 1; }
    if (nf > 16) nf = 16;
    uint16_t *yuv;
    if (posix_memalign((void **)&yuv, 4096, fsz * nf) || fread(yuv, fsz, nf, fi) != (size_t)nf) return 1;
    fclose(fi);
    vc2_enc_cfg c;
    vc2_enc_defaults(&c, w, h);
    c.bpp = bpp; c.threads = threads; c.max_slice_bytes = mtu - VC2_RTP_OVERHEAD;
    c.fr_num = (int)fps;
    vc2_enc *e = vc2_enc_new(&c);
    if (!e) { fprintf(stderr, "bad settings\n"); return 1; }
    if (gpu && vc2_enc_set_gpu(e, gpu < 0 ? 99 : gpu)) { fprintf(stderr, "--gpu: not available here\n"); return 1; }
    uint8_t *au = malloc(vc2_enc_max_size(e));
    sender sd = { .spread = spread, .fps = fps, .to = to };
    pthread_mutex_init(&sd.mu, 0);
    pthread_cond_init(&sd.cv, 0);
    sd.fd = socket(AF_INET, SOCK_DGRAM, 0);
    int sndbuf = 4 << 20;
    setsockopt(sd.fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);
    sd.buf[0] = malloc(sizeof(struct pk) * 16384); sd.buf[1] = malloc(sizeof(struct pk) * 16384);
    pthread_t th;
    pthread_create(&th, 0, send_main, &sd);
    vc2_rtp_tx *tx = vc2_rtp_tx_new(96, 0x56433200, mtu);
    struct timespec rt;
    clock_gettime(CLOCK_REALTIME, &rt);
    uint64_t t90 = (uint64_t)rt.tv_sec * 90000 + (uint64_t)rt.tv_nsec * 9 / 100000;
    double t0 = now() + 0.1, esum = 0, emax = 0, wait = 0;
    long late = 0;
    size_t bytes = 0;
    for (int f = 0; f < frames; f++) {
        const double slot = t0 + f / fps;
        sleep_until(slot); // the frame "arrives" at its slot
        const uint16_t *fr = yuv + (fsz / 2) * (f % nf);
        const uint16_t *pl[3] = { fr, fr + (size_t)w * h, fr + (size_t)w * h + (size_t)cw * h };
        const ptrdiff_t st[3] = { w, cw, cw };
        double te = now();
        size_t n = vc2_enc_encode(e, pl, st, au);
        double ed = now() - te;
        esum += ed; if (ed > emax) emax = ed;
        if (now() > slot + 1 / fps) late++;
        const int k = f & 1;
        double tw = now();
        pthread_mutex_lock(&sd.mu);
        while (sd.full[k]) pthread_cond_wait(&sd.cv, &sd.mu); // the sender still on the frame two back
        pthread_mutex_unlock(&sd.mu);
        wait += now() - tw;
        q = sd.buf[k]; nq = 0;
        if (vc2_rtp_tx_frame(tx, au, n, (uint32_t)(t90 + (uint64_t)(f * 90000.0 / fps)), collect, 0) < 0) {
            fprintf(stderr, "a slice does not fit a packet\n");
            return 1;
        }
        pthread_mutex_lock(&sd.mu);
        sd.n[k] = nq; sd.full[k] = 1;
        pthread_cond_broadcast(&sd.cv);
        pthread_mutex_unlock(&sd.mu);
        bytes += n;
    }
    pthread_mutex_lock(&sd.mu);
    sd.stop = 1;
    pthread_cond_broadcast(&sd.cv);
    pthread_mutex_unlock(&sd.mu);
    pthread_join(th, 0);
    printf("live: %d frames %dx%d at %.0f fps, %.2f bpp, encode avg %.2f ms max %.2f ms, %ld late, waited for the sender %.1f ms, %ld packets (%.0f per frame)%s\n",
           frames, w, h, fps, bytes * 8.0 / frames / w / h, esum / frames * 1e3, emax * 1e3, late, wait * 1e3, sd.sent, (double)sd.sent / frames,
           gpu ? ", GPU front end" : "");
    close(sd.fd);
    vc2_enc_free(e);
    return 0;
}

int main(int argc, char **argv)
{
    int port = 5004, frames = 500, mtu = 1200, loss = 0, reorder = 0, sdp = 0, network = 0, pin = -1;
    int ew = 0, eh = 0, ethreads = 1, egpu = 0;
    double fps = 50, spread = 1, ebpp = 3;
    const char *in = 0, *dst = "127.0.0.1";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--port")) { port = atoi(v); i++; }
        else if (!strcmp(a, "--fps")) { fps = atof(v); i++; }
        else if (!strcmp(a, "--frames")) { frames = atoi(v); i++; }
        else if (!strcmp(a, "--mtu")) { mtu = atoi(v); i++; }
        else if (!strcmp(a, "--spread")) { spread = atof(v); i++; }
        else if (!strcmp(a, "--loss")) { loss = atoi(v); i++; }
        else if (!strcmp(a, "--reorder")) { reorder = atoi(v); i++; }
        else if (!strcmp(a, "--dst")) { dst = v; i++; }
        else if (!strcmp(a, "--network")) network = 1;
        else if (!strcmp(a, "--cpu")) { pin = atoi(v); i++; }
        else if (!strcmp(a, "--sdp")) sdp = 1;
        else if (!strcmp(a, "--encode")) { if (sscanf(v, "%dx%d", &ew, &eh) != 2) ew = 0; i++; }
        else if (!strcmp(a, "--bpp")) { ebpp = atof(v); i++; }
        else if (!strcmp(a, "--threads")) { ethreads = atoi(v); i++; }
        else if (!strcmp(a, "--gpu")) { egpu = !strcmp(v, "cl") ? VC2_GPU_OPENCL : -1; i++; }
        else in = a;
    }
    if (!in) { fprintf(stderr, "usage: vc2send [--dst A [--network]] [--port 5004] [--fps 50] [--frames N] [--mtu 1200] [--spread 1.0] [--loss permille] [--reorder permille] [--cpu N] [--sdp] file.vc2\n"
                               "       vc2send --encode WxH [--bpp 3] [--threads N] [--gpu cl] [options as above] file.yuv\n"); return 2; }
    struct sockaddr_in to = { .sin_family = AF_INET, .sin_port = htons(port) };
    if (inet_pton(AF_INET, dst, &to.sin_addr) != 1) { fprintf(stderr, "--dst: not an IPv4 address\n"); return 2; }
    uint32_t ip = ntohl(to.sin_addr.s_addr);
    if (ip >> 28 == 14 || ip >> 28 == 15 || (ip & 0xff) == 255 || (ip & 0xff) == 0 || ip >> 8 == (172u << 16 | 30 << 8 | 3)) {
        fprintf(stderr, "multicast, broadcast and the lab media network 172.30.3.0/24 are refused\n"); return 2;
    }
    if (ip >> 24 != 127 && !network) { fprintf(stderr, "a destination outside loopback needs --network\n"); return 2; }
    if (pin >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(pin, &set); sched_setaffinity(0, sizeof set, &set); }
    if (ew > 0) return send_live(in, ew, eh, ebpp, ethreads, egpu, fps, frames, mtu, spread, &to);
    FILE *fi = fopen(in, "rb");
    if (!fi) { perror(in); return 1; }
    fseek(fi, 0, SEEK_END);
    size_t len = ftell(fi), pos = 0, n;
    rewind(fi);
    uint8_t *buf = calloc(1, len + VC2_INPUT_PADDING);
    if (fread(buf, 1, len, fi) != len) return 1;
    size_t au_off[4096], au_len[4096];
    int naus = 0;
    while (naus < 4096 && (n = vc2_access_unit(buf + pos, len - pos))) { au_off[naus] = pos; au_len[naus++] = n; pos += n; }
    if (!naus) { fprintf(stderr, "no pictures\n"); return 1; }
    if (sdp) {
        int code;
        vc2_seq s;
        char text[1024];
        size_t u = vc2_unit(buf, len, &code);
        if (code == VC2_PC_SEQ_HEADER && !vc2_parse_seq(buf + 13, u - 13, &s)) {
            vc2_rtp_sdp(text, sizeof text, dst, port, 96, &s, au_len[0] * 8.0 / s.width / s.height, 0);
            fputs(text, stdout);
        }
        return 0;
    }
    int fd = socket(AF_INET, SOCK_DGRAM, 0), sndbuf = 4 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);
    q = malloc(sizeof *q * 16384);
    vc2_rtp_tx *tx = vc2_rtp_tx_new(96, 0x56433200, mtu);
    struct timespec rt;
    clock_gettime(CLOCK_REALTIME, &rt); // the media clock: here the system clock stands in for PTP time
    uint64_t t90 = (uint64_t)rt.tv_sec * 90000 + (uint64_t)rt.tv_nsec * 9 / 100000;
    srand(1);
    double t0 = now() + 0.05;
    long sent = 0, dropped = 0, swapped = 0;
    for (int f = 0; f < frames; f++) {
        nq = 0;
        if (vc2_rtp_tx_frame(tx, buf + au_off[f % naus], au_len[f % naus], (uint32_t)(t90 + (uint64_t)(f * 90000.0 / fps)), collect, 0) < 0) {
            fprintf(stderr, "a slice does not fit a packet of %d bytes: encode with --max-slice %d\n", mtu, mtu - VC2_RTP_OVERHEAD);
            return 1;
        }
        for (int i = 0; i < nq; i++)
            if (reorder && i + 1 < nq && rand() % 1000 < reorder) { struct pk t = q[i]; q[i] = q[i + 1]; q[i + 1] = t; swapped++; i++; }
        double tf = t0 + f / fps;
        for (int i = 0; i < nq; i++) {
            if (i % 8 == 0) sleep_until(tf + spread * i / nq / fps);
            if (loss && rand() % 1000 < loss) { dropped++; continue; }
            sendto(fd, q[i].d, q[i].n, 0, (struct sockaddr *)&to, sizeof to);
            sent++;
        }
    }
    printf("sent %ld packets (%.0f per frame), dropped %ld, swapped %ld\n", sent, (double)(sent + dropped) / frames, dropped, swapped);
    close(fd);
    return 0;
}
