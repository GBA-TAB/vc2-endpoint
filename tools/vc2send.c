// vc2send: sends a VC-2 file as RTP (RFC 8450) over UDP, paced.
//   vc2send [--dst 127.0.0.1] [--port 5004] [--fps 50] [--frames 500] [--mtu 1200] [--spread 1.0] [--loss permille]
//           [--reorder permille] [--cpu N] [--sdp] file.vc2
// --dst: loopback by default. A unicast address outside 127/8 needs --network as well (media on a real network is
// a decision for the network's owner); multicast, broadcast and the lab media network 172.30.3.0/24 (ST 2110,
// PTP) are always refused.
// --spread: the fraction of the frame period over which a frame's packets are sent (1 = constant rate).
// --loss / --reorder: drop packets, or swap a packet with the next, at random (seeded, repeatable).
#define _GNU_SOURCE
#include "../lib/vc2.h"
#include <arpa/inet.h>
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

int main(int argc, char **argv)
{
    int port = 5004, frames = 500, mtu = 1200, loss = 0, reorder = 0, sdp = 0, network = 0, pin = -1;
    double fps = 50, spread = 1;
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
        else in = a;
    }
    if (!in) { fprintf(stderr, "usage: vc2send [--dst A [--network]] [--port 5004] [--fps 50] [--frames N] [--mtu 1200] [--spread 1.0] [--loss permille] [--reorder permille] [--cpu N] [--sdp] file.vc2\n"); return 2; }
    struct sockaddr_in to = { .sin_family = AF_INET, .sin_port = htons(port) };
    if (inet_pton(AF_INET, dst, &to.sin_addr) != 1) { fprintf(stderr, "--dst: not an IPv4 address\n"); return 2; }
    uint32_t ip = ntohl(to.sin_addr.s_addr);
    if (ip >> 28 == 14 || ip >> 28 == 15 || (ip & 0xff) == 255 || (ip & 0xff) == 0 || ip >> 8 == (172u << 16 | 30 << 8 | 3)) {
        fprintf(stderr, "multicast, broadcast and the lab media network 172.30.3.0/24 are refused\n"); return 2;
    }
    if (ip >> 24 != 127 && !network) { fprintf(stderr, "a destination outside loopback needs --network\n"); return 2; }
    if (pin >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(pin, &set); sched_setaffinity(0, sizeof set, &set); }
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
