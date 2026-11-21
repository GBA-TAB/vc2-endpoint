// vc2recv: receives a VC-2 RTP stream (RFC 8450) on a loopback UDP port, decodes it as the packets arrive and
// reports the latency. Binds 127.0.0.1 only.
//   vc2recv [--port 5004] [--frames 500] [--reorder 8] [--cpu N] [--out f.yuv] [--out-frames 4] [--out-skip 0]
// Latency per frame: from the arrival of its first packet to its last line being decoded, and from the
// arrival of its last packet to the same.
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
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv)
{
    int port = 5004, frames = 500, reorder = 8, pin = -1, out_frames = 4, out_skip = 0;
    const char *out = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--port")) { port = atoi(v); i++; }
        else if (!strcmp(a, "--frames")) { frames = atoi(v); i++; }
        else if (!strcmp(a, "--reorder")) { reorder = atoi(v); i++; }
        else if (!strcmp(a, "--cpu")) { pin = atoi(v); i++; }
        else if (!strcmp(a, "--out")) { out = v; i++; }
        else if (!strcmp(a, "--out-frames")) { out_frames = atoi(v); i++; }
        else if (!strcmp(a, "--out-skip")) { out_skip = atoi(v); i++; }
    }
    if (pin >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(pin, &set); sched_setaffinity(0, sizeof set, &set); }
    int fd = socket(AF_INET, SOCK_DGRAM, 0), rcvbuf = 8 << 20;
    struct sockaddr_in me = { .sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = htonl(INADDR_LOOPBACK) };
    struct timeval tv = { 0, 100000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUFFORCE, &rcvbuf, sizeof rcvbuf); // beyond net.core.rmem_max: needs CAP_NET_ADMIN
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (bind(fd, (struct sockaddr *)&me, sizeof me)) { perror("bind"); return 1; }
    vc2_dec *d = vc2_dec_new(1, 0);
    vc2_rtp_rx *rx = vc2_rtp_rx_new(d, reorder);
    FILE *fo = out ? fopen(out, "wb") : 0;
    uint8_t pkt[9216];
    uint32_t cur_ts = 0;
    double t_first = 0, l1 = 0, l1max = 0, l2 = 0, l2max = 0, c0 = cpu(), t_begin = 0;
    int got = 0, idle = 0, any = 0;
    uint8_t *keep = 0;
    size_t keep_n = 0;
    while (got < frames) {
        ssize_t n = recv(fd, pkt, sizeof pkt, 0);
        double t = now();
        vc2_frame f;
        int done;
        if (n < 16) { // timeout: close what is open; stop after 3 s of silence once the stream has started
            done = vc2_rtp_rx_flush(rx, &f);
            if (any && ++idle > 30) break;
        } else {
            uint32_t ts = (uint32_t)pkt[4] << 24 | pkt[5] << 16 | pkt[6] << 8 | pkt[7];
            if (!any || ts != cur_ts) { cur_ts = ts; t_first = t; }
            if (!any) { any = 1; t_begin = t; c0 = cpu(); }
            idle = 0;
            done = vc2_rtp_rx_packet(rx, pkt, n, &f);
        }
        if (!done) continue;
        double t2 = now(), a = t2 - t_first, b = t2 - t;
        // a frame closed by the first packet of the next one (its end was lost) is timed from that packet
        if (got > 0) { l1 += a; if (a > l1max) l1max = a; l2 += b; if (b > l2max) l2max = b; }
        if (fo && got >= out_skip && got < out_skip + out_frames) { // kept in memory and written at the end, not to stall the receiver
            size_t fsz = ((size_t)f.width * f.height + 2 * (size_t)(f.chroma ? f.width / 2 : f.width) * (f.chroma == 2 ? f.height / 2 : f.height)) * 2;
            if (!keep) keep = malloc(fsz * out_frames);
            for (int c = 0; c < 3; c++) {
                int w = c && f.chroma ? f.width / 2 : f.width, h = c && f.chroma == 2 ? f.height / 2 : f.height;
                for (int y = 0; y < h; y++) { memcpy(keep + keep_n, f.plane[c] + (size_t)y * f.stride[c], (size_t)w * 2); keep_n += (size_t)w * 2; }
            }
        }
        got++;
    }
    const vc2_rtp_stats *s = vc2_rtp_rx_stats(rx);
    int m = got > 1 ? got - 1 : 1;
    printf("frames %d  packets %llu  lost %llu  late %llu  concealed slices %llu  dropped frames %llu\n", got, (unsigned long long)s->packets,
           (unsigned long long)s->lost, (unsigned long long)s->late, (unsigned long long)s->concealed_slices, (unsigned long long)s->dropped_frames);
    printf("first packet -> last line: avg %.2f max %.2f ms; last packet -> last line: avg %.2f max %.2f ms; cpu %.2f ms/frame over %.1f s\n",
           l1 / m * 1e3, l1max * 1e3, l2 / m * 1e3, l2max * 1e3, (cpu() - c0) / (got ? got : 1) * 1e3, now() - t_begin);
    if (fo) { fwrite(keep, 1, keep_n, fo); fclose(fo); }
    return 0;
}
