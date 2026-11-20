// vc2enc: planar yuv (16-bit little endian samples, 4:2:2 by default) -> VC-2 HQ stream.
#define _GNU_SOURCE
#include "../lib/vc2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv)
{
    int w = 1920, h = 1080, loops = 1, quiet = 0;
    const char *in = 0, *out = 0;
    vc2_enc_cfg c;
    vc2_enc_defaults(&c, w, h);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "-s")) { sscanf(v, "%dx%d", &w, &h); i++; }
        else if (!strcmp(a, "--bpp")) { c.bpp = atof(v); i++; }
        else if (!strcmp(a, "--wavelet")) { c.wavelet = atoi(v); i++; }
        else if (!strcmp(a, "--depth")) { c.depth = atoi(v); i++; }
        else if (!strcmp(a, "--slice")) { sscanf(v, "%dx%d", &c.slice_w, &c.slice_h); i++; }
        else if (!strcmp(a, "--rc-rows")) { c.rc_rows = atoi(v); i++; }
        else if (!strcmp(a, "--tail")) { c.tail_lambda = atof(v); i++; }
        else if (!strcmp(a, "--q")) { c.fixed_q = atoi(v); i++; }
        else if (!strcmp(a, "--no-truncate")) c.no_truncate = 1;
        else if (!strcmp(a, "--fps")) { c.fr_num = atoi(v); i++; }
        else if (!strcmp(a, "--threads")) { c.threads = atoi(v); i++; }
        else if (!strcmp(a, "--chroma")) { c.chroma = atoi(v); i++; }
        else if (!strcmp(a, "--loops")) { loops = atoi(v); i++; }
        else if (!strcmp(a, "--qm")) { // LL,HL1,LH1,HH1,HL2,...
            c.qm_custom = 1;
            char *s = strdup(v), *t = strtok(s, ",");
            for (int l = 0; l <= VC2_MAX_DEPTH && t; l++)
                for (int o = l ? 1 : 0; o < (l ? 4 : 1) && t; o++, t = strtok(0, ",")) c.qm[l][o] = atoi(t);
            i++;
        }
        else if (!strcmp(a, "-q")) quiet = 1;
        else if (!in) in = a;
        else if (!out) out = a;
    }
    if (!in || !out) {
        fprintf(stderr, "usage: vc2enc [-s WxH] [--bpp 3] [--wavelet 1] [--depth 4] [--slice 128x16] [--rc-rows 0] [--tail 0.2]\n"
                        "              [--fps 50] [--threads 1] [--chroma 1] [--loops 1] [--qm list] [--q N] [--no-truncate] in.yuv out.vc2\n"
                        "  wavelets: 0 DD 9,7  1 LeGall 5,3  2 DD 13,7  3 Haar  4 Haar (shift)  6 Daubechies 9,7\n");
        return 2;
    }
    c.width = w; c.height = h;
    int cw = c.chroma ? w / 2 : w, ch = c.chroma == 2 ? h / 2 : h;
    size_t fsz = ((size_t)w * h + 2 * (size_t)cw * ch) * 2;
    FILE *fi = fopen(in, "rb"), *fo = fopen(out, "wb");
    if (!fi || !fo) { perror("open"); return 1; }
    vc2_enc *e = vc2_enc_new(&c);
    if (!e) { fprintf(stderr, "bad settings\n"); return 1; }
    uint16_t *frame = malloc(fsz);
    uint8_t *buf = malloc(vc2_enc_max_size(e));
    const uint16_t *pl[3] = { frame, frame + (size_t)w * h, frame + (size_t)w * h + (size_t)cw * ch };
    ptrdiff_t st[3] = { w, cw, cw };
    int frames = 0;
    size_t bytes = 0;
    double t_enc = 0, c_enc = 0, qs = 0;
    for (int l = 0; l < loops; l++) {
        rewind(fi);
        while (fread(frame, 1, fsz, fi) == fsz) {
            double t0 = now(), c0 = cpu();
            size_t n = vc2_enc_encode(e, pl, st, buf);
            t_enc += now() - t0; c_enc += cpu() - c0;
            qs += vc2_enc_last_q(e) / 100.0;
            if (l == 0) fwrite(buf, 1, n, fo);
            bytes += n; frames++;
        }
    }
    fwrite(buf, 1, vc2_enc_end(e, buf), fo);
    if (!quiet)
        printf("frames %d  bytes/frame %zu  bpp %.3f  q %.2f  encode %.2f ms/frame wall, %.2f ms CPU (%.1f fps)\n", frames,
               bytes / (frames ? frames : 1), bytes * 8.0 / frames / w / h, qs / frames, t_enc / frames * 1e3, c_enc / frames * 1e3, frames / t_enc);
    fclose(fo);
    return 0;
}
