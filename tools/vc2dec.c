// vc2dec: VC-2 HQ stream -> planar yuv (16-bit little endian samples).
#include "../lib/vc2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: vc2dec in.vc2 out.yuv [threads]\n"); return 2; }
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) { perror("open"); return 1; }
    fseek(fi, 0, SEEK_END);
    size_t len = ftell(fi);
    rewind(fi);
    uint8_t *buf = calloc(1, len + VC2_INPUT_PADDING);
    if (fread(buf, 1, len, fi) != len) return 1;
    vc2_dec *d = vc2_dec_new(argc > 3 ? atoi(argv[3]) : 1, 0);
    size_t pos = 0, n;
    int frames = 0;
    while ((n = vc2_access_unit(buf + pos, len - pos))) {
        vc2_frame f;
        if (vc2_dec_decode(d, buf + pos, n, &f)) { fprintf(stderr, "decode error at frame %d\n", frames); return 1; }
        for (int c = 0; c < 3; c++) {
            int w = c && f.chroma ? f.width / 2 : f.width, h = c && f.chroma == 2 ? f.height / 2 : f.height;
            for (int y = 0; y < h; y++) fwrite(f.plane[c] + (size_t)y * f.stride[c], 2, w, fo);
        }
        pos += n; frames++;
    }
    fprintf(stderr, "%d frames\n", frames);
    fclose(fo);
    return frames ? 0 : 1;
}
