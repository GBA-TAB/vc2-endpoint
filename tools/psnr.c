// psnr: compares two planar 16-bit yuv files (4:2:2). Prints the PSNR of Y, Cb, Cr over all frames (mean MSE).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: psnr a.yuv b.yuv width height [bits=10]\n"); return 2; }
    int w = atoi(argv[3]), h = atoi(argv[4]), bits = argc > 5 ? atoi(argv[5]) : 10;
    FILE *a = fopen(argv[1], "rb"), *b = fopen(argv[2], "rb");
    if (!a || !b) { perror("open"); return 1; }
    size_t n[3] = { (size_t)w * h, (size_t)w / 2 * h, (size_t)w / 2 * h };
    uint16_t *x = malloc(n[0] * 2), *y = malloc(n[0] * 2);
    double se[3] = { 0 }, cnt[3] = { 0 };
    int frames = 0, same = 1;
    for (;; frames++) {
        int c;
        for (c = 0; c < 3; c++) {
            if (fread(x, 2, n[c], a) != n[c] || fread(y, 2, n[c], b) != n[c]) break;
            uint64_t s = 0;
            for (size_t i = 0; i < n[c]; i++) { int d = (int)x[i] - y[i]; s += (uint64_t)(d * d); }
            se[c] += s; cnt[c] += n[c];
            if (s) same = 0;
        }
        if (c < 3) break;
    }
    double peak = (1 << bits) - 1;
    printf("frames %d", frames);
    for (int c = 0; c < 3; c++) printf(" %s %.3f", c == 0 ? "Y" : c == 1 ? "Cb" : "Cr", se[c] ? 10 * log10(peak * peak * cnt[c] / se[c]) : 99.0);
    printf("%s\n", same && frames ? " identical" : "");
    return 0;
}
