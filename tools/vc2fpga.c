// vc2fpga: the H2C stream for a hardware VC-2 decoder (docs/FPGA.md section 9), and its check.
//   vc2fpga in.vc2 [out.h2c [pictures]]
// Packs every access unit into a descriptor + slice bytes (written to out.h2c, the bytes the host sends the card),
// then decodes each one twice: from the original stream, and from a stream the card model rebuilds out of the
// descriptor and slice bytes alone. The pictures must be identical. Prints one line per picture.
#include "../lib/vc2_fpga.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int same_frame(const vc2_frame *a, const vc2_frame *b)
{
    if (a->width != b->width || a->height != b->height || a->chroma != b->chroma) return 0;
    for (int c = 0; c < 3; c++) {
        int w = c && a->chroma ? a->width / 2 : a->width, h = c && a->chroma == 2 ? a->height / 2 : a->height;
        for (int y = 0; y < h; y++)
            if (memcmp(a->plane[c] + (size_t)y * a->stride[c], b->plane[c] + (size_t)y * b->stride[c], (size_t)w * 2)) return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: vc2fpga in.vc2 [out.h2c [pictures]]\n"); return 2; }
    FILE *fi = fopen(argv[1], "rb"), *fo = argc > 2 ? fopen(argv[2], "wb") : 0;
    if (!fi || (argc > 2 && !fo)) { perror("open"); return 1; }
    fseek(fi, 0, SEEK_END);
    size_t len = ftell(fi);
    rewind(fi);
    uint8_t *buf = calloc(1, len + VC2_INPUT_PADDING);
    if (fread(buf, 1, len, fi) != len) return 1;
    vc2_dec *da = vc2_dec_new(1, 0), *db = vc2_dec_new(1, 0);
    vc2_fpga_tx tx = { 0 };
    size_t pos = 0, n, cap = 0, total = 0;
    uint8_t *h2c = 0, *au = 0;
    int frames = 0, bad = 0, skipped = 0, maxp = argc > 3 ? atoi(argv[3]) : 0;
    while ((!maxp || frames < maxp) && (n = vc2_access_unit(buf + pos, len - pos))) {
        if (n + 256 > cap) {
            cap = n + 256;
            free(h2c); free(au);
            h2c = calloc(1, cap); au = calloc(1, cap + VC2_INPUT_PADDING);
        }
        const char *why;
        size_t m = vc2_fpga_pack(&tx, buf + pos, n, h2c, cap, &why);
        if (!m) {
            printf("frame %d: not for the card: %s\n", frames, why);
            skipped++;
        } else {
            vc2_fpga_desc d;
            vc2_fpga_desc_read(h2c, m, &d);
            size_t k = vc2_fpga_model_au(h2c, m, au, cap);
            vc2_frame fa, fb;
            int ok = k && !vc2_dec_decode(da, buf + pos, n, &fa) && !vc2_dec_decode(db, au, k, &fb) && same_frame(&fa, &fb);
            printf("frame %d: %s  h2c %zu bytes (slices %u)  pic %u  seq %u%s  %dx%d slices %dx%d scaler %d\n", frames,
                   ok ? "ok" : "MISMATCH", m, d.slice_bytes, d.picture_number, d.seq_count,
                   d.flags & VC2_FPGA_NEW_SEQ ? " NEW_SEQ" : "", d.width, d.height, d.slices_x, d.slices_y, d.size_scaler);
            bad += !ok;
            if (fo && fwrite(h2c, 1, m, fo) != m) { perror("write"); return 1; }
            total += m;
        }
        pos += n; frames++;
    }
    if (fo) fclose(fo);
    fprintf(stderr, "%d pictures, %d not for the card, %d mismatches, %zu H2C bytes\n", frames, skipped, bad, total);
    return frames && !bad ? (skipped ? 3 : 0) : 1;
}
