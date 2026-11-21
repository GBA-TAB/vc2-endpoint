// vc2dump: a plain whole-frame VC-2 HQ decoder that writes every stage to files, as reference vectors for a
// hardware decoder (docs/FPGA.md). Scalar, 32-bit, bit-serial entropy decoding; nothing shared with the
// production decoder's cascade except the tables, the geometry and the lifting kernel (lib/vc2_lift.h, 32 bits).
//
//   vc2dump in.vc2 outdir [frames]
//
// Per frame, in outdir/fNNN/ (all binary files little endian):
//   params.txt      sequence and transform parameters, the quantisation matrix, the slice geometry
//   slices.csv      per slice: sy, sx, byte offset in the picture payload, qindex, then per component its
//                   length in bytes and its number of coefficients
//   coef_q.bin      per slice in raster order, per component Y, Cb, Cr: the quantised coefficients in stream
//                   order (LL, then per level 1..depth HL, LH, HH; raster order inside each block), int32
//   coef_dq.bin     the same after dequantisation, int32
//   band_l<L>_<o>_c<C>.bin   the whole subband after dequantisation, int32, bw[L] x bh[L] (o: LL, HL, LH, HH)
//   synth_l<L>_c<C>.bin      the output of the inverse transform of level L (L < depth), int32, 2bw[L] x 2bh[L]
//                            = the LL input of level L + 1, including the padding
//   picture.yuv     the final picture: offset, clipped, cropped, planar uint16
//   ranges.txt      the largest magnitude per subband (quantised, dequantised) and per level output
// The final picture must equal vc2dec's output (scripts/fpga-vectors.sh checks that for every frame).
#include "../lib/vc2_priv.h"
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

#define T int32_t
#define SUF _32
#include "../lib/vc2_lift.h"
#undef T
#undef SUF

#define NL (VC2_MAX_DEPTH + 1)
static const char *oname[4] = { "LL", "HL", "LH", "HH" };

// --- bit-serial reader: bits after the end of the component read as 1 ---------------------------------------
typedef struct { const uint8_t *p; size_t nbits, pos; } bits_t;
static int bit(bits_t *b)
{
    if (b->pos >= b->nbits) { b->pos++; return 1; }
    int v = (b->p[b->pos >> 3] >> (7 - (b->pos & 7))) & 1;
    b->pos++;
    return v;
}
// interleaved exp-Golomb: (0, data bit)* then 1; a sign bit follows a nonzero value (1 = negative)
static int32_t read_sint(bits_t *b)
{
    uint32_t v = 1;
    while (!bit(b)) {
        v = (v << 1) | bit(b);
        if (v > (1u << 30)) break; // not a valid stream
    }
    v -= 1;
    if (!v) return 0;
    return bit(b) ? -(int32_t)v : (int32_t)v;
}

static int dequant(int32_t a, int q)
{
    uint32_t m = a < 0 ? -(uint32_t)a : (uint32_t)a;
    int32_t c = (int32_t)(((uint64_t)m * vc2_qf[q] + vc2_qo[q]) >> 2);
    return a ? (a < 0 ? -c : c) : 0;
}

static void wbin(const char *dir, const char *name, const void *p, size_t n)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 1, n, f) != n) { perror(path); exit(1); }
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: vc2dump in.vc2 outdir [frames]\n"); return 2; }
    int maxframes = argc > 3 ? atoi(argv[3]) : 1 << 30;
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror(argv[1]); return 1; }
    fseek(fi, 0, SEEK_END);
    size_t len = ftell(fi);
    rewind(fi);
    uint8_t *buf = calloc(1, len + VC2_INPUT_PADDING);
    if (fread(buf, 1, len, fi) != len) return 1;
    fclose(fi);
    if (mkdir(argv[2], 0755) && errno != EEXIST) { perror(argv[2]); return 1; }
    vc2_tables_init();

    vc2_seq seq;
    int have_seq = 0, frame = 0;
    size_t pos = 0;
    while (pos < len && frame < maxframes) {
        int code;
        size_t n = vc2_unit(buf + pos, len - pos, &code);
        if (!n) break;
        const uint8_t *pl = buf + pos + 13;
        size_t plen = n - 13;
        if (code == VC2_PC_SEQ_HEADER) {
            if (vc2_parse_seq(pl, plen, &seq)) { fprintf(stderr, "bad sequence header\n"); return 1; }
            have_seq = 1;
        } else if (code == VC2_PC_PICTURE_HQ) {
            if (!have_seq) { fprintf(stderr, "picture before a sequence header\n"); return 1; }
            vc2_pic pic;
            if (vc2_parse_pic(pl, plen, &seq, &pic)) { fprintf(stderr, "bad transform parameters\n"); return 1; }
            const vc2_wavelet *wl = vc2_wavelet_get(pic.wavelet);
            const int depth = pic.depth, nx = pic.slices_x, ny = pic.slices_y;
            char dir[512];
            snprintf(dir, sizeof dir, "%s/f%03d", argv[2], frame);
            if (mkdir(dir, 0755) && errno != EEXIST) { perror(dir); return 1; }

            vc2_geom g[3];
            int *xs[3][NL], *ys[3][NL];
            int32_t *band[3][NL][4];
            int64_t maxq[3][NL][4] = { { { 0 } } }, maxdq[3][NL][4] = { { { 0 } } }, maxsyn[3][NL] = { { 0 } };
            int qlo = VC2_NQ, qhi = -1;
            for (int c = 0; c < 3; c++) {
                int w = c && seq.chroma ? seq.width / 2 : seq.width, h = c && seq.chroma == 2 ? seq.height / 2 : seq.height;
                vc2_geom_init(&g[c], w, h, depth);
                for (int l = 0; l <= depth; l++) {
                    xs[c][l] = malloc((nx + 1) * sizeof(int));
                    ys[c][l] = malloc((ny + 1) * sizeof(int));
                    for (int i = 0; i <= nx; i++) xs[c][l][i] = (int)((int64_t)g[c].bw[l] * i / nx);
                    for (int i = 0; i <= ny; i++) ys[c][l][i] = (int)((int64_t)g[c].bh[l] * i / ny);
                    for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++)
                        band[c][l][o] = calloc((size_t)g[c].bw[l] * g[c].bh[l], sizeof(int32_t));
                }
            }

            char path[1024];
            snprintf(path, sizeof path, "%s/params.txt", dir);
            FILE *fp = fopen(path, "w");
            fprintf(fp, "version %d.%d profile %d level %d base_format %d\n", seq.major, seq.minor, seq.profile, seq.level, seq.base_format);
            fprintf(fp, "size %dx%d chroma %d (0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0) interlaced %d bit_depth %d frame_rate %d/%d picture_coding_mode %d\n",
                    seq.width, seq.height, seq.chroma, seq.interlaced, seq.bit_depth, seq.fr_num, seq.fr_den, seq.picture_coding_mode);
            fprintf(fp, "picture_number %u wavelet %d depth %d slices %dx%d prefix_bytes %d size_scaler %d\n",
                    pic.picture_number, pic.wavelet, depth, nx, ny, pic.prefix_bytes, pic.size_scaler);
            fprintf(fp, "quant_matrix LL %d", pic.qm[0][0]);
            for (int l = 1; l <= depth; l++) fprintf(fp, " | L%d HL %d LH %d HH %d", l, pic.qm[l][1], pic.qm[l][2], pic.qm[l][3]);
            fprintf(fp, "\n");
            for (int c = 0; c < 3; c++) {
                fprintf(fp, "component %d: %dx%d padded %dx%d", c, g[c].w, g[c].h, g[c].pw, g[c].ph);
                for (int l = 0; l <= depth; l++) fprintf(fp, " | L%d band %dx%d slice %s%dx%d", l, g[c].bw[l], g[c].bh[l],
                                                         (g[c].bw[l] % nx || g[c].bh[l] % ny) ? "~" : "", g[c].bw[l] / nx, g[c].bh[l] / ny);
                fprintf(fp, "\n");
            }
            fclose(fp);

            // slices: entropy decoding and dequantisation, slice by slice
            const uint8_t *sl = pl + pic.slices_offset, *end = pl + plen;
            snprintf(path, sizeof path, "%s/slices.csv", dir);
            FILE *fcsv = fopen(path, "w");
            fprintf(fcsv, "sy,sx,offset,qindex,len_y,len_cb,len_cr,n_y,n_cb,n_cr\n");
            snprintf(path, sizeof path, "%s/coef_q.bin", dir);
            FILE *fq = fopen(path, "wb");
            snprintf(path, sizeof path, "%s/coef_dq.bin", dir);
            FILE *fdq = fopen(path, "wb");
            for (int sy = 0; sy < ny; sy++)
                for (int sx = 0; sx < nx; sx++) {
                    if (sl + pic.prefix_bytes + 4 > end) { fprintf(stderr, "frame %d: slice %d,%d past the end\n", frame, sy, sx); return 1; }
                    const uint8_t *s = sl + pic.prefix_bytes;
                    int qidx = *s++;
                    size_t clen[3];
                    int cn[3];
                    fprintf(fcsv, "%d,%d,%zu,%d", sy, sx, (size_t)(sl - pl), qidx);
                    const uint8_t *cdata[3];
                    for (int c = 0; c < 3; c++) {
                        clen[c] = (size_t)*s++ * pic.size_scaler;
                        cdata[c] = s;
                        s += clen[c];
                        fprintf(fcsv, ",%zu", clen[c]);
                    }
                    if (s > end) { fprintf(stderr, "frame %d: slice %d,%d runs past the picture\n", frame, sy, sx); return 1; }
                    for (int c = 0; c < 3; c++) {
                        bits_t b = { cdata[c], clen[c] * 8, 0 };
                        cn[c] = 0;
                        for (int l = 0; l <= depth; l++) {
                            int x0 = xs[c][l][sx], x1 = xs[c][l][sx + 1], y0 = ys[c][l][sy], y1 = ys[c][l][sy + 1];
                            for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
                                int q = qidx - pic.qm[l][o];
                                q = q < 0 ? 0 : q >= VC2_NQ ? VC2_NQ - 1 : q;
                                if (y1 > y0 && x1 > x0) { if (q < qlo) qlo = q; if (q > qhi) qhi = q; }
                                for (int y = y0; y < y1; y++)
                                    for (int x = x0; x < x1; x++) {
                                        int32_t v = read_sint(&b), dq = dequant(v, q);
                                        fwrite(&v, 4, 1, fq);
                                        fwrite(&dq, 4, 1, fdq);
                                        band[c][l][o][(size_t)y * g[c].bw[l] + x] = dq;
                                        int64_t av = v < 0 ? -(int64_t)v : v, adq = dq < 0 ? -(int64_t)dq : dq;
                                        if (av > maxq[c][l][o]) maxq[c][l][o] = av;
                                        if (adq > maxdq[c][l][o]) maxdq[c][l][o] = adq;
                                        cn[c]++;
                                    }
                            }
                        }
                    }
                    fprintf(fcsv, ",%d,%d,%d\n", cn[0], cn[1], cn[2]);
                    sl = s;
                }
            fclose(fcsv); fclose(fq); fclose(fdq);

            // whole subbands
            for (int c = 0; c < 3; c++)
                for (int l = 0; l <= depth; l++)
                    for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) {
                        char name[64];
                        snprintf(name, sizeof name, "band_l%d_%s_c%d.bin", l, oname[o], c);
                        wbin(dir, name, band[c][l][o], (size_t)g[c].bw[l] * g[c].bh[l] * 4);
                    }

            // inverse transform, level by level, whole frame
            uint16_t *pix[3];
            for (int c = 0; c < 3; c++) {
                int32_t *ll = band[c][0][0];
                pix[c] = calloc((size_t)g[c].w * g[c].h, 2);
                for (int l = 1; l <= depth; l++) {
                    int bw = g[c].bw[l], bh = g[c].bh[l], last = l == depth;
                    int32_t *out = last ? 0 : calloc((size_t)4 * bw * bh, sizeof(int32_t));
                    int32_t *tmp = vc2_alloc(vc2_strip_scratch_32(bw, bh) * sizeof(int32_t));
                    vc2_level_32 L = { wl, bw, bh, { ll, band[c][l][1], band[c][l][2], band[c][l][3] }, { -1, -1, -1, -1 }, bw,
                                       last ? VC2_OUT_PIX : VC2_OUT_I32, last ? (void *)pix[c] : (void *)out, -1,
                                       last ? g[c].w : 2 * bw, g[c].w, 0, g[c].h, 1 << (seq.bit_depth - 1), (1 << seq.bit_depth) - 1 };
                    vc2_synth_strip_32(&L, 0, bh, tmp);
                    free(tmp);
                    if (!last) {
                        for (size_t i = 0; i < (size_t)4 * bw * bh; i++) {
                            int64_t a = out[i] < 0 ? -(int64_t)out[i] : out[i];
                            if (a > maxsyn[c][l]) maxsyn[c][l] = a;
                        }
                        char name[64];
                        snprintf(name, sizeof name, "synth_l%d_c%d.bin", l, c);
                        wbin(dir, name, out, (size_t)4 * bw * bh * 4);
                    }
                    if (l > 1) free(ll);
                    ll = out;
                }
            }
            snprintf(path, sizeof path, "%s/picture.yuv", dir);
            FILE *fy = fopen(path, "wb");
            for (int c = 0; c < 3; c++) fwrite(pix[c], 2, (size_t)g[c].w * g[c].h, fy);
            fclose(fy);

            snprintf(path, sizeof path, "%s/ranges.txt", dir);
            FILE *fr = fopen(path, "w");
            fprintf(fr, "# largest magnitude: quantised / dequantised per subband; synthesised output per level (int32 reference)\n");
            fprintf(fr, "quantiser %d .. %d (factor %u .. %u)\n", qlo, qhi, vc2_qf[qlo < VC2_NQ ? qlo : 0], vc2_qf[qhi >= 0 ? qhi : 0]);
            for (int c = 0; c < 3; c++) {
                for (int l = 0; l <= depth; l++)
                    for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++)
                        fprintf(fr, "c%d L%d %s q %lld dq %lld\n", c, l, oname[o], (long long)maxq[c][l][o], (long long)maxdq[c][l][o]);
                for (int l = 1; l < depth; l++) fprintf(fr, "c%d synth L%d %lld\n", c, l, (long long)maxsyn[c][l]);
            }
            fclose(fr);

            for (int c = 0; c < 3; c++) {
                free(pix[c]);
                for (int l = 0; l <= depth; l++) {
                    free(xs[c][l]); free(ys[c][l]);
                    for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) free(band[c][l][o]);
                }
            }
            frame++;
        }
        pos += n;
        if (code == VC2_PC_END_SEQ) break;
    }
    fprintf(stderr, "%d frames\n", frame);
    return frame ? 0 : 1;
}
