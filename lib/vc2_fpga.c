// The host side of a hardware VC-2 decoder: header parsing, the subset check, the H2C descriptor (docs/FPGA.md
// section 9), and a card model that rebuilds a stream from a descriptor to prove nothing is missing.
#include "vc2_fpga.h"
#include "vc2_priv.h"
#include <string.h>

static void w16(uint8_t *p, unsigned v) { p[0] = v; p[1] = v >> 8; }
static void w32(uint8_t *p, uint32_t v) { w16(p, v & 0xffff); w16(p + 2, v >> 16); }
static unsigned r16(const uint8_t *p) { return p[0] | p[1] << 8; }
static uint32_t r32(const uint8_t *p) { return r16(p) | (uint32_t)r16(p + 2) << 16; }

void vc2_fpga_desc_write(uint8_t *o, const vc2_fpga_desc *d)
{
    memset(o, 0, VC2_FPGA_DESC_LEN);
    memcpy(o, "VC2D", 4);
    o[4] = 0;                       // version
    o[5] = VC2_FPGA_DESC_LEN / 16;  // descriptor length in 16-byte units
    w16(o + 6, d->flags);
    w32(o + 8, d->picture_number);
    w32(o + 12, d->slice_bytes);
    w16(o + 16, d->width); w16(o + 18, d->height);
    o[20] = d->chroma; o[21] = d->bit_depth; o[22] = d->wavelet; o[23] = d->depth;
    w16(o + 24, d->slices_x); w16(o + 26, d->slices_y);
    w16(o + 28, d->prefix_bytes); w16(o + 30, d->size_scaler);
    memcpy(o + 32, d->qm, 16);
    w32(o + 48, d->seq_count); w32(o + 52, d->au_index);
}

int vc2_fpga_desc_read(const uint8_t *o, size_t len, vc2_fpga_desc *d)
{
    if (len < VC2_FPGA_DESC_LEN || memcmp(o, "VC2D", 4) || o[4] != 0 || o[5] != VC2_FPGA_DESC_LEN / 16) return -1;
    memset(d, 0, sizeof *d);
    d->flags = r16(o + 6);
    d->picture_number = r32(o + 8); d->slice_bytes = r32(o + 12);
    d->width = r16(o + 16); d->height = r16(o + 18);
    d->chroma = o[20]; d->bit_depth = o[21]; d->wavelet = o[22]; d->depth = o[23];
    d->slices_x = r16(o + 24); d->slices_y = r16(o + 26);
    d->prefix_bytes = r16(o + 28); d->size_scaler = r16(o + 30);
    memcpy(d->qm, o + 32, 16);
    d->seq_count = r32(o + 48); d->au_index = r32(o + 52);
    return 0;
}

const char *vc2_fpga_subset(const vc2_seq *s, const vc2_pic *p)
{
    if (s->major != 2 && s->major != 3) return "stream version not 2 or 3";
    if (s->picture_coding_mode) return "fields, not frames";
    if (s->chroma != 1) return "not 4:2:2";
    if (s->bit_depth != 10) return "not 10 bit";
    if (!((s->width == 1920 && s->height == 1080) || (s->width == 3840 && s->height == 2160))) return "not 1920x1080 or 3840x2160";
    if (p->wavelet != VC2_WL_LEGALL5_3) return "wavelet not LeGall 5,3";
    if (p->depth != 4) return "depth not 4";
    for (int c = 0; c < 2; c++) { // luma, chroma
        vc2_geom g;
        vc2_geom_init(&g, c ? s->width / 2 : s->width, s->height, p->depth);
        for (int l = 0; l <= p->depth; l++)
            if (g.bw[l] % p->slices_x || g.bh[l] % p->slices_y) return "slices not uniform";
    }
    if (p->size_scaler > 0xffff || p->prefix_bytes > 0xffff) return "size_scaler or prefix_bytes over 16 bits";
    return 0;
}

static int fixed_changed(const vc2_pic *a, const vc2_pic *b)
{
    return a->wavelet != b->wavelet || a->depth != b->depth || a->slices_x != b->slices_x ||
           a->slices_y != b->slices_y || a->prefix_bytes != b->prefix_bytes;
}

size_t vc2_fpga_pack(vc2_fpga_tx *t, const uint8_t *au, size_t len, uint8_t *out, size_t cap, const char **why)
{
    const char *dummy;
    if (!why) why = &dummy;
    *why = "malformed";
    size_t pos = 0;
    while (pos < len) {
        int code;
        size_t n = vc2_unit(au + pos, len - pos, &code);
        if (!n) return 0;
        const uint8_t *pl = au + pos + 13;
        size_t plen = n - 13;
        if (code == VC2_PC_SEQ_HEADER) {
            vc2_seq s;
            if (vc2_parse_seq(pl, plen, &s)) return 0;
            // a repeated, identical sequence header is no change (ffmpeg repeats it before every picture)
            if (!t->have_seq || memcmp(&s, &t->seq, sizeof s)) { t->seq = s; t->have_seq = 1; t->new_seq = 1; }
        } else if (code == VC2_PC_PICTURE_HQ) {
            vc2_pic p;
            if (!t->have_seq || vc2_parse_pic(pl, plen, &t->seq, &p)) return 0;
            if ((*why = vc2_fpga_subset(&t->seq, &p))) return 0;
            // the slice lengths must stay inside the picture: the card would read past it into the next descriptor
            // (ERR_OVERRUN, then a reset). A picture damaged in transit is kept on the CPU decoder instead.
            {
                const uint8_t *sp = pl + p.slices_offset, *end = pl + plen;
                for (long k = (long)p.slices_x * p.slices_y; k > 0; k--) {
                    if ((size_t)(end - sp) < (size_t)p.prefix_bytes + 4) { *why = "slices run past the picture"; return 0; }
                    sp += p.prefix_bytes + 1;
                    for (int c = 0; c < 3; c++) {
                        size_t l = (size_t)*sp++ * p.size_scaler;
                        if ((size_t)(end - sp) < l + (c < 2)) { *why = "slices run past the picture"; return 0; }
                        sp += l;
                    }
                }
            }
            size_t sb = plen - p.slices_offset, total = (VC2_FPGA_DESC_LEN + sb + VC2_FPGA_ALIGN - 1) & ~(size_t)(VC2_FPGA_ALIGN - 1);
            if (total > cap || sb > 0xffffffffu) { *why = "output buffer too small"; return 0; }
            if (t->have_pic && fixed_changed(&p, &t->last)) t->new_seq = 1;
            vc2_fpga_desc d = { .picture_number = p.picture_number, .slice_bytes = (uint32_t)sb,
                                .seq_count = t->seq_count + t->new_seq, .au_index = t->au_index++,
                                .flags = t->new_seq ? VC2_FPGA_NEW_SEQ : 0, .width = t->seq.width, .height = t->seq.height,
                                .chroma = t->seq.chroma, .bit_depth = t->seq.bit_depth, .wavelet = p.wavelet, .depth = p.depth,
                                .slices_x = p.slices_x, .slices_y = p.slices_y, .prefix_bytes = p.prefix_bytes,
                                .size_scaler = p.size_scaler };
            int k = 0;
            for (int l = 0; l <= p.depth; l++)
                for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) d.qm[k++] = p.qm[l][o];
            vc2_fpga_desc_write(out, &d);
            memcpy(out + VC2_FPGA_DESC_LEN, pl + p.slices_offset, sb);
            memset(out + VC2_FPGA_DESC_LEN + sb, 0, total - VC2_FPGA_DESC_LEN - sb);
            t->seq_count = d.seq_count; t->new_seq = 0; t->last = p; t->have_pic = 1;
            *why = 0;
            return total;
        } // auxiliary, padding, end of sequence: nothing for the card
        pos += n;
    }
    *why = "no picture";
    return 0;
}

size_t vc2_fpga_model_au(const uint8_t *in, size_t len, uint8_t *au, size_t cap)
{
    vc2_fpga_desc d;
    if (vc2_fpga_desc_read(in, len, &d) || len < VC2_FPGA_DESC_LEN + d.slice_bytes || cap < d.slice_bytes + 256 ||
        d.depth < 1 || d.depth > 5) return 0;
    vc2_seq s = { .major = 2, .profile = 3, .width = d.width, .height = d.height, .chroma = d.chroma,
                  .fr_num = 50, .fr_den = 1, .bit_depth = d.bit_depth }; // the frame rate doesn't reach the card
    size_t n = 13;
    n += vc2_write_seq(au + n, &s);
    vc2_write_parse_info(au, VC2_PC_SEQ_HEADER, n, 0);
    size_t ps = n;
    n += 13;
    au[n] = d.picture_number >> 24; au[n + 1] = d.picture_number >> 16; au[n + 2] = d.picture_number >> 8; au[n + 3] = d.picture_number;
    n += 4;
    vc2_pic p = { .picture_number = d.picture_number, .wavelet = d.wavelet, .depth = d.depth, .slices_x = d.slices_x,
                  .slices_y = d.slices_y, .prefix_bytes = d.prefix_bytes, .size_scaler = d.size_scaler };
    int k = 0;
    for (int l = 0; l <= d.depth; l++)
        for (int o = l ? 1 : 0; o < (l ? 4 : 1); o++) p.qm[l][o] = d.qm[k++];
    n += vc2_write_tp(au + n, &p, 2);
    memcpy(au + n, in + VC2_FPGA_DESC_LEN, d.slice_bytes);
    n += d.slice_bytes;
    vc2_write_parse_info(au + ps, VC2_PC_PICTURE_HQ, (uint32_t)(n - ps), (uint32_t)ps);
    return n;
}
