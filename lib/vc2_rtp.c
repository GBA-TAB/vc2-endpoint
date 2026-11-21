// RTP payload for VC-2 HQ (RFC 8450): packetiser, and a receiver that feeds the cascade decoder as packets
// arrive, with a reorder window and concealment of lost slices. Layout: RFC 8450 section 4 (figures 1-3),
// header fields 4.1 / 4.2, stream constraints 4.4.
#include "vc2_priv.h"
#include <stdio.h>

static void wr16(uint8_t *p, unsigned v) { p[0] = v >> 8; p[1] = v; }
static void wr32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static unsigned rd16(const uint8_t *p) { return p[0] << 8 | p[1]; }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }

// size of the slice at p (prefix bytes, quantiser index, three length-prefixed components); 0 if it runs past end
static size_t slice_len(const uint8_t *p, const uint8_t *end, int prefix, int scaler)
{
    const uint8_t *s = p + prefix + 1;
    for (int c = 0; c < 3; c++) {
        if (s >= end) return 0;
        s += 1 + (size_t)*s * scaler;
    }
    return s <= end ? (size_t)(s - p) : 0;
}

// --- sender ---------------------------------------------------------------------------------------------
struct vc2_rtp_tx { int pt, mtu; uint32_t ssrc, seq; uint8_t *pkt; };

vc2_rtp_tx *vc2_rtp_tx_new(int payload_type, uint32_t ssrc, int mtu)
{
    vc2_rtp_tx *t = calloc(1, sizeof *t);
    t->pt = payload_type & 127; t->ssrc = ssrc; t->mtu = mtu < 128 ? 128 : mtu > 65000 ? 65000 : mtu;
    t->pkt = malloc(t->mtu + 16);
    return t;
}
void vc2_rtp_tx_free(vc2_rtp_tx *t) { if (t) { free(t->pkt); free(t); } }

// RTP header (12 bytes) + extended sequence number, flags, parse code (4 bytes)
static size_t tx_header(vc2_rtp_tx *t, uint32_t ts, int marker, int pc)
{
    uint8_t *p = t->pkt;
    p[0] = 0x80; p[1] = (marker ? 0x80 : 0) | t->pt;
    wr16(p + 2, t->seq & 0xffff); wr32(p + 4, ts); wr32(p + 8, t->ssrc);
    wr16(p + 12, t->seq >> 16); p[14] = 0; p[15] = pc;
    t->seq++;
    return 16;
}

int vc2_rtp_tx_frame(vc2_rtp_tx *t, const uint8_t *au, size_t len, uint32_t ts, vc2_rtp_send_fn send, void *ctx)
{
    vc2_seq seq;
    vc2_pic pic;
    size_t pos = 0, n;
    int code, have_seq = 0, packets = 0;
    while ((n = vc2_unit(au + pos, len - pos, &code)) && code != VC2_PC_PICTURE_HQ) {
        if (code == VC2_PC_SEQ_HEADER && !vc2_parse_seq(au + pos + 13, n - 13, &seq)) have_seq = 1;
        pos += n;
    }
    if (!n || !have_seq || vc2_parse_pic(au + pos + 13, n - 13, &seq, &pic)) return -1;
    const uint8_t *sl = au + pos + 13 + pic.slices_offset, *end = au + pos + n;
    // sequence header, as version 3 (fragments are a version 3 feature: RFC 8450 section 7.1)
    seq.major = 3;
    size_t h = tx_header(t, ts, 0, VC2_PC_SEQ_HEADER);
    h += vc2_write_seq(t->pkt + h, &seq);
    send(ctx, t->pkt, h); packets++;
    // transform parameters
    h = tx_header(t, ts, 0, 0xEC);
    wr32(t->pkt + h, pic.picture_number); wr16(t->pkt + h + 4, pic.prefix_bytes); wr16(t->pkt + h + 6, pic.size_scaler);
    size_t tp = vc2_write_tp(t->pkt + h + 12, &pic, 3);
    wr16(t->pkt + h + 8, tp); wr16(t->pkt + h + 10, 0);
    send(ctx, t->pkt, h + 12 + tp); packets++;
    // slices, as many whole ones per packet as fit
    const int total = pic.slices_x * pic.slices_y, room = t->mtu - VC2_RTP_OVERHEAD;
    for (int i = 0; i < total;) {
        size_t used = 0;
        int cnt = 0;
        const uint8_t *first = sl;
        while (i + cnt < total) {
            size_t l = slice_len(sl, end, pic.prefix_bytes, pic.size_scaler);
            if (!l) return -1;
            if (used + l > (size_t)room) break;
            used += l; sl += l; cnt++;
        }
        if (!cnt) return -1; // one slice is larger than a packet: encode with max_slice_bytes
        h = tx_header(t, ts, i + cnt == total, 0xEC);
        uint8_t *p = t->pkt + h;
        wr32(p, pic.picture_number); wr16(p + 4, pic.prefix_bytes); wr16(p + 6, pic.size_scaler);
        wr16(p + 8, used); wr16(p + 10, cnt); wr16(p + 12, i % pic.slices_x); wr16(p + 14, i / pic.slices_x);
        memcpy(p + 16, first, used);
        send(ctx, t->pkt, h + 16 + used); packets++;
        i += cnt;
    }
    return packets;
}

int vc2_rtp_sdp(char *out, size_t cap, const char *dst_ip, int port, int payload_type, const vc2_seq *s, double bpp, const char *ts_refclk)
{
    // m=, rtpmap and fmtp profile/version/level: RFC 8450 section 7.2. The rest follows the ST 2110-22 /
    // -10 / -21 conventions as used for our JPEG XS streams (bit rate in b=AS, frame rate, sender type TP,
    // reference and media clock); the video description is given as in ST 2110-20 for receivers that want it.
    const char *samp = s->chroma == 0 ? "YCbCr-4:4:4" : s->chroma == 1 ? "YCbCr-4:2:2" : "YCbCr-4:2:0";
    double fps = s->fr_den ? (double)s->fr_num / s->fr_den : 0;
    char rate[32];
    if (s->fr_den == 1) snprintf(rate, sizeof rate, "%d", s->fr_num); else snprintf(rate, sizeof rate, "%d/%d", s->fr_num, s->fr_den);
    return snprintf(out, cap,
                    "m=video %d RTP/AVP %d\r\nc=IN IP4 %s\r\nb=AS:%.0f\r\na=rtpmap:%d vc2/90000\r\n"
                    "a=fmtp:%d profile=HQ;version=3;level=0;sampling=%s;depth=%d;width=%d;height=%d;exactframerate=%s;"
                    "colorimetry=BT709;TCS=SDR;PM=2110GPM;SSN=ST2110-22:2019;TP=2110TPN\r\n"
                    "a=framerate:%.6g\r\na=mediaclk:direct=0\r\n%s%s%s",
                    port, payload_type, dst_ip, s->width * (double)s->height * bpp * fps / 1000, payload_type, payload_type, samp,
                    s->bit_depth, s->width, s->height, rate, fps, ts_refclk ? "a=ts-refclk:" : "", ts_refclk ? ts_refclk : "", ts_refclk ? "\r\n" : "");
}

// --- receiver -------------------------------------------------------------------------------------------
#define RX_WIN 64
#define RX_PKT 9216
typedef struct { uint8_t *buf; size_t cap, len, hdr; size_t *off; int nslices, nx, prefix, scaler, valid; } rx_pic;
struct vc2_rtp_rx {
    vc2_dec *dec;
    int reorder, started;
    uint32_t expected;              // next sequence number (32 bits)
    uint8_t *slot[RX_WIN];
    uint16_t slot_len[RX_WIN];
    uint8_t slot_used[RX_WIN];
    int held;
    uint8_t seqhdr[256], tp[256];
    size_t seqhdr_len, tp_len;
    vc2_seq seq;
    rx_pic cur, prev;
    int open, next_slice, lines;
    uint32_t ts, picnum;
    vc2_frame frame;
    vc2_rtp_stats st;
};

vc2_rtp_rx *vc2_rtp_rx_new(vc2_dec *d, int reorder)
{
    vc2_rtp_rx *r = calloc(1, sizeof *r);
    r->dec = d; r->reorder = reorder < 0 ? 0 : reorder >= RX_WIN ? RX_WIN - 1 : reorder;
    for (int i = 0; i < RX_WIN; i++) r->slot[i] = malloc(RX_PKT);
    return r;
}
void vc2_rtp_rx_free(vc2_rtp_rx *r)
{
    if (!r) return;
    for (int i = 0; i < RX_WIN; i++) free(r->slot[i]);
    free(r->cur.buf); free(r->cur.off); free(r->prev.buf); free(r->prev.off);
    free(r);
}
const vc2_rtp_stats *vc2_rtp_rx_stats(const vc2_rtp_rx *r) { return &r->st; }
int vc2_rtp_rx_lines(const vc2_rtp_rx *r) { return r->lines; }

// start a picture in r->cur from the stored sequence header and transform parameters
static int pic_open(vc2_rtp_rx *r, uint32_t ts, int prefix, int scaler)
{
    vc2_pic pic;
    rx_pic *c = &r->cur;
    uint8_t hdr[600];
    if (!r->seqhdr_len || !r->tp_len) return -1;
    size_t n = vc2_write_parse_info(hdr, VC2_PC_SEQ_HEADER, 13 + r->seqhdr_len, 0);
    memcpy(hdr + n, r->seqhdr, r->seqhdr_len); n += r->seqhdr_len;
    vc2_write_parse_info(hdr + n, VC2_PC_PICTURE_HQ, 0, 13 + r->seqhdr_len);
    wr32(hdr + n + 13, r->picnum);
    memcpy(hdr + n + 17, r->tp, r->tp_len);
    if (vc2_parse_seq(r->seqhdr, r->seqhdr_len, &r->seq) || vc2_parse_pic(hdr + n + 13, 4 + r->tp_len, &r->seq, &pic)) return -1;
    if (pic.prefix_bytes != prefix || pic.size_scaler != scaler) {
        // transform parameters kept from an earlier picture: the two values that change per picture are in every packet
        pic.prefix_bytes = prefix; pic.size_scaler = scaler;
        r->tp_len = vc2_write_tp(r->tp, &pic, r->seq.major);
        memcpy(hdr + n + 17, r->tp, r->tp_len);
    }
    size_t need = n + 17 + r->tp_len + (size_t)r->seq.width * r->seq.height * 6 + 65536;
    if (c->cap < need) { free(c->buf); c->buf = malloc(need + VC2_INPUT_PADDING); c->cap = need; }
    int ns = pic.slices_x * pic.slices_y;
    if (c->nslices != ns) { free(c->off); c->off = malloc((ns + 1) * sizeof *c->off); }
    memcpy(c->buf, hdr, n + 17 + r->tp_len);
    c->hdr = c->len = n + 17 + r->tp_len;
    c->nslices = ns; c->nx = pic.slices_x; c->prefix = prefix; c->scaler = scaler; c->valid = 0;
    c->off[0] = c->len;
    if (vc2_dec_begin(r->dec, c->buf, c->len, &r->frame)) return -1;
    r->open = 1; r->next_slice = 0; r->ts = ts; r->lines = 0;
    return 0;
}

// fill slices [next_slice, upto) from the previous frame, or with empty slices if there is none that fits
static void conceal(vc2_rtp_rx *r, int upto)
{
    rx_pic *c = &r->cur, *p = &r->prev;
    for (int i = r->next_slice; i < upto; i++) {
        uint8_t *d = c->buf + c->len;
        int done = 0;
        if (p->valid && p->nslices == c->nslices && p->nx == c->nx && p->prefix == c->prefix) {
            const uint8_t *s = p->buf + p->off[i];
            memcpy(d, s, c->prefix + 1);
            uint8_t *o = d + c->prefix + 1;
            s += c->prefix + 1;
            done = 1;
            for (int k = 0; k < 3 && done; k++) { // lengths are in units of the scaler, which may have changed
                size_t l = (size_t)*s * p->scaler, units = (l + c->scaler - 1) / c->scaler;
                if (units > 255) { done = 0; break; }
                *o++ = (uint8_t)units;
                memcpy(o, s + 1, l);
                memset(o + l, 0xff, units * c->scaler - l);
                o += units * c->scaler; s += 1 + l;
            }
            if (done) c->len = o - c->buf;
        }
        if (!done) { memset(d, 0, c->prefix + 4); c->len += c->prefix + 4; }
        c->off[i + 1] = c->len;
        r->st.concealed_slices++;
    }
    if (upto > r->next_slice) r->next_slice = upto;
}

static int pic_close(vc2_rtp_rx *r, vc2_frame *out)
{
    if (!r->open) return 0;
    conceal(r, r->cur.nslices);
    r->lines = vc2_dec_feed(r->dec, r->cur.len);
    r->open = 0;
    r->cur.valid = 1;
    rx_pic t = r->prev; r->prev = r->cur; r->cur = t;
    r->st.frames++;
    if (out) *out = r->frame;
    return 1;
}

// one packet, in sequence order
static int deliver(vc2_rtp_rx *r, const uint8_t *p, size_t len, vc2_frame *out)
{
    int got = 0;
    if (len < 16 || (p[0] >> 6) != 2) return 0;
    size_t h = 12 + 4 * (p[0] & 15);
    if (p[0] & 0x10) { if (len < h + 4) return 0; h += 4 + 4 * rd16(p + h + 2); } // header extension
    if (len < h + 4) return 0;
    uint32_t ts = rd32(p + 4);
    int marker = p[1] >> 7, pc = p[h + 3];
    const uint8_t *q = p + h + 4;
    size_t n = len - h - 4;
    if (r->open && ts != r->ts) got = pic_close(r, out); // the end of the last picture was lost
    if (pc == VC2_PC_SEQ_HEADER) {
        if (n <= sizeof r->seqhdr) { memcpy(r->seqhdr, q, n); r->seqhdr_len = n; }
    } else if (pc == 0xEC && n >= 12) {
        int prefix = rd16(q + 4), scaler = rd16(q + 6), nsl = rd16(q + 10);
        size_t flen = rd16(q + 8);
        if (nsl == 0) {
            if (n >= 12 + flen && flen <= sizeof r->tp && !r->open) { memcpy(r->tp, q + 12, flen); r->tp_len = flen; r->picnum = rd32(q); pic_open(r, ts, prefix, scaler); }
        } else if (n >= 16 + flen) {
            // without this picture's transform parameters, the last ones are taken to still hold (RFC 8450 4.2)
            if (!r->open) { r->picnum = rd32(q); if (pic_open(r, ts, prefix, scaler)) { r->st.dropped_frames++; return got; } }
            rx_pic *c = &r->cur;
            int idx = rd16(q + 14) * c->nx + rd16(q + 12);
            if (idx >= r->next_slice && idx + nsl <= c->nslices && scaler == c->scaler) {
                conceal(r, idx);
                const uint8_t *s = q + 16, *end = s + flen;
                for (int i = 0; i < nsl; i++) {
                    size_t l = slice_len(s, end, c->prefix, c->scaler);
                    if (!l) break;
                    memcpy(c->buf + c->len, s, l);
                    c->len += l; s += l;
                    c->off[++r->next_slice] = c->len;
                }
                r->lines = vc2_dec_feed(r->dec, c->len);
            }
            if (marker || r->next_slice == c->nslices) got |= pic_close(r, out);
        }
    }
    return got;
}

static int drain(vc2_rtp_rx *r, int force, vc2_frame *out)
{
    int got = 0;
    while (r->held) {
        int i = r->expected % RX_WIN;
        if (!r->slot_used[i]) {
            if (!force) break;
            r->st.lost++; r->expected++;
            continue;
        }
        got |= deliver(r, r->slot[i], r->slot_len[i], out);
        r->slot_used[i] = 0; r->held--; r->expected++;
    }
    return got;
}

int vc2_rtp_rx_packet(vc2_rtp_rx *r, const uint8_t *pkt, size_t len, vc2_frame *out)
{
    if (len < 16 || len > RX_PKT) return 0;
    size_t h = 12 + 4 * (pkt[0] & 15);
    if (pkt[0] & 0x10) { if (len < h + 4) return 0; h += 4 + 4 * rd16(pkt + h + 2); }
    if (len < h + 4) return 0;
    uint32_t seq = (uint32_t)rd16(pkt + h) << 16 | rd16(pkt + 2);
    int got = 0;
    r->st.packets++;
    if (!r->started) { r->started = 1; r->expected = seq; }
    int32_t d = (int32_t)(seq - r->expected);
    if (d < 0) { r->st.late++; return 0; } // late or duplicate (also the second leg of a ST 2022-7 pair)
    if (d == 0) {
        got = deliver(r, pkt, len, out);
        r->expected++;
        return got | drain(r, 0, out);
    }
    // ahead of the expected packet: wait for the missing ones while it fits the window, then give them up
    while ((d = (int32_t)(seq - r->expected)) > r->reorder) {
        if (d > 4 * RX_WIN) { // a long break: resynchronise
            got |= drain(r, 1, out);
            r->st.lost += seq - r->expected; r->expected = seq;
            break;
        }
        if (!r->slot_used[r->expected % RX_WIN]) { r->st.lost++; r->expected++; }
        got |= drain(r, 0, out);
    }
    if (d == 0) {
        got |= deliver(r, pkt, len, out);
        r->expected++;
        return got | drain(r, 0, out);
    }
    int i = seq % RX_WIN;
    if (!r->slot_used[i]) { memcpy(r->slot[i], pkt, len); r->slot_len[i] = (uint16_t)len; r->slot_used[i] = 1; r->held++; }
    return got;
}

int vc2_rtp_rx_flush(vc2_rtp_rx *r, vc2_frame *out)
{
    int got = drain(r, 1, out);
    return got | pic_close(r, out);
}
