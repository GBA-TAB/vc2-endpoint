// Last inverse-wavelet level of the VC-2 decoder as one OpenCL kernel: the steps of lib/vc2_lift.h on its
// 16-bit path (vertical lifting, horizontal lifting, (v + 1) >> 1, offset, clip), in wrapping 16-bit lanes.
// The Vulkan version of the same steps is lib/vc2_idwt.comp.
//
// One work-item = a tile of TW column pairs x RB row pairs of one component (of Cb and Cr for the CbCr plane
// of NV20); the tiles of all components lie side by side in the launch's x range, so that one launch does a
// band of rows of the whole picture (a launch costs about 0.1 ms on the Mali-G610, whatever its size).
// A work-item walks down its rows with a rolling window: per row pair it loads one more row of the four
// bands, finishes the vertical lifting of that row pair and runs the horizontal lifting on the two rows.
// A lane vector holds the band columns k0-2 .. k0+LANES-3 (k0 = the tile's first column pair), which is what
// the horizontal lifting of the tile reaches; neighbouring tiles compute the overlap again. Nothing is
// stored but the output. Loads may start two samples before a row and end past it (the caller pads).
//
// Build options: -DWL=0 (Deslauriers-Dubuc 9,7) or 1 (LeGall 5,3); -DLANES=8 or 16 (16 is slower on the
// G610: too many live registers); -DRB=row pairs per item; -DNV20=0 planar 16-bit output, 1 NV20 (Y plane and
// CbCr plane); -DPIX_OFF, -DPIX_MAX.
#if LANES == 8
typedef short8 V;
typedef ushort8 UV;
typedef short4 H;  // the tile's own columns
typedef ushort8 P; // a row of the tile's output samples, one component
#define TW 4
#define IOTA ((short8)(0, 1, 2, 3, 4, 5, 6, 7))
#define SHM1(v) ((v).s00123456)
#define SHP1(v) ((v).s12345677)
#define SHP2(v) ((v).s23456777)
#define MID(v) ((v).s2345)
#define ZIP(a, b) ((short8)((a).s0, (b).s0, (a).s1, (b).s1, (a).s2, (b).s2, (a).s3, (b).s3))
#define VLOAD(p) vload8(0, p)
#define TO_UV(v) convert_ushort8(v)
#define TO_P(v) as_ushort8(v)
#else
typedef short16 V;
typedef ushort16 UV;
typedef short8 H;
typedef ushort16 P;
#define TW 8
#define IOTA ((short16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
#define SHM1(v) ((v).s00123456789abcde)
#define SHP1(v) ((v).s123456789abcdeff)
#define SHP2(v) ((v).s23456789abcdefff)
#define MID(v) ((v).s23456789)
#define ZIP(a, b) ((short16)((a).s0, (b).s0, (a).s1, (b).s1, (a).s2, (b).s2, (a).s3, (b).s3, (a).s4, (b).s4, (a).s5, (b).s5, (a).s6, (b).s6, (a).s7, (b).s7))
#define VLOAD(p) vload16(0, p)
#define TO_UV(v) convert_ushort16(v)
#define TO_P(v) as_ushort16(v)
#endif
#define ZIP4(a, b) ((ushort8)((a).s0, (b).s0, (a).s1, (b).s1, (a).s2, (b).s2, (a).s3, (b).s3))
enum { PLANAR, NV20_Y, NV20_C };

// the lifting steps, as in the 16-bit lanes of vc2_lift.h
#define UPD(l, h0, h1) ((l) - (((h0) + (h1) + (short)2) >> (short)2)) // even -= (odd[i-1] + odd[i] + 2) >> 2
#if WL == 1
#define LA 1 // rows (columns) the second step looks ahead
static inline V pred(V h, V em, V e0, V e1, V e2) // odd += (even[i] + even[i+1] + 1) >> 1
{
    return h + ((e0 + e1 + (short)1) >> (short)1);
}
#else
#define LA 2
static inline V pred(V h, V em, V e0, V e1, V e2) // odd += (9 (e0 + e1) - (em + e2) + 8) >> 4
{
    V s1 = e0 + e1, s2 = em + e2;
    return h + ((s1 >> (short)1) + ((s1 - s2 + (short)8 + ((s1 & (short)1) << (short)3)) >> (short)4));
}
#endif

// even row i of the vertical lifting, with the row index clamped as the CPU does
static inline V evrow(__global const short *l, __global const short *h, int i, int bw, int bh)
{
    i = clamp(i, 0, bh - 1);
    return UPD(VLOAD(l + i * bw), VLOAD(h + max(i - 1, 0) * bw), VLOAD(h + i * bw));
}

// horizontal lifting of one row (lo = from LL | LH, hi = from HL | HH), then round, offset, clip
static inline P hrow(V lo, V hi, int edge, UV em)
{
    if (edge) hi = shuffle(hi, em);
    V e = UPD(lo, SHM1(hi), hi);
    if (edge) e = shuffle(e, em);
    V o = pred(hi, SHM1(e), e, SHP1(e), SHP2(e));
    H a = clamp(rhadd(MID(e), (H)0) + (short)PIX_OFF, (short)0, (short)PIX_MAX);
    H b = clamp(rhadd(MID(o), (H)0) + (short)PIX_OFF, (short)0, (short)PIX_MAX);
    return TO_P(ZIP(a, b));
}

// 8 samples of 10 bits -> 10 bytes (the NV15 / NV20 packing: sample 0 in the low bits), in 16-bit operations
static inline void pack8(ushort8 s, __global uchar *p)
{
#if PACK64
    uint4 a = convert_uint4(s.s0246) | (convert_uint4(s.s1357) << 10);
    ulong2 q = convert_ulong2(a.s02) | (convert_ulong2(a.s13) << 20);
    ulong w = q.s0 | (q.s1 << 40);
    vstore4(as_ushort4(w), 0, (__global ushort *)p);
    ((__global ushort *)p)[4] = (ushort)(q.s1 >> 24);
#else
    const ushort4 w = (ushort4)(s.s0 | (s.s1 << (ushort)10), (s.s1 >> (ushort)6) | (s.s2 << (ushort)4) | (s.s3 << (ushort)14),
                                (s.s3 >> (ushort)2) | (s.s4 << (ushort)8), (s.s4 >> (ushort)8) | (s.s5 << (ushort)2) | (s.s6 << (ushort)12));
    vstore4(w, 0, (__global ushort *)p);
    ((__global ushort *)p)[4] = (s.s6 >> (ushort)4) | (s.s7 << (ushort)6);
#endif
}

// one tile: nc components (2 = Cb and Cr for the CbCr plane: Cb first, its rows kept while Cr is computed; both at
// once would need more registers than the G610 has), band offsets off[LL, HL, LH, HH], first column pair k0, first row i0
static inline void tile(__global const short *cf, __global uchar *out, const int nc, const int mode, const int4 offa, const int4 offb,
                        const int bw, const int bh, const int k0, const int i0, const int oofs, const int ostride, const int oh)
{
    const int i1 = min(i0 + RB, bh);
    if (k0 >= bw || i0 >= bh) return;
    const int jl = k0 == 0 ? 2 : 0, jr = min(bw + 1 - k0, LANES - 1); // the lanes of columns 0 and bw - 1
    const int edge = k0 == 0 || bw + 1 - k0 < LANES - 1;
    const UV em = TO_UV(clamp(IOTA, (short)jl, (short)jr));
    P keep[2 * RB];
    for (int c = 0; c < nc; c++) {
        const int4 off = c ? offb : offa;
        __global const short *l[2] = { cf + off.s0 + k0 - 2, cf + off.s1 + k0 - 2 }, *h[2] = { cf + off.s2 + k0 - 2, cf + off.s3 + k0 - 2 };
        V h0[2], e0[2];
#if LA == 2
        V h1[2], e1[2], em1[2];
#endif
        for (int s = 0; s < 2; s++) {
            h0[s] = VLOAD(h[s] + i0 * bw);
            e0[s] = evrow(l[s], h[s], i0, bw, bh);
#if LA == 2
            h1[s] = VLOAD(h[s] + min(i0 + 1, bh - 1) * bw);
            e1[s] = evrow(l[s], h[s], i0 + 1, bw, bh);
            em1[s] = evrow(l[s], h[s], i0 - 1, bw, bh);
#endif
        }
        for (int i = i0; i < i1; i++) {
            V ev[2], od[2];
            for (int s = 0; s < 2; s++) {
                const int n = i + LA;
#if LA == 1
                V hn = h0[s], en = e0[s];
                if (n < bh) { hn = VLOAD(h[s] + n * bw); en = UPD(VLOAD(l[s] + n * bw), h0[s], hn); }
                ev[s] = e0[s];
                od[s] = pred(h0[s], en, e0[s], en, en);
                h0[s] = hn; e0[s] = en;
#else
                V hn = h1[s], en = e1[s];
                if (n < bh) { hn = VLOAD(h[s] + n * bw); en = UPD(VLOAD(l[s] + n * bw), h1[s], hn); }
                ev[s] = e0[s];
                od[s] = pred(h0[s], em1[s], e0[s], e1[s], en);
                h0[s] = h1[s]; h1[s] = hn;
                em1[s] = e0[s]; e0[s] = e1[s]; e1[s] = en;
#endif
            }
            for (int t = 0; t < 2; t++) {
                const int y = 2 * i + t;
                if (y >= oh) break;
                const P a = hrow(t ? od[0] : ev[0], t ? od[1] : ev[1], edge, em);
                if (mode == PLANAR) {
                    __global ushort *p = (__global ushort *)out + oofs + y * ostride + 2 * k0;
#if LANES == 8
                    vstore8(a, 0, p);
#else
                    vstore16(a, 0, p);
#endif
                } else if (mode == NV20_Y) {
                    __global uchar *p = out + oofs + y * ostride + k0 / 2 * 5;
#if LANES == 8
                    pack8(a, p);
#else
                    pack8(a.lo, p); pack8(a.hi, p + 10);
#endif
                } else if (c == 0) keep[y - 2 * i0] = a;
                else {
                    __global uchar *p = out + oofs + y * ostride + k0 * 5;
                    const P b = keep[y - 2 * i0];
#if LANES == 8
                    pack8(ZIP4(b.lo, a.lo), p); pack8(ZIP4(b.hi, a.hi), p + 10);
#else
                    pack8(ZIP4(b.s0123, a.s0123), p); pack8(ZIP4(b.s4567, a.s4567), p + 10);
                    pack8(ZIP4(b.s89ab, a.s89ab), p + 20); pack8(ZIP4(b.scdef, a.scdef), p + 30);
#endif
                }
            }
        }
    }
}

// x = tiles of Y, then of Cb, then of Cr (planar), or of Y, then of Cb and Cr together (NV20); y = blocks of RB band rows
__kernel void idwt(__global const short *cf, __global uchar *out,
                   int4 off0, int4 off1, int4 off2, // where LL, HL, LH, HH start in cf (in shorts), for Y, Cb, Cr
                   int4 bw, int bh,                 // band widths of Y, Cb, Cr (rows are that far apart); band height
                   int4 oofs, int4 ostride, int oh) // output per plane: start and row distance (planar: in samples,
                                                    // NV20: in bytes), rows to write
{
    const int gx = (int)get_global_id(0), i0 = (int)get_global_id(1) * RB;
    const int n0 = (bw.s0 + TW - 1) / TW, n1 = (bw.s1 + TW - 1) / TW;
#if NV20
    if (gx < n0) tile(cf, out, 1, NV20_Y, off0, off0, bw.s0, bh, gx * TW, i0, oofs.s0, ostride.s0, oh);
    else tile(cf, out, 2, NV20_C, off1, off2, bw.s1, bh, (gx - n0) * TW, i0, oofs.s1, ostride.s1, oh);
#else
    const int c = gx < n0 ? 0 : gx < n0 + n1 ? 1 : 2;
    const int4 o = c == 0 ? off0 : c == 1 ? off1 : off2;
    tile(cf, out, 1, PLANAR, o, o, c == 0 ? bw.s0 : c == 1 ? bw.s1 : bw.s2, bh, (gx - (c == 0 ? 0 : c == 1 ? n0 : n0 + n1)) * TW, i0,
         c == 0 ? oofs.s0 : c == 1 ? oofs.s1 : oofs.s2, c == 0 ? ostride.s0 : c == 1 ? ostride.s1 : ostride.s2, oh);
#endif
}
