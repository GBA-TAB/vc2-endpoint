// Last inverse-wavelet level of the VC-2 decoder as one OpenCL kernel: the steps of lib/vc2_lift.h on its
// 16-bit path (vertical lifting, horizontal lifting, (v + 1) >> 1, offset, clip), in wrapping 16-bit lanes.
// The Vulkan version of the same steps is lib/vc2_idwt.comp.
//
// One work-item = a tile of TW column pairs x RB row pairs of one component (of Cb and Cr for the CbCr plane
// of NV20). It walks down its rows with a rolling window: per row pair it loads one more row of the four
// bands, finishes the vertical lifting of that row pair and runs the horizontal lifting on the two rows.
// A lane vector holds the band columns k0-2 .. k0+LANES-3 (k0 = the tile's first column pair), which is what
// the horizontal lifting of the tile reaches; neighbouring tiles compute the overlap again. Nothing is
// stored but the output. Loads may start two samples before a row and end past it (the caller pads).
//
// Build options: -DWL=0 (Deslauriers-Dubuc 9,7) or 1 (LeGall 5,3); -DLANES=8 or 16; -DRB=row pairs per item;
// -DOUT=0 planar 16-bit, 1 NV20 Y plane, 2 NV20 CbCr plane; -DPIX_OFF, -DPIX_MAX.
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
#if OUT == 2
#define NC 2
#else
#define NC 1
#endif
#if OUT == 0
typedef ushort OT;
#else
typedef uchar OT;
#endif

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

// 8 samples of 10 bits -> 10 bytes (the NV15 / NV20 packing: sample 0 in the low bits)
static inline void pack8(ushort8 s, __global uchar *p)
{
    uint4 a = convert_uint4(s.s0246) | (convert_uint4(s.s1357) << 10);
    ulong2 q = convert_ulong2(a.s02) | (convert_ulong2(a.s13) << 20);
    ulong w = q.s0 | (q.s1 << 40);
    vstore4(as_ushort4(w), 0, (__global ushort *)p);
    ((__global ushort *)p)[4] = (ushort)(q.s1 >> 24);
}

__kernel void idwt(__global const short *cf, __global OT *out,
                   int4 off0, int4 off1, // LL, HL, LH, HH: where each band starts in cf, per component
                   int bw, int bh,       // band size; rows are bw apart
                   int oofs, int ostride, int oh) // output: start and row distance in OT units, rows to write
{
    const int k0 = (int)get_global_id(0) * TW, i0 = (int)get_global_id(1) * RB, i1 = min(i0 + RB, bh);
    if (k0 >= bw || i0 >= bh) return;
    const int jl = k0 == 0 ? 2 : 0, jr = min(bw + 1 - k0, LANES - 1); // the lanes of columns 0 and bw - 1
    const int edge = k0 == 0 || bw + 1 - k0 < LANES - 1;
    const UV em = TO_UV(clamp(IOTA, (short)jl, (short)jr));
    const int bo[2][4] = { { off0.s0, off0.s1, off0.s2, off0.s3 }, { off1.s0, off1.s1, off1.s2, off1.s3 } };
    V h0[NC][2], e0[NC][2];
#if LA == 2
    V h1[NC][2], e1[NC][2], em1[NC][2];
#endif
    for (int c = 0; c < NC; c++)
        for (int s = 0; s < 2; s++) {
            __global const short *l = cf + bo[c][s] + k0 - 2, *h = cf + bo[c][2 + s] + k0 - 2;
            h0[c][s] = VLOAD(h + i0 * bw);
            e0[c][s] = evrow(l, h, i0, bw, bh);
#if LA == 2
            h1[c][s] = VLOAD(h + min(i0 + 1, bh - 1) * bw);
            e1[c][s] = evrow(l, h, i0 + 1, bw, bh);
            em1[c][s] = evrow(l, h, i0 - 1, bw, bh);
#endif
        }
    for (int i = i0; i < i1; i++) {
        P r0[NC], r1[NC];
        for (int c = 0; c < NC; c++) {
            V ev[2], od[2];
            for (int s = 0; s < 2; s++) {
                __global const short *l = cf + bo[c][s] + k0 - 2, *h = cf + bo[c][2 + s] + k0 - 2;
                const int n = i + LA;
#if LA == 1
                V hn = h0[c][s], en = e0[c][s];
                if (n < bh) { hn = VLOAD(h + n * bw); en = UPD(VLOAD(l + n * bw), h0[c][s], hn); }
                ev[s] = e0[c][s];
                od[s] = pred(h0[c][s], en, e0[c][s], en, en);
                h0[c][s] = hn; e0[c][s] = en;
#else
                V hn = h1[c][s], en = e1[c][s];
                if (n < bh) { hn = VLOAD(h + n * bw); en = UPD(VLOAD(l + n * bw), h1[c][s], hn); }
                ev[s] = e0[c][s];
                od[s] = pred(h0[c][s], em1[c][s], e0[c][s], e1[c][s], en);
                h0[c][s] = h1[c][s]; h1[c][s] = hn;
                em1[c][s] = e0[c][s]; e0[c][s] = e1[c][s]; e1[c][s] = en;
#endif
            }
            r0[c] = hrow(ev[0], ev[1], edge, em);
            r1[c] = hrow(od[0], od[1], edge, em);
        }
        for (int t = 0; t < 2; t++) {
            const int y = 2 * i + t;
            if (y >= oh) break;
            const P a = t ? r1[0] : r0[0];
#if OUT == 0
            __global ushort *p = out + oofs + y * ostride + 2 * k0;
#if LANES == 8
            vstore8(a, 0, p);
#else
            vstore16(a, 0, p);
#endif
#elif OUT == 1
            __global uchar *p = out + oofs + y * ostride + k0 / 2 * 5;
#if LANES == 8
            pack8(a, p);
#else
            pack8(a.lo, p); pack8(a.hi, p + 10);
#endif
#else
            __global uchar *p = out + oofs + y * ostride + k0 * 5;
            const P b = t ? r1[1] : r0[1];
#if LANES == 8
            pack8(ZIP4(a.lo, b.lo), p); pack8(ZIP4(a.hi, b.hi), p + 10);
#else
            pack8(ZIP4(a.s0123, b.s0123), p); pack8(ZIP4(a.s4567, b.s4567), p + 10);
            pack8(ZIP4(a.s89ab, b.s89ab), p + 20); pack8(ZIP4(a.scdef, b.scdef), p + 30);
#endif
#endif
        }
    }
}
