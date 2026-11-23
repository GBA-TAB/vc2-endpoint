// Encoder front end on the GPU (lib/vc2_enc_cl.c): the two 16-bit forward levels, bit-exact with the CPU.
// The encoder's forward LeGall 5,3 level in 16-bit lanes (lib/vc2_enc.c analyse_level16): per input row the
// horizontal lifting (inputs << 1, hi -= (lo + lo_right + 1) >> 1, lo += (hi_left + hi + 2) >> 2, edges
// replicated), then the vertical steps the same way between even rows (type 0) and odd rows (type 1), edges
// clamped. One work-item = 8 band columns x RB band rows; it recomputes one row pair above its strip.
typedef struct { short8 lo, hi; } R;

static R hrow(__global const short *in, int istride, int w, int k0, int y)
{
    __global const short *r = in + (size_t)y * istride;
    short16 v = vload16(0, r + 2 * k0);
    short8 lo = v.even << (short8)1, hi = v.odd << (short8)1;
    int km = k0 > 0 ? k0 - 1 : 0, kp = k0 + 8 < w ? k0 + 8 : w - 1;
    short lom = (short)(r[2 * km] << 1), him = (short)(r[2 * km + 1] << 1), lop = (short)(r[2 * kp] << 1);
    short8 lon = (short8)(lo.s1234, lo.s567, lop);
    short8 hp = hi - ((lo + lon + (short8)1) >> (short8)1);
    short hpm = k0 > 0 ? (short)(him - ((lom + lo.s0 + 1) >> 1)) : hp.s0;
    short8 hprev = (short8)(hpm, hp.s012, hp.s3456);
    R o = { lo + ((hprev + hp + (short8)2) >> (short8)2), hp };
    return o;
}

__kernel void fwd53(__global const short *in, int istride, int w, int h, int rb,
                    __global short *ll, __global short *hl, __global short *lh, __global short *hh)
{
    const int k0 = get_global_id(0) * 8, r0 = get_global_id(1) * rb;
    if (k0 >= w || r0 >= h) return;
    const int r1 = r0 + rb < h ? r0 + rb : h;
    int s = r0 > 0 ? r0 - 1 : 0;
    R c0 = hrow(in, istride, w, k0, 2 * s), c1 = hrow(in, istride, w, k0, 2 * s + 1);
    R n0 = hrow(in, istride, w, k0, 2 * (s + 1 < h ? s + 1 : h - 1));
    R x1 = { c1.lo - ((c0.lo + n0.lo + (short8)1) >> (short8)1), c1.hi - ((c0.hi + n0.hi + (short8)1) >> (short8)1) };
    R p1 = x1; // X1'[i - 1], clamped at the top
    if (s < r0) {
        c0 = n0; c1 = hrow(in, istride, w, k0, 2 * r0 + 1);
        n0 = hrow(in, istride, w, k0, 2 * (r0 + 1 < h ? r0 + 1 : h - 1));
        x1.lo = c1.lo - ((c0.lo + n0.lo + (short8)1) >> (short8)1);
        x1.hi = c1.hi - ((c0.hi + n0.hi + (short8)1) >> (short8)1);
    }
    for (int i = r0; i < r1; i++) {
        short8 xl = c0.lo + ((p1.lo + x1.lo + (short8)2) >> (short8)2), xh = c0.hi + ((p1.hi + x1.hi + (short8)2) >> (short8)2);
        size_t o = (size_t)i * w + k0;
        vstore8(xl, 0, ll + o); vstore8(xh, 0, hl + o); vstore8(x1.lo, 0, lh + o); vstore8(x1.hi, 0, hh + o);
        if (i + 1 < r1) {
            p1 = x1; c0 = n0; c1 = hrow(in, istride, w, k0, 2 * (i + 1) + 1);
            n0 = hrow(in, istride, w, k0, 2 * (i + 2 < h ? i + 2 : h - 1));
            x1.lo = c1.lo - ((c0.lo + n0.lo + (short8)1) >> (short8)1);
            x1.hi = c1.hi - ((c0.hi + n0.hi + (short8)1) >> (short8)1);
        }
    }
}
