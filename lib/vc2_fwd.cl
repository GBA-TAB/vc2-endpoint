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

// The rate control's counting for the 16-bit bands (lib/vc2_enc.c count_row_w, the same float arithmetic:
// OpenCL rounds a single multiply correctly, and nothing here can be contracted): per band row and slice, the
// bits of the coefficients at the band's quantiser and the truncation cost. desc[i] = { offset in shorts, band
// width, band rows, first output index }; one work-item per (band row, slice), dimension 1 = the band.
__kernel void count16(__global const short *bands, __global const int4 *desc, __global const float *r16, int nx,
                      __global uint2 *out)
{
    const int4 d = desc[get_global_id(1)];
    const int seg = get_global_id(0), y = seg / nx, sx = seg - y * nx;
    if (y >= d.z) return;
    const float r = r16[get_global_id(1)];
    const int x0 = (int)((long)d.y * sx / nx), x1 = (int)((long)d.y * (sx + 1) / nx);
    __global const short *c = bands + d.x + (size_t)y * d.y;
    uint b = 0, dd = 0;
    int x = x0;
    if (!(x0 & 7))
        for (; x + 8 <= x1; x += 8) { // eight at a time, the same arithmetic per lane
            int8 n = convert_int8(abs(vload8(0, c + x))) << 2;
            int8 u = convert_int8_rtz(convert_float8(n) * r), v = u >> 4;
            uint8 ex = (as_uint8(convert_float8(v + 1)) >> 23) - 127;
            int8 vv = min(v, 255), uu = min(u, 4096);
            uint8 bb = 2 * ex + 1 + as_uint8(-(v != 0));
            uint8 d8 = select((uint8)0, as_uint8((2 * vv + 1) * (uu - 8 * vv - 4)), v != 0);
            b += bb.s0 + bb.s1 + bb.s2 + bb.s3 + bb.s4 + bb.s5 + bb.s6 + bb.s7;
            dd += d8.s0 + d8.s1 + d8.s2 + d8.s3 + d8.s4 + d8.s5 + d8.s6 + d8.s7;
        }
    for (; x < x1; x++) {
        int n = abs((int)c[x]) << 2;
        int u = convert_int_rtz((float)n * r), v = u >> 4;
        uint ex = (as_uint((float)(v + 1)) >> 23) - 127;
        int vv = min(v, 255), uu = min(u, 4096);
        b += 2 * ex + 1 + (v != 0);
        dd += v ? (uint)((2 * vv + 1) * (uu - 8 * vv - 4)) : 0;
    }
    out[d.w + seg] = (uint2)(b, dd);
}
