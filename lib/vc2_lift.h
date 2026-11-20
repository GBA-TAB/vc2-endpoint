// Lifting kernels and the strip-wise inverse wavelet, as a template over the coefficient type.
// Include with T (int32_t or int16_t) and SUF defined. The 16-bit instance computes in wrapping 16-bit
// lanes (GCC vector extensions: NEON on arm64, SSE/AVX on x86), which is exact while the sums fit 16 bits.
#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define FN(n) CAT(n, SUF)

#ifndef VC2_LIFT_COMMON
#define VC2_LIFT_COMMON
#define VC2_G 2 // guard samples on each side of a half row
typedef int16_t vc2_v16 __attribute__((vector_size(16), aligned(2), may_alias));
enum { VC2_OUT_I32, VC2_OUT_I16, VC2_OUT_PIX };
// reach of a step into the other band: rows/samples i+lo .. i+hi
static inline void vc2_step_reach(const vc2_step *s, int *lo, int *hi)
{
    if (s->kind == VC2_K_ONE) *lo = *hi = 0;
    else if (s->kind == VC2_K_DD4) { *lo = s->tgt ? -1 : -2; *hi = s->tgt ? 2 : 1; }
    else { *lo = s->tgt ? 0 : -1; *hi = s->tgt ? 1 : 0; }
}
#endif

// dst[x] = src[x] + sign * f(a, b, c, d); the taps are the other band at [i-2..i+1] (tgt 0) or [i-1..i+2] (tgt 1)
static void FN(vc2_lift)(T *dst, const T *src, const T *a, const T *b, const T *c, const T *d, int n,
                         const vc2_step *s, int sign)
{
    const int sh = s->shift, r = sh ? 1 << (sh - 1) : 0, mul = s->mul;
    int x = 0;
    if (s->kind == VC2_K_ONE) b = s->tgt ? b : c;
#ifdef T16
    const int dsh = sh - 3, dm = (1 << (dsh > 0 ? dsh : 0)) - 1;
#define LD(p) (*(const vc2_v16 *)(p))
#define ST(f) do { if (sign > 0) *(vc2_v16 *)(dst + x) = LD(src + x) + (f); else *(vc2_v16 *)(dst + x) = LD(src + x) - (f); } while (0)
    switch (s->kind) {
    case VC2_K_AVG2:
        for (; x + 8 <= n; x += 8) { vc2_v16 f = (LD(b + x) + LD(c + x) + (int16_t)r) >> sh; ST(f); }
        break;
    case VC2_K_DD4:
        for (; x + 8 <= n; x += 8) {
            vc2_v16 s1 = LD(b + x) + LD(c + x), s2 = LD(a + x) + LD(d + x);
            vc2_v16 f = (s1 >> dsh) + ((s1 - s2 + (int16_t)r + ((s1 & (int16_t)dm) << 3)) >> sh);
            ST(f);
        }
        break;
    case VC2_K_ONE:
        for (; x + 8 <= n; x += 8) { vc2_v16 f = (LD(b + x) + (int16_t)r) >> sh; ST(f); }
        break;
    }
#undef LD
#undef ST
#endif
    switch (s->kind) {
    case VC2_K_AVG2:
        if (sign > 0) for (; x < n; x++) dst[x] = src[x] + ((b[x] + c[x] + r) >> sh);
        else for (; x < n; x++) dst[x] = src[x] - ((b[x] + c[x] + r) >> sh);
        break;
    case VC2_K_DD4:
        if (sign > 0) for (; x < n; x++) dst[x] = src[x] + ((9 * (b[x] + c[x]) - (a[x] + d[x]) + r) >> sh);
        else for (; x < n; x++) dst[x] = src[x] - ((9 * (b[x] + c[x]) - (a[x] + d[x]) + r) >> sh);
        break;
    case VC2_K_MUL2:
        if (sign > 0) for (; x < n; x++) dst[x] = src[x] + ((mul * (b[x] + c[x]) + r) >> sh);
        else for (; x < n; x++) dst[x] = src[x] - ((mul * (b[x] + c[x]) + r) >> sh);
        break;
    case VC2_K_ONE:
        if (sign > 0) for (; x < n; x++) dst[x] = src[x] + ((b[x] + r) >> sh);
        else for (; x < n; x++) dst[x] = src[x] - ((b[x] + r) >> sh);
        break;
    }
}

// replicate the edge samples of a half row into its guards
static inline void FN(vc2_guard)(T *p, int w)
{
    p[-1] = p[-2] = p[0];
    p[w] = p[w + 1] = p[w - 1];
}

// One step along a row: lo / hi are the two half rows (with guards), in place. sign as in vc2_lift.
static inline void FN(vc2_lift_h)(T *lo, T *hi, int w, const vc2_step *s, int sign)
{
    if (s->tgt == 0) { FN(vc2_guard)(hi, w); FN(vc2_lift)(lo, lo, hi - 2, hi - 1, hi, hi + 1, w, s, sign); }
    else { FN(vc2_guard)(lo, w); FN(vc2_lift)(hi, hi, lo - 1, lo, lo + 1, lo + 2, w, s, sign); }
}

typedef struct {
    const vc2_wavelet *wl;
    int w, h;              // band size
    const T *band[4];      // LL, HL, LH, HH
    ptrdiff_t bstride;
    int out_kind;
    void *out;             // next level's LL (2w x 2h) or the picture plane
    ptrdiff_t ostride;     // in samples
    int ow, oh;            // picture crop (VC2_OUT_PIX)
    int pix_off, pix_max;
    uint8_t *nv20;         // VC2_OUT_PIX: write packed NV20 rows here instead (see vc2_dec.c)
} FN(vc2_level);

static inline int FN(vc2_strip_halfw)(int w) { return (w + 2 * VC2_G + 7) & ~7; }
// scratch samples needed for strips of up to `rows` rows
static inline size_t FN(vc2_strip_scratch)(int w, int rows) { return (size_t)2 * (rows + 12) * 2 * FN(vc2_strip_halfw)(w); }

static __attribute__((unused)) void FN(vc2_emit_row)(const FN(vc2_level) *L, int oy, const T *lo, const T *hi);

// Inverse transform of band rows [y0, y1) into output rows [2*y0, 2*y1). Reads only the bands, so strips
// can run in any order and in parallel.
static __attribute__((unused)) void FN(vc2_synth_strip)(const FN(vc2_level) *L, int y0, int y1, T *tmp)
{
    const vc2_wavelet *wl = L->wl;
    const int w = L->w, h = L->h, ns = wl->nsteps, hw = FN(vc2_strip_halfw)(w), rw = 2 * hw;
    int r0[4], r1[4], need0[2] = { y0, y0 }, need1[2] = { y1, y1 };
    for (int j = ns - 1; j >= 0; j--) {
        const vc2_step *s = &wl->st[j];
        int t = s->tgt, lo, hi;
        vc2_step_reach(s, &lo, &hi);
        r0[j] = need0[t]; r1[j] = need1[t];
        int a = r0[j] + lo, b = r1[j] + hi;
        if (a < 0) a = 0;
        if (b > h) b = h;
        if (a < need0[!t]) need0[!t] = a;
        if (b > need1[!t]) need1[!t] = b;
    }
    int base = need0[0] < need0[1] ? need0[0] : need0[1];
    T *tb[2] = { tmp, tmp + (size_t)(y1 - y0 + 12) * rw };
    int written[2] = { 0, 0 };
#define ROW(which, half, i) (written[which] ? tb[which] + (size_t)((i) - base) * rw + ((half) ? hw + VC2_G : VC2_G) \
                                             : L->band[(which) * 2 + (half)] + (size_t)(i) * L->bstride)
#define CLAMP(i) ((i) < 0 ? 0 : (i) >= h ? h - 1 : (i))
    for (int j = 0; j < ns; j++) {
        const vc2_step *s = &wl->st[j];
        int t = s->tgt, o = !t, off = t ? -1 : -2;
        for (int i = r0[j]; i < r1[j]; i++)
            for (int half = 0; half < 2; half++) {
                T *dst = tb[t] + (size_t)(i - base) * rw + (half ? hw + VC2_G : VC2_G);
                const T *src = ROW(t, half, i);
                const T *a = ROW(o, half, CLAMP(i + off)), *b = ROW(o, half, CLAMP(i + off + 1)),
                        *c = ROW(o, half, CLAMP(i + off + 2)), *d = ROW(o, half, CLAMP(i + off + 3));
                FN(vc2_lift)(dst, src, a, b, c, d, w, s, s->sign);
            }
        written[t] = 1;
    }
#undef ROW
#undef CLAMP
    for (int i = y0; i < y1; i++)
        for (int t = 0; t < 2; t++) {
            T *row = tb[t] + (size_t)(i - base) * rw, *lo = row + VC2_G, *hi = row + hw + VC2_G;
            for (int j = 0; j < ns; j++) FN(vc2_lift_h)(lo, hi, w, &wl->st[j], wl->st[j].sign);
            FN(vc2_emit_row)(L, 2 * i + t, lo, hi);
        }
}

static void FN(vc2_emit_row)(const FN(vc2_level) *L, int oy, const T *lo, const T *hi)
{
    const int w = L->w, sh = L->wl->shift, r = sh ? 1 : 0;
    if (L->out_kind == VC2_OUT_I32) {
        int32_t *o = (int32_t *)L->out + (size_t)oy * L->ostride;
        for (int x = 0; x < w; x++) { o[2 * x] = (lo[x] + r) >> sh; o[2 * x + 1] = (hi[x] + r) >> sh; }
    } else if (L->out_kind == VC2_OUT_I16) {
        int16_t *o = (int16_t *)L->out + (size_t)oy * L->ostride;
        for (int x = 0; x < w; x++) { o[2 * x] = (lo[x] + r) >> sh; o[2 * x + 1] = (hi[x] + r) >> sh; }
    } else {
        if (oy >= L->oh) return;
        uint16_t *o = (uint16_t *)L->out + (size_t)oy * L->ostride;
        const int off = L->pix_off, max = L->pix_max, n = L->ow / 2;
        for (int x = 0; x < n; x++) {
            int a = ((lo[x] + r) >> sh) + off, b = ((hi[x] + r) >> sh) + off;
            o[2 * x] = a < 0 ? 0 : a > max ? max : a;
            o[2 * x + 1] = b < 0 ? 0 : b > max ? max : b;
        }
        if (L->ow & 1) { int a = ((lo[n] + r) >> sh) + off; o[2 * n] = a < 0 ? 0 : a > max ? max : a; }
    }
}
#undef FN
#undef CAT
#undef CAT2
