#!/usr/bin/env python3
# Worst-case values of the VC-2 LeGall 5,3 transform at depth 4 (docs/FPGA.md section 6), for a hardware decoder.
#   fpga-worst.py bounds                     the largest magnitude each band can reach for any picture with samples
#                                            in [-512, 511] (10 bit, offset removed), from the filters' L1 norms
#   fpga-worst.py picture W H out.yuv [fill] a 4:2:2 10-bit picture (2 frames) with one patch per band and level that
#                                            drives that band's coefficient to its worst case (pixel = full scale *
#                                            sign(column filter) * sign(row filter)); "fill" repeats the patches over
#                                            the whole frame
#   fpga-worst.py noisy in.yuv out.yuv       the patches at 60 % with gaussian noise (sigma 220), so that a rate-
#                                            controlled encode needs coarse quantisers
# The forward transform is the encoder's (lib/vc2_enc.c): per level samples << 1, H -= (L[i] + L[i+1]) / 2,
# L += (H[i-1] + H[i]) / 4, rows then columns. Rounding is left out: the measured values come within 0.1 %.
import random, struct, sys

N = 256
def analyse(x):
    L, H = x[0::2], x[1::2]
    n = len(L)
    H = [H[i] - (L[i] + L[min(i + 1, n - 1)]) / 2 for i in range(n)]
    L = [L[i] + (H[max(i - 1, 0)] + H[i]) / 4 for i in range(n)]
    return L, H

def responses():
    # per analysis step k (1 = the first, finest = synthesis level 4): the centre coefficient's lo and hi filters
    lo = [[1.0 if j == i else 0.0 for j in range(N)] for i in range(N)]
    r = {}
    for k in range(1, 5):
        pairs = [analyse(v) for v in lo]
        c = len(pairs[0][0]) // 2
        r[k] = ([p[0][c] for p in pairs], [p[1][c] for p in pairs])
        lo = [p[0] for p in pairs]
    return r

def bounds():
    r = responses()
    l1 = lambda f: sum(abs(v) for v in f)
    print("level  L1(lo)  L1(hi)   HL / LH       HH   LL out of this analysis (= the next level's LL input)")
    for k in range(1, 5):
        a, b = l1(r[k][0]), l1(r[k][1]); g = 512 * 2 ** k
        print(f"  {5 - k}    {a:6.3f}  {b:6.3f}  {g * a * b:9.1f}  {g * b * b:9.1f}  {g * a * a:9.1f}")

def picture(W, H, out, fill):
    r = responses()
    sg = lambda v: 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    patches = []
    for k in range(1, 5):
        l, h = r[k]
        patches += [(l, h), (h, l), (h, h), (l, l)] # HL (horizontally high), LH, HH, and the LL out of this level
    def plane(w, h):
        p = [[0] * w for _ in range(h)]
        per = w // N
        n = per * (h // N) if fill else len(patches)
        for i in range(n):
            y0, x0 = (i // per) * N, (i % per) * N
            if y0 + N > h: break
            fv, fh = patches[i % len(patches)]
            for y in range(N):
                sy = sg(fv[y])
                if not sy: continue
                row = p[y0 + y]
                for x in range(N):
                    s = sy * sg(fh[x])
                    if s: row[x0 + x] = 511 if s > 0 else -512
        return p
    planes = [plane(W, H), plane(W // 2, H), plane(W // 2, H)]
    with open(out, "wb") as f:
        for _ in range(2):
            for p in planes:
                for row in p: f.write(struct.pack(f"<{len(row)}H", *[v + 512 for v in row]))

def noisy(src, out):
    random.seed(1)
    d = open(src, "rb").read(); n = len(d) // 2
    v = struct.unpack(f"<{n}H", d)
    o = [min(1023, max(0, int(512 + 0.6 * (x - 512) + random.gauss(0, 220)))) for x in v]
    open(out, "wb").write(struct.pack(f"<{n}H", *o))

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["bounds"]: bounds()
    elif a[:1] == ["picture"]: picture(int(a[1]), int(a[2]), a[3], a[4:5] == ["fill"])
    elif a[:1] == ["noisy"]: noisy(a[1], a[2])
    else: print(__doc__ if __doc__ else open(__file__).read().split("\nimport")[0]); sys.exit(2)
