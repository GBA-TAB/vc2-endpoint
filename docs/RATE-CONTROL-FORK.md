# Rate control on the fork branch `fork/rc-bracketed`

**This branch is a fork of `main`, not mainline.** It changes how the encoder chooses the frame quantiser, and
with it the encoder output. Streams are standard VC-2 HQ either way: nothing changes in the decoder, the RTP
code or the player. `main` keeps the step search, and every speed change on `main` stays byte-identical with
the previous `main`. On this branch, `VC2_RC_STEP=1` runs `main`'s search again (byte-identical with `main`).

## What `main` does

The encoder fits one quantiser index per frame (or per `--rc-rows` unit) to the budget by exact counting: a
count is one pass over the unit's coefficients at one index. Then a refinement moves some slices one index
lower while budget is left. `main` starts at the last frame's index and steps:
- one index up per count while the unit does not fit;
- one index down per count while the size is at most 82 % of the budget.

Two things go wrong with that:
- **Scene cuts.** Each index step is a full count. Between very different pictures the index moves by up to
  40, so one frame costs up to 47 counts. On a Pi 5 that is 154 ms of rate control (two threads, 3 bpp),
  several late frames in a live sender.
- **Flat size curves.** On synthetic or 8-bit-like content the size hardly changes over the fine indices
  (a test pattern: q 0 to q 5 within 1 %). `main` stops at the first index above 82 % fill, for example q 5,
  although q 0 still fits.

## What this branch does

- **Steady state, unchanged:** the last frame's index is counted first and kept when it fits with more than
  82 % fill. That is one count, as on `main`.
- **Otherwise a bracketed search** for the lowest index that fits:
  - `lo` is the largest index known not to fit, `hi` the smallest known to fit. Both start from every index
    already counted for these rows in this frame, which is free (the counts are cached). That makes the second
    pass cheap when the slice size scaler changes.
  - **Both ends known:** the secant between them on log2(size / budget). Illinois rule: an end that has stayed
    twice counts half, which gets the secant past the size floor of very coarse indices.
  - **One end known:** extrapolate with the slope of the last two points on that side, or with one point the
    slope the last search measured. The step is capped at twice the previous step (8 at first).
  - Ends at `hi = lo + 1`, so `hi` is exact under the same assumption as `main`: the size does not grow as the
    index rises.
- **Drift:** when the refinement moved every slice of the first unit one index lower, the next frame starts
  there. This corrects the 82 % shortcut over a few frames, with no extra count.

## Measurements

**Test sequence:** 99 frames, 1080p, 4:2:2, 10-bit, with hard cuts between:
- a moving test pattern;
- a natural camera clip;
- flat grey;
- a synthetic stress picture;
- a fractal;
- noise;
- a worst-case picture;
- colour bars;
- a fade-in.

Every frame was checked against a full scan for the lowest index that fits.

**Counts per frame**, x86, two threads:

| bpp | counts per frame (search) | worst frame | frames not at the lowest fitting index |
|---|---|---|---|
| 1.5 | 2.92 -> 1.64 | 47 -> 9 | 30 -> 3 |
| 3 | 3.29 -> 1.51 | 42 -> 9 | 42 -> 2 |
| 8 | 2.35 -> 1.15 | 31 -> 6 | 10 -> 0 |

**Quality** (luma PSNR per segment): never lower than `main`. Where `main` got stuck it is higher:

| segment | `main` | this branch |
|---|---|---|
| test pattern, 3 bpp | 69.8 dB | 89.2 dB |
| test pattern, 1.5 bpp | 59.1 dB | 60.9 dB |
| fractal, 8 bpp | 66.4 dB | 72.9 dB |

All other segments are identical.

**Budget and modes:**
- The budget is held on every frame.
- Over a stretch of 40 frames made only of cuts, the output is byte-identical with `main`; only the counting
  is faster.
- With `--no-truncate` the budget is used better (mean index 19.6 -> 17.5).

**Pi 5, 3 bpp**, the same 40 frames of cuts:

| threads | rate control, mean | worst frame |
|---|---|---|
| 2 | 13.1 -> 5.6 ms | 154 -> 28 ms |
| 4 | 8.1 -> 3.5 ms | 82 -> 16 ms |

Frames without a picture change cost the same as on `main`.

**Checks:**
- `scripts/verify.sh` 254/254.
- ffmpeg's decoder equals ours on multi-frame `--no-truncate` streams with cuts at 1.5, 3 and 8 bpp.
- ASan + UBSan: 320 runs, no report.
- `scripts/enc-check.sh` (the fast paths against the plain ones, byte for byte) on this branch, all identical:
  - x86: 0 of 312 cases differ;
  - Pi 5: 0 of 312;
  - RK3588 board: 0 of 366, including 54 cases on its GPU path.

## Open

- The 82 % shortcut is still a heuristic. The drift corrects it over frames, but only for the first
  `--rc-rows` unit.
- The search assumes the size falls monotonically with the index, as `main` does. A frame where that does not
  hold could end on a different index than a full scan; none did on the test sequence.
- Not on `main` until it has been reviewed.
