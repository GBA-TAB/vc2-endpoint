# vc2-endpoint: VC-2 (SMPTE ST 2042, HQ profile) encoder, decoder and player

A royalty-free alternative to JPEG XS for our low-latency video endpoints: encode on x86, decode and show
on embedded hardware (Firefly ROC-RK3588-RT) with sub-frame latency. Everything here is our own code.
Measured 2026-11-20; the JPEG XS figures it is compared with are in `jxs-endpoint/README.md` (sections 6b, 6d),
on the same clips and the same board.

**Result in one paragraph.** With our encoder VC-2 reaches the PSNR of SVT-JPEG-XS (at its defaults) with
15-25 % less bitrate on our test clip, and our decoder needs 15 ms of one Cortex-A76 core per 1080p frame
at 3 bpp (JPEG XS on the same board: 41 ms CPU-only, 28 ms with the Mali). 1080p50 plays on the
Firefly's screen with 0 missed vblanks out of 3000 at 0.96 cores. Fed slice row by slice row, the last
picture line is ready 0.65 ms after the last byte of a frame. UHD decodes at 38 fps (3 bpp) to 46 fps (2 bpp):
2160p50 is not reached on this board's CPU. The quality figures are PSNR on 4 frames of one clip; nobody has
looked at the pictures critically yet.

## Layout

| Path | What |
|---|---|
| `lib/vc2.h` | public API: stream helpers, decoder (whole frame or fed in pieces), encoder |
| `lib/vc2_common.c` | quantiser tables, default matrices, wavelet definitions, headers, thread pool |
| `lib/vc2_lift.h` | lifting kernels and the strip-wise inverse wavelet (template: 32-bit and 16-bit) |
| `lib/vc2_dec.c` | decoder |
| `lib/vc2_enc.c` | encoder |
| `tools/` | `vc2enc`, `vc2dec` (to planar yuv), `vc2bench` (speed, CPU, latency), `psnr` |
| `player/vc2play.c` | DRM player (NV20 on the CRTC's primary plane, vblank pacing), `play.sh` |
| `scripts/verify.sh` | bit-exactness against ffmpeg's decoder |
| `docs/RTP.md` | RFC 8450 carriage, design only |

Licence: none chosen yet. Suggestion: MIT (or BSD-2-Clause). Nothing was copied in: ffmpeg's `vc2enc` /
`diracdec` (LGPL) were read as a reference and used as a test oracle; the SMPTE documents are not in the
repo and are not quoted beyond the constants the code needs. No JPEG XS / SVT code is here. The DRM part of
the player follows our own `jxs-endpoint/player/jxsplay.c`.

Supported: HQ profile, progressive, 4:2:2 10-bit (the tested case); 4:4:4 / 4:2:0 and 8 / 12 bit are coded
for but untested. Wavelets 0-4 (Deslauriers-Dubuc 9,7 and 13,7, LeGall 5,3, Haar with and without shift),
depth 1-6, any slice grid. Not supported: low-delay profile, interlaced, fragments, the Fidelity filter;
Daubechies 9,7 (index 6) is implemented but differs from ffmpeg's decoder (not resolved which is right),
so do not use it for interchange.

## 1. ffmpeg's VC-2 encoder: why it breaks, and how far behind it is

- **Below about 2.1 bpp (4:2:2) it cannot work at all.** In the HQ profile a zero coefficient costs one
  bit; 4:2:2 has two coefficients per pixel, so writing every coefficient costs at least 2 bpp plus 4 bytes
  per slice. The format's way out is that a slice component may end early (the missing bits read as
  zeros); ffmpeg never uses it. Asked for 2 bpp, its rate control runs the quantiser index to the top
  (`Qavg: 114`) and every picture comes out grey (13.7 dB, whatever the wavelet or slice size).
- Above that, the same thing costs quality: at 3 bpp, 2 of the 3 bits per pixel are that one bit per
  coefficient. Writing
  all coefficients, our encoder gets 49.95 dB at 3 bpp; ffmpeg 49.2 (default slices) to 50.3 (64x32);
  with truncation ours gets 54.95 dB. So **the 4 dB gap to JPEG XS was the encoder, not the format.**
- Smaller things: every slice gets the same byte budget (flat areas waste it; larger slices hide that,
  +1.1 dB at 64x32); `-qm flat` and `color` are worse than the default matrix for PSNR; 256-pixel slices
  give 31.8 dB (not investigated); `haar_noshift` is broken in the encoder (our Haar streams decode
  correctly in ffmpeg's decoder).
- Speed: 73 ms per 1080p frame on caspar's two free cores (13.6 fps).
- Two bugs in ffmpeg's **decoder**, found on the way (ours follows the format; both are reproduced by
  `scripts/verify.sh` if its workarounds are removed):
  - it parses one byte past a slice component that ends early, which can add a stray coefficient
    (a few wrong samples per frame); and it misreads zero-length components, so our encoder never
    writes one;
  - its x86 SIMD dequantisation is wrong when a slice's subband width is not a multiple of 8 (e.g.
    240-pixel slices). `-cpuflags 0`, and ffmpeg on arm64, decode those correctly.

## 2. Encoder

Own encoder (`lib/vc2_enc.c`, C, AVX2 through the compiler's vectoriser), because the gain is in the
rate control and that part of ffmpeg would have to be replaced anyway.

- Forward wavelet in 32 bits; one quantiser index for the whole frame (the lowest that fits the frame's
  byte budget), then single slices one index lower until the budget is used.
- **Tail truncation**: per slice component the encoder drops the trailing subband rows whose bits are worth
  more than the distortion of dropping them (`--tail`, default 0.2; 0 drops only all-zero tails), and ends the
  component after the last nonzero coefficient.
- Per-frame budget is never exceeded. `--rc-rows N` budgets per N slice rows instead (for a sender that
  must not wait for the whole frame): -0.23 dB at 3 bpp for N = 1.
- Defaults: LeGall 5,3, depth 4, slices 128x16, the format's default quantisation matrix (written out).

PSNR in dB on the 4-frame clips (Y, and the average over all samples in brackets):

| bpp | 1080p: JPEG XS (SVT defaults) | 1080p: VC-2 LeGall 5,3 | 1080p: VC-2 DD 9,7 | UHD: JPEG XS | UHD: VC-2 LeGall 5,3 |
|---|---:|---:|---:|---:|---:|
| 1.5 | 48.9 (50.6) | 50.2 (51.5) | 50.3 (51.6) | 50.1 (51.8) | 51.2 (52.5) |
| 2 | 50.3 (51.9) | 51.7 (52.7) | 51.7 (52.8) | 51.6 (53.2) | 52.8 (53.7) |
| 2.5 | 51.6 (53.0) | 53.4 (53.9) | 53.3 (54.1) | 53.0 (54.4) | 54.4 (55.0) |
| 3 | 53.0 (54.2) | 55.0 (55.2) | 54.8 (55.5) | 54.4 (55.7) | 55.8 (56.1) |
| 4 | 55.8 (56.9) | 57.7 (57.7) | 57.7 (58.1) | 57.3 (58.3) | 58.8 (58.6) |
| 5 | 59.1 (59.5) | 59.6 (59.4) | 60.1 (60.3) | 60.3 (60.9) | 60.1 (59.9) |

- ffmpeg's VC-2 encoder at 3 bpp: 49.2 dB (1080p), 48.5 dB (UHD).
- Other wavelets at 3 bpp, 1080p: DD 13,7 55.0 (55.7); Haar with shift 53.2 (54.1). Depth 3: -0.5 dB; depth 5:
  +0.1 dB. Slice size 32x16 to 240x16: within 0.1 dB.
- Read with care: the source was H.264 (PSNR is high, see jxs-endpoint 6d); JPEG XS weights its subbands for
  the eye, not for PSNR, so PSNR under-rates it; SVT's "signs handling 2" adds 1.3 dB to XS at +42-68 %
  decoder cost, which would put the two level. A viewing test on camera content is still to be done, in
  particular for what tail truncation does to low-contrast texture.

Speed on caspar (i7-7800X, cores 10-11 at nice 19, shared with other work), `--threads 2`:

| | wall per frame | CPU per frame | fps |
|---|---:|---:|---:|
| 1080p 3 bpp, LeGall | 20.1-20.6 ms | 36 ms | 48-50 |
| 1080p 2 bpp, LeGall | 17.7 ms | 31 ms | 57 |
| 1080p 3 bpp, DD 9,7 | 22.0 ms | 37 ms | 46 |
| 1080p 3 bpp, one thread | 31.5 ms | 31.5 ms | 32 |
| UHD 3 bpp | 86 ms | 156 ms | 11.6 |

So 1080p50 in real time on the two cores holds up to about 2.5 bpp and is at the limit at 3 bpp; UHD50 would
need about eight such cores. (An AVX-512 build gains nothing: the time is in memory order and the
bit writer, not in vector width.) Where the time goes at 3 bpp, one thread, roughly: wavelet 9 ms, counting
13 ms (two passes), writing 9 ms.

## 3. Decoder

- **Entropy decoding**: a 12-bit lookup yields all coefficients complete within those bits (up to 12 zeros,
  up to 3 nonzero values, stored without a branch); four slice components are decoded side by side, since one
  lookup chain is latency-bound. Long codes and coefficients beyond 16 bits fall back to a bit-serial
  decoder.
- **Dequantisation** is a separate vector pass per subband row (16-bit lanes for the last level, 32-bit for
  the deep ones).
- **Inverse wavelet** in strips of subband rows: vertical lifting on a strip, horizontal lifting per row,
  interleave / round / offset / clip into the output. The last level (3/4 of the samples) runs in 16 bits,
  the deep levels in 32 bits. The 16-bit path wraps if a last-level sum exceeds 16 bits, which 10-bit video
  cannot produce (the deep levels, where a full-swing checkerboard does overflow 16 bits, are 32-bit).
  SIMD is written with GCC vector extensions (NEON on arm64, SSE/AVX2 on x86), plus NEON intrinsics for the
  NV20 packing.
- **Cascade**: slice rows are decoded in order and every wavelet level runs as far as the data allows, so
  the subbands live in ring buffers of 16-64 rows instead of whole planes, and picture lines come out about
  30 lines (LeGall) or 60 lines (DD 9,7) behind the slice row being decoded.
- **Threads**: each thread runs its own cascade over a band of slice rows, started 2 (LeGall) or 4 (DD 9,7)
  slice rows early (16-line slices); after that overlap its output does not depend on anything above. No locks, no shared
  state but the output picture. `vc2_dec_set_weights` sizes the bands for unequal cores.
- **Sub-frame**: `vc2_dec_begin` / `vc2_dec_feed(bytes so far)` run one cascade as the bytes arrive.
- Output: planar 16-bit, or packed NV20 written straight into the display's buffers.

Bit-exactness (`scripts/verify.sh`, 254 cases, all pass): streams from ffmpeg's encoder (3 wavelets, 3 slice
sizes, 2 depths, 2 rates) and from ours (5 wavelets, 4 slice sizes, 3 quantisers, 2 depths), each decoded
with one thread, three threads and fed in 40 pieces, against ffmpeg's decoder (C path). Truncated streams
are compared with the same content written out in full, because of the ffmpeg bug above. The Firefly's
output has the same checksums as caspar's for every clip measured below. Address and UB sanitizers are
clean on the encoder and on all decoder modes.

Firefly (RK3588, A76 at 2.0 GHz, no fan), 1080p LeGall 5,3 at 3 bpp, one A76 core, ms per frame:

| Step (clip with 64x16 slices unless noted) | total | entropy | wavelet |
|---|---:|---:|---:|
| ffmpeg 4.4 `dirac` decoder (no NEON), for reference | 49 | | |
| (a) portable C: bit-serial entropy decoding straight into the subbands, strip wavelet | 26.6 | 18.9 | 7.6 |
| (b) table lookups, vector dequantisation | 20.9 | 12.9 | 7.6 |
| four streams side by side, dequantisation in memory order | 18.5 | 10.4 | 7.7 |
| ring-buffer cascade | 17.7 | 10.4 | 7.0 |
| 32-bit vector dequantisation for the deep levels | 16.9 | 9.6 | 7.0 |
| 128x16 slices (the encoder's default now) | **15.4** | 8.3 | 6.9 |

On the final decoder, switching single parts back (128x16 clip): everything in 32 bits (`VC2_NO16=1`) 18.7 ms
(entropy 9.3 + wavelet 9.3); bit-serial entropy decoding (`VC2_REF_ENTROPY=1`) 29.8 ms (22.6 + 7.0). Of the
8.3 ms of entropy decoding, the lookups are about 4.3 ms, the rest is dequantisation and set-up.

| 1080p, 3 bpp unless noted | fps flat out | decode time | CPU per frame |
|---|---:|---:|---:|
| 1 x A76 | 64.8 | 15.4 ms | 15.4 ms |
| 2 x A76 | 116 | 8.6 ms | 16.4 ms |
| 3 x A76 (cores 4-6) | 163 | 6.1 ms | 17.3 ms |
| 3 x A76, NV20 output | 150 | 6.7 ms | 19.2 ms |
| 3 x A76, 2 bpp | 187 | 5.4 ms | 15.1 ms |
| 3 x A76, DD 9,7 | 126 | 7.9 ms | 22.0 ms |
| 1 x A76, DD 9,7 | 56.5 | 17.7 ms | 17.7 ms |
| 1 x A55 | 17.6 | 56.9 ms | 56.9 ms |

Latency at 50 fps pacing (time from the arrival of a frame's last byte to its last line being ready):

| Mode | avg | max |
|---|---:|---:|
| whole frame, 1 x A76 | 15.7 ms | 17.0 ms |
| whole frame, 3 x A76 | 6.3 ms | 6.6 ms |
| sub-frame (fed in 68 slice rows spread over the frame period), 1 x A76 | **0.65 ms** | 0.67 ms |
| the same, 2 bpp | 0.61 ms | 1.09 ms |
| the same, DD 9,7 | 1.01 ms | 1.17 ms |

In sub-frame mode the first lines are ready 0.26 ms after the first slice row; one A76 core keeps up with
1080p50 (15.8 ms of work per 20 ms).

**(c) GPU: not done, by the numbers.** The inverse wavelet is 7 ms of the 15 ms per 1080p frame and runs in
cache. The Mali offload of the JPEG XS work cost 3.4 ms of GPU time per 1080p frame and still left
28 ms of CPU per frame; here the whole CPU decode is 15-17 ms, so at 1080p there is nothing to win. It
could pay for UHD (section 5).

## 4. Player

`vc2play` decodes into NV20 scanout buffers (three, flipped on vblank). Firefly, HDMI-A-2 at 1920x1080@50,
3 threads on cores 4-6:

    frames 3000 in 60.0 s (50.01 fps), missed vblanks 0, decode avg 6.75 ms max 7.58 ms, cpu 19.17 ms/frame (0.96 cores)

The NV20 output is checked against the planar output in `vc2bench --nv20` (identical). The picture on the
screen itself was not looked at by a person during these runs.

## 5. VC-2 against JPEG XS

Same clips, same board. JPEG XS = SVT-JPEG-XS at its defaults (jxs-endpoint 6d); VC-2 = this repo, LeGall
5,3, defaults.

| | JPEG XS | VC-2 |
|---|---|---|
| PSNR Y at 3 bpp, 1080p / UHD | 53.0 / 54.4 dB | 55.0 / 55.8 dB |
| PSNR Y at 2 bpp, 1080p / UHD | 50.3 / 51.6 dB | 51.7 / 52.8 dB |
| Bitrate for the quality of XS at 3 bpp, 1080p / UHD | 3 bpp (311 Mbit/s at 1080p50) | 2.4 / 2.5 bpp (249 Mbit/s) |
| Bitrate for the quality of XS at 2 bpp, 1080p / UHD | 2 bpp (207 Mbit/s) | 1.5 / 1.6 bpp (156 Mbit/s) |
| Decoder CPU per 1080p frame at 3 bpp (Firefly) | 40.7 ms CPU path, 28.1 ms Mali path | 15.4 ms (1 thread), 17.3 ms (3 threads), 19.2 ms incl. NV20 output |
| ... at 2 bpp | 35.9 / 23.5 ms | 13.5 ms (1 thread), 15.1 ms (3 threads) |
| ... at equal quality (XS 3 bpp, VC-2 2.4 bpp) | 40.7 / 28.1 ms | about 14.3 ms (between the 2 and 3 bpp figures) |
| GPU needed | for the lower figure | no |
| Decoder CPU per UHD frame at 3 bpp | 157 ms CPU path, 102 ms Mali path | 61 ms (1 thread), 72 ms (3 threads) |
| UHD frame rate on the Firefly | about 44 fps (Mali, 2 frames in flight; other clip) | 38.5 fps at 3 bpp, 42.4 at 2 bpp (3 x A76); 42.8 / 45.8 with the four A55 added |
| Latency after the last byte, 1080p50 | whole frames only; not measured this way (UHD: 39-47 ms, 6b) | 6.3 ms whole frame (3 threads); 0.65 ms sub-frame |
| Encoder, 1080p 3 bpp on caspar's 2 cores | (SVT, not measured here) | 48-50 fps |
| Royalties | patent pool (Vectis) | none |

**UHD 2160p50 is not reached** on the Firefly: 26 ms per frame on the three free A76 cores at 3 bpp (20 ms
needed), 23.6 ms at 2 bpp. The A55 cores are 3.7 times slower and help little (42.8 fps at 3 bpp, 45.8 at
2 bpp), because their short bands pay the overlap. The screen there shows UHD at 30 Hz only, which the
decoder does reach. What could close the gap: the last wavelet level on the Mali (33 of the 72 ms; the
entropy part alone is 13 ms per frame on three A76), or the fourth A76 (core 7, kept for the display today).

## 6. Open

- A viewing test on camera content (tail truncation, slice-boundary visibility at low rates); VC-2 against
  JPEG XS by eye, not by PSNR.
- UHD50: the GPU path for the last level, or more cores; UHD on screen (30 Hz) was not run.
- RTP (RFC 8450): design in `docs/RTP.md`; needs a per-slice size cap in the rate control, a fragmenter and
  loss concealment. Nothing was sent on a network.
- Encoder: 3 bpp at 1080p50 is at the limit of two cores; a 16-bit forward wavelet and a row-wise (sub-frame)
  mode are the next steps. Its latency today is one frame.
- Line-locked display (scan-out following the decoder by a fixed number of lines) for a true sub-frame
  glass-to-glass path; the player shows whole frames.
- 4:2:0 / 4:4:4, 8 / 12 bit, Daubechies 9,7 against a second reference decoder, interlaced.
- A cross-check on the Pi 5 (offline during this work).

## Build and reproduce

    # caspar (builds and runs only on cores 10-11)
    nice -n 19 taskset -c 10,11 make BUILD=$HOME/build/vc2-endpoint
    B=$HOME/build/vc2-endpoint; T="nice -n 19 taskset -c 10,11"
    $T $B/vc2enc --bpp 3 --threads 2 --loops 25 ~/build/jxs-sweep/k1080.yuv k1080_b3_w1.vc2   # speed; first pass is written
    $T $B/vc2enc -s 3840x2160 --bpp 3 ~/build/jxs-sweep/kuhd.yuv kuhd_b3_w1.vc2
    $T $B/vc2dec k1080_b3_w1.vc2 out.yuv && $B/psnr ~/build/jxs-sweep/k1080.yuv out.yuv 1920 1080
    RUN="$T" scripts/verify.sh $B ~/build/jxs-sweep/k1080.yuv 1920 1080 ~/build/vc2/verify    # about 12 minutes
    # ffmpeg, for comparison (-b:v = pixels x bpp x 50); decode with -fps_mode passthrough, or frames repeat
    ffmpeg -f rawvideo -pix_fmt yuv422p10le -s 1920x1080 -r 50 -i k1080.yuv -c:v vc2 -strict -1 -b:v 311040000 ff.vc2

    # Firefly (sources in ~/src/vc2-endpoint, clips in ~/vc2/final)
    make BUILD=$HOME/vc2/build all player
    b=~/vc2/build/vc2bench; f=~/vc2/final/k1080_b3_w1.vc2
    taskset -c 4 $b --frames 150 $f                                   # one A76
    $b --frames 300 --threads 3 --cpus 4,5,6 [--nv20] $f              # three A76
    $b --frames 250 --fps 50 --threads 3 --cpus 4,5,6 $f              # paced: latency of whole-frame decoding
    $b --frames 250 --fps 50 --subframe 68 --cpus 4 $f                # paced, fed slice row by slice row
    $b --frames 100 --threads 7 --cpus 4,5,6,0,1,2,3 --weights 1,1,1,0.18,0.18,0.18,0.18 ~/vc2/final/kuhd_b3_w1.vc2
    VC2_REF_ENTROPY=1 taskset -c 4 $b --frames 60 $f                  # bit-serial entropy decoder
    VC2_NO16=1 taskset -c 4 $b --frames 60 $f                         # everything in 32 bits
    sudo ~/src/vc2-endpoint/player/play.sh ~/vc2/build $f --frames 3000 --threads 3 --cpus 4,5,6

The Firefly throttles at 85 C after about a minute of full load: keep runs short and check
`/sys/class/thermal/thermal_zone0/temp` (it stayed at 52-59 C in these runs).
