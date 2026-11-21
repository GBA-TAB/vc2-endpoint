# vc2-endpoint: VC-2 (SMPTE ST 2042, HQ profile) encoder, decoder, RTP and player

VC-2 is the royalty-free intra-frame wavelet codec from the BBC (Dirac Pro). This repo has an encoder, a
low-latency decoder (CPU, with an optional GPU stage through OpenCL or Vulkan), the RTP payload of RFC 8450,
and a DRM player for embedded boards. It was written as an alternative to JPEG XS for low-latency video
endpoints: encode on x86, decode and show on an RK3588 board with sub-frame latency. C11, no dependencies
beyond pthreads (libdrm for the player, OpenCL / Vulkan only if asked for). Licence: MIT (`LICENSE`).

**Results in one paragraph** (measured 2026-11-20/21; details and commands below). On our test clip the
encoder reaches the PSNR of SVT-JPEG-XS (at its defaults) with 15-25 % less bitrate. The decoder needs
15 ms of one Cortex-A76 core per 1080p frame at 3 bpp (JPEG XS on the same board: 41 ms CPU-only, 28 ms with
the Mali). 1080p50 plays on the board's screen with 0 missed vblanks out of 3000 at 0.96 cores; fed slice
row by slice row (or packet by packet over RTP), the last picture line is ready 0.4-0.65 ms after the last
byte of a frame. UHD 2160p50 is reached with the last wavelet level on the Mali: 60 fps flat out on four A76
cores plus the GPU, 52 fps on three. The quality figures are PSNR on 4 frames of one clip; a viewing
test has not been done yet.

The comparison figures for JPEG XS come from our own measurements of SVT-JPEG-XS on the same clips and
board; that work is not part of this repo, and no JPEG XS code is.

## Layout

| Path | What |
|---|---|
| `lib/vc2.h` | public API: stream helpers, decoder (whole frame or fed in pieces), encoder, RTP |
| `lib/vc2_common.c` | quantiser tables, default matrices, wavelet definitions, headers, thread pool |
| `lib/vc2_lift.h` | lifting kernels and the strip-wise inverse wavelet (template: 32-bit and 16-bit) |
| `lib/vc2_fpga.h`, `vc2_fpga.c` | host side of a hardware decoder: subset check, the H2C descriptor (docs/FPGA.md section 9), a card model |
| `lib/vc2_xdma.h`, `vc2_xdma.c`, `vc2_xdma_model.c` | the card's host library (by pmod-b3: xdma windows, status block) and its software model (docs/FPGA.md section 10) |
| `lib/vc2_dec.c`, `lib/vc2_dec_gpu.h` | decoder, and its side of the GPU path |
| `lib/vc2_cl.c`, `lib/vc2_idwt.cl` | GPU backend: OpenCL (Mali), optional |
| `lib/vc2_vk.c`, `lib/vc2_idwt.comp` | GPU backend: Vulkan compute, optional |
| `lib/vc2_enc.c` | encoder |
| `lib/vc2_rtp.c` | RFC 8450 packetiser and receiver |
| `tools/` | `vc2enc`, `vc2dec` (to planar yuv), `vc2bench` (speed, CPU, latency), `vc2send` / `vc2recv` (RTP on loopback), `psnr`, `vc2dump` (every decoding stage to files, for a hardware decoder), `vc2fpga` (the H2C stream for the card, checked by a card model), `vc2xbench` (a stream through the card or its model: check, MB/s, latency) |
| `player/vc2play.c` | DRM player (NV20 on the CRTC's primary plane, vblank pacing), `play.sh` |
| `scripts/verify.sh` | bit-exactness against ffmpeg's decoder |
| `docs/RTP.md` | the RTP mapping and what is open |
| `docs/FPGA.md` | a hardware (FPGA) decoder: the subset, the exact arithmetic, the reference vectors (`scripts/fpga-vectors.sh`) |

Provenance: all code here was written for this repo. ffmpeg's `vc2enc` / `diracdec` were read as a
reference and ffmpeg is used as a test oracle (`scripts/verify.sh` runs the `ffmpeg` binary); none of its
code is included. The SMPTE documents are not included and not quoted; the code carries only the constants
the format needs. RFC 8450 is cited by section number.

Supported: HQ profile, progressive, 4:2:2 10-bit (the tested case); 4:4:4 / 4:2:0 and 8 / 12 bit are coded
for but untested. Wavelets 0-4 (Deslauriers-Dubuc 9,7 and 13,7, LeGall 5,3, Haar with and without shift),
depth 1-6, any slice grid. Not supported: low-delay profile, interlaced, the Fidelity filter;
Daubechies 9,7 (index 6) is implemented but differs from ffmpeg's decoder (not resolved which is right),
so do not use it for interchange.

## Build and run

    make BUILD=build                      # library and tools; gcc or clang, x86-64 or arm64
    make BUILD=build player               # the DRM player (needs libdrm)
    make BUILD=build GPU=cl               # with the OpenCL backend (libOpenCL.so.1, e.g. libmali)
    make BUILD=build GPU=vk GLSLC=glslc   # with the Vulkan backend (libvulkan, a GLSL compiler)

    build/vc2enc -s 1920x1080 --bpp 3 --threads 2 in.yuv out.vc2     # planar 4:2:2, 16-bit little-endian samples
    build/vc2dec out.vc2 back.yuv && build/psnr in.yuv back.yuv 1920 1080
    build/vc2bench --frames 300 --threads 3 out.vc2                  # see the head of tools/vc2bench.c for options
    scripts/verify.sh build in.yuv 1920 1080                         # against ffmpeg, about 12 minutes

A raw clip can be made with `ffmpeg -i some.mp4 -frames 4 -pix_fmt yuv422p10le -f rawvideo in.yuv`. ffmpeg
reads the `.vc2` files (`ffmpeg -i out.vc2 ...`; add `-fps_mode passthrough` when decoding to raw frames).
The test hosts below were an i7-7800X (two cores only, shared with other work, "the x86 host") and a
Firefly ROC-RK3588-RT (4 x A76 at 2.0 GHz, 4 x A55, Mali-G610, no fan, "the board").

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
- Speed: 73 ms per 1080p frame on the x86 host's two cores (13.6 fps).
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
- Read with care: the source was H.264, so PSNR is high; JPEG XS weights its subbands for
  the eye, not for PSNR, so PSNR under-rates it; SVT's "signs handling 2" adds 1.3 dB to XS at +42-68 %
  decoder cost, which would put the two level. A viewing test on camera content is still to be done, in
  particular for what tail truncation does to low-contrast texture.

Speed on the x86 host (i7-7800X, two cores at nice 19, shared with other work), `--threads 2`:

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
are compared with the same content written out in full, because of the ffmpeg bug above. The board's
output has the same checksums as the x86 host's for every clip measured below. Address and UB sanitizers are
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

**GPU**: at 1080p the CPU path is enough and is the default; the GPU path (section 6) is for UHD.

## 4. Player

`vc2play` decodes into NV20 scanout buffers (three, flipped on vblank). Firefly, HDMI-A-2 at 1920x1080@50,
3 threads on cores 4-6:

    frames 3000 in 60.0 s (50.01 fps), missed vblanks 0, decode avg 6.75 ms max 7.58 ms, cpu 19.17 ms/frame (0.96 cores)

The NV20 output is checked against the planar output in `vc2bench --nv20` (identical). The picture on the
screen itself was not looked at by a person during these runs.

## 5. VC-2 against JPEG XS

Same clips, same board. JPEG XS = SVT-JPEG-XS at its defaults; VC-2 = this repo, LeGall
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
| UHD frame rate on the board, 3 bpp | about 44 fps (Mali, 2 frames in flight; other clip) | CPU only: 38.5 fps (3 x A76), 46 fps (4 x A76); with the Mali: 52 fps (3 x A76), 60 fps (4 x A76) |
| Latency after the last byte, 1080p50 | whole frames only; not measured this way (UHD: 39-47 ms) | 6.3 ms whole frame (3 threads); 0.65 ms sub-frame |
| Encoder, 1080p 3 bpp on the x86 host's 2 cores | (SVT, not measured here) | 48-50 fps |
| Royalties | patent pool (Vectis) | none |

On the CPU alone UHD 2160p50 is not reached (26 ms per frame on three A76 cores at 3 bpp, 21.6 ms on four);
with the GPU path it is: section 6.

## 6. UHD: the last wavelet level on the GPU

Design (`lib/vc2_dec_gpu.h`, `lib/vc2_gpu.h`): with the GPU path on, the CPU threads do the entropy decoding,
the dequantisation and the deep wavelet levels as before, but write the four inputs of the last level (3/4
of the samples) into whole-frame 16-bit planes shared with the GPU. One fused kernel does the vertical and
horizontal lifting, rounding, offset and clip, and writes planar samples or packed NV20. It is launched in
blocks of band rows while the CPU is still decoding the rest of the frame, so nothing waits for the frame
to be complete. Same output, bit for bit. LeGall 5,3 and Deslauriers-Dubuc 9,7, 10-bit; anything else stays on
the CPU.

- **OpenCL (Mali, libmali)**: the planes come from the uncached dma-heap and are imported as dma-bufs (no
  copies, no cache cleaning by the driver); the NV20 output can be a scanout dma-buf, which the player uses.
- **Vulkan compute**: the same blocks and steps as a GLSL shader without 16-bit types; host-visible buffers,
  planar output (NV20 is then packed on the CPU). Checked on lavapipe (Mesa's software Vulkan) on the x86
  host and on a hardware driver, the Raspberry Pi 5's V3D 7.1 (Mesa v3dv 26.2): bit-identical there, but
  about 40 times slower than the CPU (below), so it is not a speed-up on that GPU.

Bit-exactness: on the board, 64 of 64 comparisons of the OpenCL path with the CPU path are identical (8
clips: 1080p and UHD, both wavelets, 2 and 3 bpp; 1, 3, 4, 7 threads, sub-frame, cached heap, NV20 into a
dma-buf and through a copy), and `GPU=cl scripts/verify.sh` passes 254 of 254 with 348 decodes through the
GPU. On lavapipe, `GPU=vk scripts/verify.sh` passes 254 of 254 with 348 decodes through Vulkan.

The board, UHD 4:2:2 10-bit, LeGall 5,3 (GPU driver threads on the A55 cores, `--gpu-cpus 0,1,2,3`):

| Flat out | 3 bpp: fps | decode | CPU per frame | GPU busy | 2 bpp: fps | decode |
|---|---:|---:|---:|---:|---:|---:|
| CPU, 3 x A76 | 39.0 | 25.6 ms | 72.8 ms | | 43.2 | 23.2 ms |
| CPU, 3 x A76, NV20 | 34.7 | 28.8 ms | 82.2 ms | | 38.0 | 26.3 ms |
| CPU, 4 x A76, NV20 | 40.9 | 24.4 ms | 91.5 ms | | 43.9 | 22.8 ms |
| GPU, 3 x A76, planar | 52.3 | 19.1 ms | 55.6 ms | 4.6 ms | 59.9 | 16.7 ms |
| GPU, 3 x A76, NV20 | 52.2 | 19.2 ms | 53.7 ms | 7.4 ms | 58.8 | 17.0 ms |
| GPU, 4 x A76, NV20 | 60.2 | 16.6 ms | 59.2 ms | 7.2 ms | 66.7 | 15.0 ms |

Paced at 50 fps for 60 s (3000 frames), NV20 output, latency from a frame's last byte to its last line;
"late" = frames that took longer than one frame period:

| | avg | max | late | CPU per frame | temperature, start to end |
|---|---:|---:|---:|---:|---|
| 3 x A76 + GPU, 3 bpp | 19.2 ms | 23.8 ms | 18 | 53.8 ms | 55 to 72 C |
| 3 x A76 + GPU, 2 bpp | 17.3 ms | 27.7 ms | 4 | 47.2 ms | 55 to 71 C |
| 4 x A76 + GPU, 3 bpp | 16.9 ms | 27.5 ms | 4 | 59.9 ms | 56 to 74 C |
| 4 x A76 + GPU, 2 bpp | 15.4 ms | 30.2 ms | 5 | 53.3 ms | 55 to 73 C |

On screen (`vc2play --gpu cl`, the UHD NV20 buffers scaled by the display controller to the screen's
1920x1080@50 mode, since the screen has 2160p only at 30 Hz), 3000 frames each:

| | missed vblanks | decode avg / max | CPU cores | end temperature |
|---|---:|---|---:|---:|
| 3 x A76 + GPU, 3 bpp | 6 | 19.1 / 20.1 ms | 2.69 | 71 C |
| 3 x A76 + GPU, 2 bpp | 0 | 17.2 / 18.7 ms | 2.36 | 69 C |
| 4 x A76 + GPU, 3 bpp | 0 | 16.9 / 18.3 ms | 3.01 | 72 C |

**Verdict.** UHD 2160p50 is reached on this board with the Mali doing the last level: with four A76 cores
at 3 bpp (17 ms average of the 20 ms, 0 missed vblanks in 3000; a handful of frames in a 60 s paced run
took longer than a period) and with three A76 cores at 2 bpp. Three A76 cores at 3 bpp are at the limit
(4 % headroom, 6 missed vblanks in 3000). The headroom is thin and the board is not thermally stable
without a fan: the temperature was still rising at 72-74 C after 60 s, and it throttles at 85 C. The fourth
core is the one otherwise kept free for the display thread; the player shared it without missed vblanks.

At 1080p the GPU path lowers the CPU time a little (one A76: 15.5 to 12.9 ms per frame; three A76 with
NV20: 19.4 to 14.7 ms) but not the latency (three threads: 6.2 to 7.4 ms; sub-frame: 0.66 to 1.16 ms, and
the first lines no longer come out early), so the CPU path stays the default there. For Deslauriers-Dubuc 9,7
at UHD the GPU path gives 40 fps on three A76 (CPU: 32): 50p is not reached with that wavelet.

    # the board: OpenCL
    make BUILD=build GPU=cl all player
    build/vc2bench --frames 150 --threads 4 --cpus 4,5,6,7 --gpu cl --gpu-cpus 0,1,2,3 --nv20 uhd.vc2
    build/vc2bench --frames 3000 --fps 50 --threads 4 --cpus 4,5,6,7 --gpu cl --gpu-cpus 0,1,2,3 --nv20 uhd.vc2
    sudo VC2_MODE=1920x1080@50 player/play.sh build uhd.vc2 --threads 4 --cpus 4,5,6,7 --gpu cl --gpu-cpus 0,1,2,3
    GPU=cl scripts/verify.sh build in.yuv 1920 1080
    # any host with Mesa: Vulkan on lavapipe
    make BUILD=build GPU=vk && GPU=vk scripts/verify.sh build in.yuv 1920 1080

Raspberry Pi 5 (4 x A76 at 2.4 GHz, active cooler, V3D 7.1 with Mesa v3dv 26.2), same content, 4:2:2 10-bit,
LeGall 5,3:

| | fps | decode | CPU per frame |
|---|---:|---:|---:|
| 1080p 3 bpp, CPU, 4 threads, flat out | 84.5 | 11.8 ms | 39.4 ms |
| 1080p 3 bpp, Vulkan (v3dv), 4 threads | 2.2 | 459 ms | 27.5 ms |
| UHD 3 bpp, CPU, 4 threads | 19.6 | 51.1 ms | 172 ms |
| UHD 2 bpp, CPU, 4 threads | 19.8 | 50.6 ms | 170 ms |
| UHD 3 bpp, Vulkan (v3dv), 4 threads | 0.6 | 1768 ms | 121 ms |

Paced 1080p50, 1500 frames, latency from the last byte to the last line:

| | avg | max | late | CPU cores at 50 fps |
|---|---:|---:|---:|---:|
| 3 bpp, 2 threads | 8.1 ms | 18.6 ms | 0 | 0.77 |
| 3 bpp, 3 threads | 8.9 ms | 15.7 ms | 0 | 1.26 |
| 2 bpp, 2 threads | 7.3 ms | 14.2 ms | 0 | 0.69 |

So the Pi 5 decodes 1080p50 on the CPU with two cores to spare; UHD 50p is out of reach there (about 20 fps).
On v3dv the output matches the CPU path in every mode tried (1080p and UHD; 1 and 3 threads, sub-frame).

Orange Pi 5 Plus (RK3588, A76 at 2.26 / 2.30 GHz, fan at full, Armbian vendor 6.1 with Mali kbase g25p0;
the OpenCL library was the board's libmali g13p0 loaded from a user directory, which the newer kernel
driver accepts; run as root because `/dev/dma_heap` is root-only on that image), same streams:

| UHD, 3000 frames paced at 50 fps | latency avg | max | late | CPU per frame |
|---|---:|---:|---:|---:|
| CPU only, 4 x A76, 2 bpp | 18.2 ms | 18.5 ms | 0 | 67.8 ms |
| CPU only, 4 x A76, 3 bpp | (49.2 fps, falls behind) | | 3000 | 75.4 ms |
| 3 x A76 + GPU, 3 bpp, NV20 | 18.6 ms | 19.5 ms | 0 | 51.9 ms |
| 4 x A76 + GPU, 3 bpp, NV20 | 16.4 ms | 17.0 ms | 0 | 59.4 ms |
| 3 x A76 + GPU, 2 bpp, NV20 | 16.8 ms | 19.9 ms | 0 | 46.0 ms |
| 4 x A76 + GPU, 2 bpp, NV20 | 15.1 ms | 20.4 ms | 1 | 54.9 ms |

Flat out with the GPU: 53.9 fps (3 x A76) and 61.3 fps (4 x A76) at 3 bpp; GPU time 6.3-6.6 ms per frame.
1080p 3 bpp on the CPU: 191 fps on 3 x A76 (5.2 ms), and paced 1080p50 on two cores at 8.4 ms (max 8.6),
0 late in 1500. The GPU output was identical to the CPU path (UHD, 2 and 3 bpp). With the faster clock and
the fan this board reaches UHD 50p at 3 bpp with three A76 cores and the GPU, and at 2 bpp on the CPU alone;
the SoC stayed at 51-56 C. Not tried on screen (no display was connected).

## 7. RTP (RFC 8450)

`lib/vc2_rtp.c` packetises an encoded frame (sequence header, transform parameters, fragments of whole
slices up to an MTU, 90 kHz timestamps, marker on the last packet) and receives it: a reorder window, slices
placed by position, lost slices replaced by those of the previous picture, and every packet's slices fed to
the decoder at once. `docs/RTP.md` has the mapping, the SDP and what is open. The encoder caps the slice size
for this (`--max-slice`, 0.06 dB at 3 bpp for 1200-byte packets).

Tested on the board's own loopback interface only (sender on one A76 core, receiver and decoder on another),
1080p50 at 3 bpp, 910 packets per frame spread over the frame period, 300 frames:

| Case | lost packets | concealed slices | first packet to last line, avg / max | last packet to last line, avg / max | PSNR Y against a clean decode (20 frames) |
|---|---:|---:|---|---|---:|
| clean | 0 | 0 | 19.8 / 20.1 ms | 0.37 / 0.38 ms | identical |
| 5 % of packets swapped with the next, window 8 | 0 | 0 | 19.8 / 20.1 ms | 0.37 / 0.39 ms | identical |
| the same, no window | 13011 (counted late) | 14598 | 19.8 / 20.1 ms | 0.37 / 0.39 ms | 34.2 dB |
| 0.1 % loss | 288 | 321 | 19.6 / 20.2 ms | 0.37 / 0.52 ms | 50.6 dB |
| 0.5 % loss | 1395 | 1560 | 18.7 / 20.2 ms | 0.37 / 0.38 ms | 42.7 dB |
| 2 % loss and 2 % swapped | 5400 | 6037 | 17.2 / 20.3 ms | 0.37 / 0.60 ms | 37.4 dB |

So the end-to-end latency on loopback is the time the sender takes to send the frame plus 0.37 ms: with
the packets spread over the whole period that is 19.8 ms from the first packet, with a tenth of the period
16.2 ms (then the decoder, at 16.3 ms of work per frame, sets the pace). No frame was dropped in any case.
The test clip repeats 4 frames, so "the previous picture" is a poor match every fourth frame; real video
conceals better. The receiver needs a socket buffer larger than the usual 208 kB default limit
(`vc2recv` asks for 8 MB, which needs CAP_NET_ADMIN or a raised `net.core.rmem_max`), or packets are lost
whenever the decoder is busy for more than 0.7 ms.

    build/vc2enc --bpp 3 --max-slice 1168 in.yuv rtp.vc2
    sudo taskset -c 5 build/vc2recv --port 5870 --frames 300 --out rx.yuv --out-skip 200 --out-frames 20 &
    taskset -c 6 build/vc2send --port 5870 --frames 300 [--loss 5] [--reorder 50] [--spread 0.1] rtp.vc2
    build/vc2send --sdp rtp.vc2

## 8. Open

- A viewing test on camera content (tail truncation, slice-boundary visibility at low rates, concealment);
  VC-2 against JPEG XS by eye, not by PSNR.
- UHD: more headroom (the entropy decoding is now 2/3 of the CPU time), a fan, the Vulkan path on real
  hardware, a frame-in-flight mode if latency allows.
- RTP: a run on a real network, ST 2022-7, PTP timestamps, RTP into the player (see `docs/RTP.md`).
- Encoder: 3 bpp at 1080p50 is at the limit of two x86 cores; a 16-bit forward wavelet and a row-wise
  (sub-frame) mode are the next steps. Its latency today is one frame.
- Line-locked display (scan-out following the decoder by a fixed number of lines) for a true sub-frame
  glass-to-glass path; the player shows whole frames.
- 4:2:0 / 4:4:4, 8 / 12 bit, Daubechies 9,7 against a second reference decoder, interlaced.

## Reproducing the board measurements

    make BUILD=build all player
    b=build/vc2bench; f=k1080_b3_w1.vc2                               # vc2enc --bpp 3 on a 1080p clip
    taskset -c 4 $b --frames 150 $f                                   # one A76
    $b --frames 300 --threads 3 --cpus 4,5,6 [--nv20] $f              # three A76
    $b --frames 250 --fps 50 --threads 3 --cpus 4,5,6 $f              # paced: latency of whole-frame decoding
    $b --frames 250 --fps 50 --subframe 68 --cpus 4 $f                # paced, fed slice row by slice row
    $b --frames 100 --threads 7 --cpus 4,5,6,0,1,2,3 --weights 1,1,1,0.18,0.18,0.18,0.18 uhd.vc2
    VC2_REF_ENTROPY=1 taskset -c 4 $b --frames 60 $f                  # bit-serial entropy decoder
    VC2_NO16=1 taskset -c 4 $b --frames 60 $f                         # everything in 32 bits
    sudo player/play.sh build $f --frames 3000 --threads 3 --cpus 4,5,6   # stops and restarts lightdm

The player's defaults (connector 224, CRTC 93) are this board's HDMI-A-2; pass `--conn` / `--crtc` elsewhere.
The board throttles at 85 C after about a minute of full load: keep runs short and watch
`/sys/class/thermal/thermal_zone0/temp`.
