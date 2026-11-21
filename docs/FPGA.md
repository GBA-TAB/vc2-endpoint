# A hardware VC-2 decoder: the subset, the arithmetic, the reference vectors

For a decoder in an FPGA (first target: an Artix-7 200T on PCIe Gen2 x4 next to an RK3588, 1080p50, then
2160p50). This file says what the hardware must accept, exactly how it computes, and how to check it against
the C code. The arithmetic below is what `lib/` does; where this file and the code disagree, the code wins
(and this file gets fixed).

## 1. The subset

| | Must accept | Later / optional | Reject (flag an error, skip the picture) |
|---|---|---|---|
| Profile | HQ (picture parse code 0xE8) | | LD, core-syntax pictures |
| Version | 2 and 3 (in 3 both asymmetric-transform flags are 0) | | 3 with asymmetric transforms |
| Scan | progressive frames (picture_coding_mode 0) | | fields |
| Sampling | 4:2:2, 10 bit | 4:4:4, 4:2:0; 8 and 12 bit | |
| Size | 1920x1080, 3840x2160 | any width/height | |
| Wavelet | 1 = LeGall 5,3 | 0 = Deslauriers-Dubuc 9,7 (one more lifting kernel) | others |
| Depth | 4 | 3 (only a counter) | > 4 |
| Slices | uniform: every level's band width divisible by slices_x and height by slices_y (true for our defaults: 128x16 luma slices at 1080p and 2160p) | non-uniform (integer-divided edges) | |
| prefix_bytes | any (skipped) | | |
| size_scaler | any ≥ 1 | | |
| Quant matrix | default or custom, per picture | | |

Fixed within a sequence (a change = flush and reconfigure): the sequence header, wavelet, depth,
slices_x/slices_y, prefix_bytes. Free per picture: size_scaler and the quantisation matrix (13 entries at
depth 4). Free per slice: the quantisation index and the three component lengths.

At 1080p, depth 4, 4:2:2, our default slices are 15x68: the luma slice is 8x1 coefficients in the LL band
and level 1, 16x2 at level 2, 32x4 at level 3, 64x8 at level 4 (= 128x16 picture samples); chroma has half
the width. 2160p: 30x135 slices, the same sizes per slice.

## 2. Stream structure (as `lib/vc2_common.c` parses it)

- **Data units:** each starts with a 13-byte parse info: "BBCD", the parse code, the next-unit offset and
  the previous-unit offset (32-bit big endian). 0x00 sequence header, 0x10 end of sequence, 0x20/0x30
  auxiliary/padding (skip), 0xE8 HQ picture. A next offset of 0 means: find the next "BBCD".
- **Header fields** are interleaved exp-Golomb unsigned values (below), read MSB first, byte-aligned
  at the end of each header.
- **The HQ picture payload:** picture number (32 bits), the transform parameters (wavelet, depth,
  [v3: two flags], slices_x, slices_y, prefix_bytes, size_scaler, a custom-matrix flag and, if set, the
  matrix: LL, then HL, LH, HH for levels 1..depth), byte alignment, then the slices.
- **A slice:** prefix_bytes (ignored), 1 byte quantisation index, then for Y, Cb, Cr in turn: 1 byte length
  L, then L x size_scaler bytes of coefficient data. Slices follow each other in raster order with no gaps.
  The slice's coefficients per component, in this order: the LL block (level 0), then for level 1..depth
  the HL, LH and HH blocks, each in raster order. Block (sx, sy) of a band of width bw, height bh covers
  x in [bw*sx/slices_x, bw*(sx+1)/slices_x), y likewise (integer division).

## 3. Entropy decoding

Interleaved exp-Golomb, signed: read bit pairs; while the first bit of a pair is 0, the second is the next
value bit; a 1 ends the value. value = (1 followed by the value bits) - 1. If the value is nonzero, one more
bit is the sign (1 = negative). So 0 is a single 1 bit; 1 is "0 0 1" + sign, 2 is "0 1 1" + sign, 3 is
"0 0 0 0 1" + sign, and so on.

- **Past the end of a component's L x size_scaler bytes every bit reads as 1**, so the remaining
  coefficients are 0. Our encoder truncates slices on purpose: this is the normal case, not an error.
- A component's data may also end with unused bits; they are ignored.
- The three components of a slice and all slices are independent, so decoders run in parallel per
  component or per slice. Their start positions come from the length bytes alone (`slices.csv`).

## 4. Dequantisation

Per slice and subband: q = quantisation index of the slice - matrix[level][orientation], clamped to
[0, 115].

    factor(q) = round(4 * 2^(q/4)), computed exactly: the largest n with (2n - 1)^4 <= 2^(q + 12)
    offset(q) = 1 (q = 0), 2 (q = 1), (factor + 1) / 2 (q >= 2), then + 2   (the + 2 is the rounding of the >> 2)
    c = (|v| * factor(q) + offset(q)) >> 2, with v's sign; v = 0 gives 0

The two 116-entry tables are what `vc2_tables_init()` makes (`lib/vc2_common.c`); generate them, don't
retype them. factor(115) needs 31 bits. The indices that real streams use, and so the multiplier width
the hardware needs, come out of the vectors (section 8).

## 5. Inverse wavelet (LeGall 5,3), per level 1..depth

Inputs: LL (the coded LL at level 1, else the previous level's output) and HL, LH, HH, all bw x bh.
Output: 2bw x 2bh.

1. **Vertical**, on whole rows, first on the columns of [LL | HL] and of [LH | HH] (low rows L = LL/HL,
   high rows H = LH/HH, row index i):
   - L[i] -= (H[i-1] + H[i] + 2) >> 2
   - H[i] += (L[i] + L[i+1] + 1) >> 1     (with the updated L)
   - Indices past the edge use the nearest edge row (H[-1] = H[0], L[bh] = L[bh-1]).
2. **Horizontal**, the same two steps along each of the 2bh rows, with the low half (LL/LH) as L and the high
   half (HL/HH) as H, the same edge rule.
3. **Interleave and scale:** output row 2i comes from the low rows, 2i+1 from the high rows; within a row,
   sample 2x from L[x], 2x+1 from H[x]; every output sample is (s + 1) >> 1.
4. **After the last level:** add 2^(bit_depth-1) (512), clip to [0, 2^bit_depth - 1], and crop the padding
   (1080 lines are coded as 1088 at depth 4).

All shifts are arithmetic (rounding towards minus infinity). The vertical-then-horizontal order and the
rounding are exactly `vc2_synth_strip` in `lib/vc2_lift.h`. The production decoder does the same per strip
of rows, and the reference dump per whole level.

Line-based hardware: level l's output row r needs its band rows up to about r/2 + 2. A slice row of the
deepest level feeds 16 picture lines, so ring buffers of a few slice rows per level are enough (see the
"cascade" in `lib/vc2_dec.c`). At 2160p that is roughly 0.7-1 MB in all.

## 6. Bit widths: the worst case, proven

Each coefficient is a separable linear filter of the picture (rounding aside), so its largest magnitude for any
legal picture is 512 x 2^k x L1(row filter) x L1(column filter) after k analysis levels (samples in [-512, 511],
the encoder's << 1 per level). `scripts/fpga-worst.py bounds` computes it. `fpga-worst.py picture` builds a picture
that reaches it: one patch per band and level, each pixel at full scale with the sign of the filter. The vectors
`worst_53_d4_q1_full` reach every bound within 0.1 %.

| Level (synthesis order) | HL / LH | HH | LL input to this level | Signed bits needed |
|---|---:|---:|---:|---|
| 1 (coarsest; LL = the coded LL band) | 38974 | 64440 | 23572 | HL/LH/HH 17, LL 16 |
| 2 | 19008 | 30976 | 11664 | bands 16, LL 15 |
| 3 | 8320 | 12800 | 5408 | bands 15, LL 14 |
| 4 (finest, the widest buffers) | 3072 | 4096 | 2304 | bands 13, LL 13 |

- **16 bits is not enough for every legal picture:** level 1 needs 17 bits. Natural content stays far below
  (section 8), but full-scale fine patterns (test signals, zone plates, a 1-pixel checkerboard) get there.
- **Level 4 needs 13 bits, not 12.** 12 bits saturates on a full-scale 1-pixel checkerboard (HH 4092) and on
  LL inputs above 2047.
- **Quantisation doesn't push values past these bounds** in our tests (`worstnoisy_53_d4_b3`: quantisers 36-44,
  every value below the lossless bound), but a hostile stream can carry anything: saturate at the chosen width
  and raise a flag, never wrap.
- **Quantisers:** rate-controlled streams of hard content use up to 44 (factor 8192, 14 bits). The factor
  table should cover all 116 entries.
- **Inside a lifting step** a sum of two neighbours needs one bit more than the band it reads.
- **The production decoder** runs only the last level (4) in 16 bits: its inputs are at most 4096 (bands) and
  2304 (LL), so it is exact for every legal picture (checked: vc2dump = vc2dec on the worst-case vectors).

## 7. Reference vectors: `tools/vc2dump`

`vc2dump in.vc2 outdir [frames]` is a plain whole-frame decoder: bit-serial, scalar, 32-bit. It writes per
frame, in outdir/fNNN/ (binary files are little endian):

| File | Content |
|---|---|
| params.txt | sequence and transform parameters, the matrix, the band and slice sizes per level |
| slices.csv | per slice: sy, sx, byte offset in the picture payload, qindex, the three lengths in bytes, the three coefficient counts |
| coef_q.bin | per slice in raster order, per component Y, Cb, Cr, the quantised values in stream order, int32 |
| coef_dq.bin | the same, dequantised, int32 |
| band_l\<L\>_\<LL/HL/LH/HH\>_c\<C\>.bin | whole subbands after dequantisation, int32, bw x bh |
| synth_l\<L\>_c\<C\>.bin | output of level L < depth, int32, 2bw x 2bh, padding included |
| picture.yuv | the final picture, planar uint16, cropped |
| ranges.txt | the largest magnitudes of this frame |

Its final picture equals the production decoder's (`vc2dec`). `scripts/fpga-vectors.sh check` tests that over
a few hundred streams: both encoders, wavelets 0-4, depths 1-4, several slice sizes and quantisers, slices
truncated or written out in full. `scripts/fpga-vectors.sh make` produces the vector set: the subset's
streams plus two outside it, two frames each, each packed as \<name\>.tar.xz with its stream.vc2.

## 8. Measured (2026-11-21, 1080p50 4:2:2 10-bit)

- **`check`:** 388 streams, vc2dump = vc2dec in all of them.
- **The vector set:** 11 streams, 2 frames each, 7-61 MB per stream packed (1.4 GB unpacked, 219 MB packed).
  - Natural content: a market scene, already compressed once. Streams from our encoder at 2, 3 and 5 bpp,
    at a fixed quantiser with every coefficient written, with slices capped at 1168 bytes, and with 64x32
    slices; one stream from ffmpeg's encoder.
  - Stress: ffmpeg testsrc2 with noise at 8 bpp and at the finest quantiser (q 1, nothing truncated).
  - Outside the subset: Deslauriers-Dubuc 9,7, and depth 3. At 1080p, depth 3 gives 15x67 slices, which
    are not uniform.
- **Largest magnitudes over the subset's streams** (all components; the stress streams set every maximum):

| | quantised | dequantised |
|---|---:|---:|
| LL | 10981 | 10996 |
| level 1 HL / LH / HH | 10733 / 11866 / 7894 | 10733 / 11907 / 9868 |
| level 2 | ≤ 5652 | ≤ 5696 |
| level 3 | ≤ 2905 | ≤ 2941 |
| level 4 | ≤ 1782 | ≤ 1782 |
| output of level 1 / 2 / 3 | | 5214 / 2778 / 1363 |

- **Quantisation indices used** (after the matrix): 0 to 30.
- **So, measured on these streams:** every coefficient and every level output fits 16-bit signed with more than a
  bit of margin. **Not for every legal picture:** see section 6 (level 1 needs 17 bits). Inside a lifting step the sums are about twice the outputs, which also fits 16 bits for these
  streams. The production decoder runs its last level in 16 bits for the same reason.
- **Not a proof:** a hostile stream can carry larger values, since a slice can code up to 2^30. The
  hardware should saturate at its width and raise a flag, rather than wrap.
- **Factor table:** at least the indices 0-30 (factors 4 to 724), better all 116 entries as a ROM.

### 2160p50 4:2:2 10-bit (2026-11-21)

- **The same 11 streams at 3840x2160,** default slices 30x135, 2 frames each, every one dumped and equal to vc2dec
  (5.4 GB unpacked, 831 MB packed).
- **Not uniform at 2160p:** 64x32 slices give 60x67 slices (135 rows at level 4 are not divisible by 67), so
  `ours_53_d4_b3_64x32` is outside the subset at this size.
- **Largest magnitudes over the subset's streams,** where they differ from 1080p:
  - dequantised coefficient: 12608 (level 1 LH; 1080p: 11907);
  - LL: 11061;
  - level 2: ≤ 9308 dequantised (HH);
  - output of level 1: 5369 (1080p: 5214).
- **Quantisation indices used:** 0 to 36 (ffmpeg's encoder goes to 36; factor(36) = 2048).
- **So, measured on these streams:** everything fits 16-bit signed, and they use the indices 0-36. The worst-case
  vectors (section 6) need 17 bits at level 1 and use quantisers up to 44.

## 9. Host interface: the H2C stream (format version 0, a proposal)

The host parses the parse info, the sequence header and the transform parameters (`lib/vc2_fpga.c`), and sends
the card, per picture, a 64-byte descriptor followed by the picture's slice bytes. The card parses only the slices.
The per-picture values travel with the picture, so no register write has to be timed against DMA progress.

**The descriptor** (little endian):

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | magic "VC2D" (the bytes 56 43 32 44) |
| 4 | 1 | version, 0 |
| 5 | 1 | descriptor length in 16-byte units, 4 |
| 6 | 2 | flags: bit 0 NEW_SEQ (first picture, or anything below that is fixed per sequence changed: flush) |
| 8 | 4 | picture number (from the stream) |
| 12 | 4 | slice bytes: the length of the slice data that follows, without the padding |
| 16 | 2 + 2 | width, height (luma samples) |
| 20 | 1 each | chroma (1 = 4:2:2), bit depth, wavelet, depth |
| 24 | 2 + 2 | slices_x, slices_y |
| 28 | 2 + 2 | prefix_bytes, size_scaler |
| 32 | 16 | quantisation matrix: LL, then HL, LH, HH for levels 1..depth (13 used at depth 4), the rest 0 |
| 48 | 4 | sequence count: +1 with every NEW_SEQ |
| 52 | 4 | picture count: +1 with every picture the host sends |
| 56 | 8 | reserved, 0 |

- **After the descriptor:** the slice bytes exactly as in the stream (from the first slice's prefix to the end of
  the picture), then zeros up to the next multiple of 16 bytes. The next descriptor follows directly.
- **Fixed per sequence** (NEW_SEQ when one changes): width, height, chroma, bit depth, wavelet, depth, slices_x,
  slices_y, prefix_bytes. A repeated identical sequence header is no change (ffmpeg repeats it before every
  picture). **Per picture:** size_scaler and the matrix.
- **The host only sends what the subset covers** (`vc2_fpga_subset`); anything else stays on the CPU decoder.
- **The host also walks the slice headers:** a picture whose slice lengths run past its end (damaged in transit)
  stays on the CPU decoder. On the card it would read into the next descriptor (ERR_OVERRUN, then a reset).

**`tools/vc2fpga in.vc2 [out.h2c [pictures]]`** packs a stream, writes the H2C bytes, and checks them with a card
model: it rebuilds a VC-2 stream from each descriptor and its slice bytes alone, and decodes it against the
original. Over both vector sets (1080p, 2160p): every picture identical, and exactly the streams outside the
subset refused (DD 9,7, depth 3, the non-uniform 60x67 slices at 2160p). `fpga-vectors.sh make` puts the H2C
bytes of the dumped frames in each stream's folder as `stream.h2c`.

## 10. The card's host library: `lib/vc2_xdma` (by pmod-b3) and its model

- **The card side** (pmod-b3's interface note 2026-11-21T2116Z):
  - H2C window 0x9000_0000 and C2H window 0xA000_0000, both byte streams (every transfer starts at the base);
  - the status block at user BAR 0x60000: ID "VC2D", version, flags, the picture count and luma lines of the
    picture being output, pictures done, reset;
  - the output is NV20 with the lines interleaved (Y line, CbCr line, ...), so in memory NV20 with both planes at
    pitch 2 x line and CbCr at offset 1 line. One C2H transfer per picture, no frame buffer on the card.
  - Checked on the RK3588 (rockchip-drm, user space): the driver accepts this layout as an NV20 framebuffer at
    1080p and 2160p, and its dumb buffers can be pinned by get_user_pages, so `vc2x_recv` can read straight into
    the scanout buffer.
- **`lib/vc2_xdma.c`** (pmod-b3) is the hardware backend over the xdma character devices: `vc2x_open("/dev/xdma0")`,
  `vc2x_send` / `vc2x_recv` (one thread each; the receiver best on an A76), `vc2x_status`, `vc2x_reset`.
- **`lib/vc2_xdma_model.c`** is `vc2x_open("model:[threads=N,cpus=a/b,queue=N]")`. It passes the H2C stream
  through the card model and the CPU decoder into interleaved NV20, with back-pressure, the status block, and the
  card's error flags, behaving as the RTL does (pmod-b3, 2026-11-21):
  - ERR_MAGIC, ERR_FORMAT, ERR_OVERRUN: the card stops at that descriptor and puts nothing out for it. The
    pictures before it come out complete, and it takes no data until `vc2x_reset`. On the card a send blocks
    until the xdma driver's 10 s timeout; the model fails it at once with EIO.
  - ERR_GEOM (more than 8 level-4 rows per slice row, e.g. 64x32 slices): a full-length picture with undefined
    content (the model's is correct), and the flag stays until reset.
- **`tools/vc2xbench [--dev D] [--frames N] [--fps F] [--piece N] [--send-cpu C] [--recv-cpu C] in.vc2`** sends a
  stream through either backend. It compares every picture with the CPU decoder and reports MB/s, pictures/s,
  latency to the first and the last lines, CPU per picture, and the status flags.
  - Through the model: all subset streams of both vector sets identical, and ERR_GEOM on 64x32 at 1080p.
  - 1080p50 paced: 200 of 200 pictures on caspar; 100 of 100 on the Orange Pi (model on A76 cores 5-6,
    receiver on 4).
  - The same command with `--dev /dev/xdma0` is the first test on the card.
