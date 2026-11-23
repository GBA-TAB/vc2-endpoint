# GPU route: Mali on RK3588 boards, then CUDA for density

Status 2026-11-22: a plan with the measurements it starts from. Nothing here is built yet except where it says so.

## What limits this codec family

VC-2 (and JPEG XS, which uses the same LeGall 5,3 wavelet) does little arithmetic per byte. What sets the speed
is how often the coefficients cross the memory bus:

- **The encoder's forward wavelet on a Pi 5** took 18.7 ms per 1080p frame with 1, 2 or 4 threads: memory-bound.
  Running the vertical lifting steps line by line behind the horizontal pass (rows still in cache) brought it to
  12.5 ms, same output (README section 2).
- **The decoder's last wavelet level on Mali** gains because the GPU does it from planes in uncached memory that
  the CPU writes once; the launch cost (about 0.1 ms of GPU time each) decides the block size (README section 6).
- **On a server GPU** the same holds at a larger scale: streams per GPU are set by bytes moved per frame, and,
  once the GPU is fast enough, by PCIe.

So every step below is judged by bytes moved per frame first, then by launches, then by arithmetic.

## Part 1: embedded (RK3588, libmali)

The RK3588 boards are the endpoints. libmali (the vendor's OpenCL, Valhall g13p0 here) is the driver that works
today on the vendor kernels, and it carries the decoder's UHD path.

### Where the encoder stands (1080p 3 bpp, LeGall 5,3, after the line-by-line wavelet)

| | wavelet | rate control (counting) | writing | fps, 4 threads |
|---|---:|---:|---:|---:|
| Pi 5, 1 thread | 12.5 ms | 23.8 ms | 11.8 ms | 39.7 |
| RK3588 A76 at 2.0 GHz, 1 thread | 11.4 ms | 23.4 ms | 13.9 ms | 54.3 |

The RK3588 encodes 1080p50 on its four A76 cores, with nothing left for anything else. UHD50 needs four times
that.

### Steps, in order

1. **CPU, counting in one pass** (all platforms). The rate control counts the whole picture at two or three
   quantisers, one pass each. An exp-Golomb length is `2 * floor(log2(|v| + 1)) + 1 (+1 sign)`, computable
   directly from the value, so one pass can produce each slice's bits for a window of quantisers around the
   last frame's (say q-1 .. q+2) while the coefficients are read once. Expected: counting from ~24 ms to
   ~10 ms on one A76. Byte-identical output is the check (the rate control sees the same numbers).
   **Measured first (2026-11-22, Pi 5, one thread):** one counting pass is 9.8 ms, of which about 8.8 ms is the
   per-coefficient loop (vectorised; about 5 cycles per coefficient) and 0.4 ms the truncation search. At
   about 1.6 GB/s it is arithmetic-bound, not memory-bound, so reading the coefficients once for several
   quantisers saves less than hoped: what costs is the work per coefficient and quantiser. An integer
   multiply-shift quantiser instead of the float reciprocal (fewer conversions) gave -2 ms on the A76 and +2 ms
   on AVX2, so it was set aside. The levers left: fewer lanes of work (16 bits, step 2), fewer counting passes
   (a better starting quantiser), or the count on the GPU (step 3).
   **The window, measured (2026-11-23):** counting q - 1 in the same pass as q (identical output, 72 of 72
   cases) is slower, not faster: Pi 5 one thread 52.6 -> 57.4 ms per frame, Firefly 53.6 -> 60.0 ms; four threads
   about 4 % fewer frames. The refinement counts q - 1 only for the slice rows its budget can pay for, so counting
   it for every row costs more than the pass it saves. Dropped.
   **The quantisation matrix, measured:** the matrix matched to the synthesis L2 weights ({7; 4,4,0; 4,4,0;
   4,4,1; 6,6,4} against the default {4; 2,2,0; 4,4,2; 5,5,3; 7,7,5}) gives -0.01 to -0.08 dB Y and +-0.02 dB on
   the average over two natural clips (1080p, UHD) at 1.5-4 bpp; coarser finest-level variants lose up to 0.7 dB.
   The default stays.
2. **CPU, 16-bit storage for the finest levels.** Done 2026-11-23 (README section 2): Pi 5 1080p50 on four
   cores (39.5 -> 55 fps), Firefly 55.8 -> 65.3 fps, byte-identical. The counting still widens to 32 bits after
   the load; 16-bit counting lanes are the next step there. Done the same day: exact (constants checked at
   start-up), about 10 % off the counting (one vector multiply pipe on the A76, seven multiplies per
   coefficient); Firefly 66.7 fps with four threads. The proven bounds (FPGA.md section 6): the finest level's
   bands fit 13 bits, the next 15-16; only the two coarsest need 17. Storing levels 3-4 (three quarters of the
   samples) in 16 bits halves the wavelet's and the counter's memory traffic and doubles the NEON lanes.
3. **Mali, the encoder's front end (OpenCL, libmali).** First part done 2026-11-23: the two 16-bit levels on the
   GPU, bit-exact, 1.4-1.9 ms per 1080p frame on the Mali-G610 against about 8 ms on one A76; the encoder's CPU
   time falls by 5-6 ms per frame, the frame rate not yet (the CPU waits for the GPU). Then the counting on the
   GPU, the same day: CPU per 1080p frame 47 -> 26 ms, 1080p50 on two A76 cores and the Mali (54 fps), UHD
   15.7 -> 19.3 fps on four, byte-identical.
   **Tried and dropped (2026-11-23), both byte-identical:**
   - *Running the GPU one frame ahead* (two buffer sets, the counting on a second queue): 1080p four threads
     64.5 -> 69.2 fps, two threads 53.0 -> 54.7, UHD four threads 18.5 -> 19.6, two threads no change. Most of
     the GPU time is the counting, which sits in the rate control's path; only the wavelet overlaps. About 5 %
     for one more frame of latency.
   - *Coalesced counting* (eight neighbouring values per work-item, sums per slice in local memory): slower on
     the Mali-G610 (GPU 5.4 -> 6.3 ms per 1080p frame); its caches already serve the per-segment loop.
   What is left on the GPU side: fewer counting round trips per frame (several quantisers in one launch, which
   on the GPU costs nothing extra in rows, unlike on the CPU). Forward wavelet and per-slice counting on the GPU:
   - the input planes from the dma-heap (a capture or decoder buffer can be imported as a dma-buf, no copy);
   - the wavelet as the decoder's kernel in reverse (blocks of band rows, one launch per band of rows for all
     components);
   - counting: one work-group per slice, the bits for the quantiser window as above, reduced in local memory;
   - the CPU keeps the rate-control decision (tiny) and the bit writer (per slice, threads), reading the
     coefficients once.
   - **To measure first:** whether the Mali's share of the memory bandwidth is larger than the A76 cluster's.
     If the wavelet stays memory-bound on the GPU too, the gain is in freeing the CPU, not in time.
4. **The open stack as a comparison:** Panthor (kernel) + PanVK (Vulkan) / Rusticl (OpenCL) in Mesa run the
   Mali-G610 on mainline kernels (6.10 and later). The decoder's Vulkan shader runs unchanged there. Worth one
   measurement on a spare SD card against libmali: if PanVK is close, the boards could leave the vendor kernel.
   The Pi 5's V3D is not a candidate (measured: about 40 times slower than its CPU for this work, and its only
   memory type is uncached for the CPU).

### Decoder on the boards

Already done: 1080p50 on two A76 cores; UHD50 at 3 bpp with three A76 + Mali (last level on the GPU), 0 late in
3000 frames. Next only if needed: the last two levels on the GPU (the bounds allow 16 bits there too).

## Part 2: CUDA (density on Ada and Blackwell)

NVENC / NVDEC have neither VC-2 nor JPEG XS, so this is compute. The aim is streams per GPU at a fixed latency
(one frame or less), on cards such as:

| GPU | memory bandwidth | PCIe |
|---|---:|---|
| T4 (Turing) | 320 GB/s | 3.0 x16 |
| RTX 3070 (Ampere) | 448 GB/s | 4.0 x16 |
| L4 (Ada) | 300 GB/s | 4.0 x16 |
| L40S (Ada) | 864 GB/s | 4.0 x16 |
| RTX PRO 6000 (Blackwell) | 1.8 TB/s | 5.0 x16 |

### The three limits

1. **Device memory.** A decoder that reads the coefficients once, keeps a slice tile's levels in shared memory
   and writes the picture once moves roughly 10-20 MB per 1080p frame: 0.5-1 GB/s per 1080p50 stream. That
   would allow hundreds of streams on an L4 and over a thousand on a Blackwell card, before the next limits.
2. **Entropy decoding / writing.** VC-2's slices are independent and byte-aligned: a 1080p frame at 128x16
   slices has about 1000 slices, 3000 components. One component is a serial bit stream; batching many streams
   per launch gives the GPU enough independent work. The table-driven decode (12-bit lookups) suits a thread
   per component; a warp-per-slice variant can share the table in shared memory.
3. **PCIe, the real ceiling.** Uncompressed 1080p50 4:2:2 10-bit is about 260 MB/s per stream: roughly 90
   streams each way on PCIe 4 x16, 190 on PCIe 5, if pictures cross the bus. Density beyond that needs the
   uncompressed video to stay off the host: the NIC writes into and reads from GPU memory (GPUDirect RDMA:
   Rivermax, or DOCA GPUNetIO, with a ConnectX), so only compressed streams touch the host, or none.

### Steps

1. **Decoder kernel, measured on the cards at hand (T4, RTX 3070):**
   - entropy decode per component, batched over N streams per launch (CUDA graphs for the launch cost);
   - dequantisation fused into it (the quantiser per slice is in the slice header);
   - inverse wavelet in 16 bits for the finest levels, 32 for the coarse ones (FPGA.md section 6), per tile
     with halos in shared memory, or level by level through L2 if the halos cost more;
   - output NV20 or v210 in device memory, a copy to the host only for the check;
   - bit-exact against the C decoder on the `scripts/verify.sh` set; then streams per GPU at 1080p50 and UHD50
     with latency, scaled by bandwidth and SMs to Ada and Blackwell (no such card in the lab yet).
2. **Encoder kernel:** forward wavelet per tile, counting for a quantiser window per slice (as in part 1), the
   rate control on the GPU (a per-frame search over the slice tables), the writer with one thread per
   component (or a warp with a prefix sum of code lengths). Byte-identical to the C encoder.
3. **The NIC path:** ST 2110-20 in and out of GPU memory through a ConnectX (GPUDirect RDMA), compressed
   streams (RFC 8450) through the host stack or the same path.

### Code layout

- `lib/vc2_cuda.cu` behind the existing GPU hook (`vc2_dec_set_gpu`, a new backend), plus a batch API for many
  streams (`vc2_cuda_batch_*`), since density is the point.
- The C decoder stays the reference; `scripts/verify.sh GPU=cuda` as for OpenCL and Vulkan.
- VC-2 is royalty-free; nothing in this repository depends on a patent licence.

## JPEG XS

What carries over: the 5,3 wavelet kernels (forward and inverse, the 16-bit bounds differ with XS's
asymmetric decomposition and must be recomputed), the tile scheme, batching, the NIC path, the RTP receiver's
concealment by slice. What does not: XS's entropy coding (bit-plane counts per group of four coefficients,
vertical prediction between precinct lines) and its rate control (budget per precinct, truncation positions).
Commercial GPU JPEG XS SDKs exist, so the approach is proven. JPEG XS is under a patent pool licence: that work
is kept apart from this repository.
