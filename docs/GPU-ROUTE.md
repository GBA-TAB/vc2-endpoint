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
2. **CPU, 16-bit storage for the finest levels.** The proven bounds (FPGA.md section 6): the finest level's
   bands fit 13 bits, the next 15-16; only the two coarsest need 17. Storing levels 3-4 (three quarters of the
   samples) in 16 bits halves the wavelet's and the counter's memory traffic and doubles the NEON lanes.
3. **Mali, the encoder's front end (OpenCL, libmali).** Forward wavelet and per-slice counting on the GPU:
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
