// The last inverse-wavelet level on a GPU: what the decoder (vc2_dec_gpu.h) needs from a backend
// (vc2_cl.c: OpenCL on Mali, vc2_vk.c: Vulkan compute). Private to the library.
//
// The backend owns one coefficient buffer the CPU threads write (per component the four inputs of the last
// level, LL / HL / LH / HH, as planes of int16 with rows bw apart) and the planar output. The transform runs
// in blocks of `rb` band rows (2 rb picture lines), launched in pieces while the CPU is still decoding the
// rest of the frame: a launch may read `reach` band rows above and below its blocks, nothing else.
#ifndef VC2_GPU_H
#define VC2_GPU_H
#include "vc2.h"

typedef struct {
    int wavelet, bit_depth, ncomp;
    int bw[3], bh;               // band size of the last level (the same height for all components)
    int w[3], h;                 // picture size of each component
    int threads;                 // threads that launch (indices 0 .. threads-1)
    const int *cpus; int ncpus;  // cores for the driver's own threads (0 = wherever the caller runs)
} vc2_gpu_cfg;

typedef struct {
    int16_t *band[3][4];         // where the CPU writes
    uint16_t *pix[3];            // planar output, valid after finish()
    ptrdiff_t pstride[3];
    int rb;                      // band rows per block
    int nv20;                    // the backend writes NV20 itself (else the decoder packs the planar output)
} vc2_gpu_bufs;

typedef struct {                 // where a frame goes
    int nv20;                    // 0 = the planar output; 1 = NV20 (4:2:2 10-bit): Y plane, then the CbCr plane
    int fd;                      // NV20: a dma-buf the GPU writes directly (>= 0), else the host pointers below (a copy)
    size_t y_off, c_off;         // plane offsets within the dma-buf
    uint8_t *y, *c;
    ptrdiff_t pitch;             // bytes per row, both planes
} vc2_gpu_target;

typedef struct vc2_gpu vc2_gpu;
typedef struct {
    const char *name;
    vc2_gpu *(*create)(const vc2_gpu_cfg *cfg, vc2_gpu_bufs *bufs);
    void (*destroy)(vc2_gpu *g);
    int (*begin)(vc2_gpu *g, const vc2_gpu_target *t);                  // a new frame
    void (*wrote)(vc2_gpu *g, const void *p, size_t bytes);             // the CPU has written this part of the coefficients (any thread)
    int (*launch)(vc2_gpu *g, int thread, int blk0, int blk1);          // transform blocks [blk0, blk1), all components
    int (*finish)(vc2_gpu *g, double *gpu_seconds);                     // wait for all launches; the output is complete
} vc2_gpu_api;

#ifdef VC2_OPENCL
extern const vc2_gpu_api vc2_cl_api;
#endif
#ifdef VC2_VULKAN
extern const vc2_gpu_api vc2_vk_api;
#endif
#endif
