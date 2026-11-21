// vc2_xdma: the host side of the VC-2 decoder on the M.2 XC7A200T card, over the Xilinx xdma driver.
// SPDX-License-Identifier: MIT
//
// The card (interface note 2026-11-21T2116Z-pmod-b3-xdma-interface.md):
//   H2C 0x9000_0000  the decoder's input: access units of the H2C format v0 (vc2-endpoint docs/FPGA.md section 9)
//   C2H 0xA000_0000  the decoder's output: the pictures as NV20 with interleaved lines (Y line 0, CbCr line 0,
//                    Y line 1, ...): in memory NV20 with both planes at pitch 2 x line, CbCr at offset 1 line
//   user BAR 0x60000 the status block (struct vc2x_status)
// Both DMA windows are byte streams: every transfer starts at the window base, the order of the data is all that
// counts, and a transfer waits (back-pressure) rather than fails while the decoder is busy or a line isn't there.
//
// Threads: one thread may send while another receives (separate file descriptors). Pin the receiving thread to
// an A76 core (CPUs 4-7 on the RK3588). Buffers: any user memory, best page-aligned and reused; the xdma driver
// pins the pages for each call, so a receive can go straight into a scanout (dumb) buffer.
#ifndef VC2_XDMA_H
#define VC2_XDMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VC2X_H2C_BASE     0x90000000ull
#define VC2X_C2H_BASE     0xA0000000ull
#define VC2X_STATUS_BASE  0x60000u      // in the user BAR
#define VC2X_ID           0x56433244u   // "VC2D"

// status flags (struct vc2x_status.flags)
#define VC2X_IDLE         (1u << 0)
#define VC2X_ERR_MAGIC    (1u << 1)     // the H2C stream is lost: vc2x_reset() needed
#define VC2X_ERR_FORMAT   (1u << 2)     // a picture outside the subset arrived
#define VC2X_ERR_OVERRUN  (1u << 3)     // the slices ran past the descriptor's slice bytes
#define VC2X_ERR_GEOM     (1u << 4)     // slices larger than the card's buffers (the picture is wrong)
#define VC2X_SATURATED    (1u << 5)     // a value saturated (sticky until reset): outside the proven bounds
#define VC2X_ERRORS       (VC2X_ERR_MAGIC | VC2X_ERR_FORMAT | VC2X_ERR_OVERRUN | VC2X_ERR_GEOM)

struct vc2x_status {
    uint32_t id;               // VC2X_ID when the decoder partition is loaded, else 0 or 0xFFFFFFFF
    uint32_t version;          // RTL version
    uint32_t flags;            // VC2X_IDLE, VC2X_ERR_*, VC2X_SATURATED
    uint32_t picture_count;    // the descriptor's picture count (byte 52) of the picture being output
    uint32_t lines_done;       // luma lines of that picture already handed to C2H
    uint32_t pictures_done;    // pictures completed since reset
};

typedef struct vc2x vc2x;

// "/dev/xdma0": the card (opens <dev>_h2c_0, <dev>_c2h_0, <dev>_user).
// "model:<arg>": the software stand-in (vc2x_model_open, in a separate file; without it: NULL, errno ENOSYS).
// Returns NULL with errno set on failure.
vc2x *vc2x_open(const char *dev);
void vc2x_close(vc2x *x);

// 0 if the decoder answers with VC2X_ID, else -1 (errno ENODEV): e.g. while the DFX partition is reloaded.
int vc2x_check(vc2x *x);

// Resets the decoder (drops what it holds, clears the errors) and waits until it is idle (up to 1 s).
int vc2x_reset(vc2x *x);

// H2C: one or more access units of format v0, len a multiple of 16. Blocks until the DMA is done. 0 or -1/errno.
int vc2x_send(vc2x *x, const void *buf, size_t len);

// C2H: the next len bytes of the picture stream, len a multiple of 2 lines (2 x width x 10 / 8 bytes). A picture
// is vc2x_picture_bytes() long; read it in one call or in pieces. Blocks until the bytes are there. 0 or -1/errno.
int vc2x_recv(vc2x *x, void *buf, size_t len);

// The status block. 0 or -1/errno.
int vc2x_status(vc2x *x, struct vc2x_status *s);

// Sizes of the output: one line of one plane (width x 10 / 8), and a whole picture (2 planes x height lines).
static inline size_t vc2x_line_bytes(unsigned width) { return (size_t)width * 10 / 8; }
static inline size_t vc2x_picture_bytes(unsigned width, unsigned height) { return 2 * vc2x_line_bytes(width) * height; }

// --- for other backends (the model) ------------------------------------------------------------------------
struct vc2x_ops {
    int (*check)(vc2x *x);
    int (*reset)(vc2x *x);
    int (*send)(vc2x *x, const void *buf, size_t len);
    int (*recv)(vc2x *x, void *buf, size_t len);
    int (*status)(vc2x *x, struct vc2x_status *s);
    void (*close)(vc2x *x);
};
struct vc2x {
    const struct vc2x_ops *ops;
    void *priv;
};
// Implemented by the model backend (vc2_xdma_model.c). A weak default in vc2_xdma.c returns NULL / ENOSYS.
vc2x *vc2x_model_open(const char *arg);

#ifdef __cplusplus
}
#endif
#endif
