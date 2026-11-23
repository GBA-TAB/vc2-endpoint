// vc2-endpoint: SMPTE ST 2042 (VC-2) HQ-profile encoder and decoder. Own code, no third-party sources.
#ifndef VC2_H
#define VC2_H
#include <stddef.h>
#include <stdint.h>

#define VC2_MAX_DEPTH 6
#define VC2_INPUT_PADDING 16 // readable bytes the caller keeps after the end of a stream buffer

enum { VC2_PC_SEQ_HEADER = 0x00, VC2_PC_END_SEQ = 0x10, VC2_PC_AUX = 0x20, VC2_PC_PADDING = 0x30, VC2_PC_PICTURE_HQ = 0xE8 };
enum { VC2_WL_DD9_7 = 0, VC2_WL_LEGALL5_3 = 1, VC2_WL_DD13_7 = 2, VC2_WL_HAAR0 = 3, VC2_WL_HAAR1 = 4, VC2_WL_DAUB9_7 = 6 };

typedef struct {
    int major, minor, profile, level, base_format;
    int width, height;
    int chroma;     // 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0
    int interlaced, top_field_first;
    int fr_num, fr_den;
    int bit_depth;
    int picture_coding_mode;
} vc2_seq;

typedef struct {
    uint32_t picture_number;
    int wavelet, depth, slices_x, slices_y, prefix_bytes, size_scaler;
    uint8_t qm[VC2_MAX_DEPTH + 1][4]; // [level][orientation]: level 0 = LL only, levels 1..depth = HL, LH, HH (1..3)
    size_t slices_offset;             // byte offset of the first slice within the picture data unit payload
} vc2_pic;

// --- stream helpers -------------------------------------------------------------------------------------
// Finds the data unit at buf (must start with a parse info header). Returns its total length (header included),
// 0 if incomplete or not a parse info. *code = parse code.
size_t vc2_unit(const uint8_t *buf, size_t len, int *code);
// Length of one access unit at buf: everything up to and including the first picture (and a directly
// following end-of-sequence). 0 at the end.
size_t vc2_access_unit(const uint8_t *buf, size_t len);
int vc2_parse_seq(const uint8_t *payload, size_t len, vc2_seq *s);
int vc2_parse_pic(const uint8_t *payload, size_t len, const vc2_seq *s, vc2_pic *p);

// --- decoder --------------------------------------------------------------------------------------------
typedef struct vc2_dec vc2_dec;
typedef struct {
    int width, height, chroma, bit_depth;
    uint16_t *plane[3];  // samples LSB-aligned
    ptrdiff_t stride[3]; // in samples
} vc2_frame;

// threads: worker threads in total (1 = caller only). cpus: optional list of `threads` cpu ids to pin to (the
// first is for the caller and is not applied), or NULL.
vc2_dec *vc2_dec_new(int threads, const int *cpus);
void vc2_dec_free(vc2_dec *d);
// Decodes one access unit into the decoder's own frame (valid until the next call). 0 on success.
int vc2_dec_decode(vc2_dec *d, const uint8_t *au, size_t len, vc2_frame *out);
// The same in steps, for sub-frame latency: begin parses the headers (the slice data need not be there yet),
// feed says how many bytes of the access unit have arrived; it decodes every complete slice row and runs the
// inverse wavelet as far as the data allows. Returns the number of finished output lines (== height when done).
int vc2_dec_begin(vc2_dec *d, const uint8_t *au, size_t len_now, vc2_frame *out);
int vc2_dec_feed(vc2_dec *d, size_t len_now);
// Relative speed of each thread's core (default: all equal). A whole-frame decode gives every thread a band
// of slice rows in proportion, e.g. 1,1,1,0.27,0.27 for three A76 and two A55 cores of an RK3588.
void vc2_dec_set_weights(vc2_dec *d, const float *w);
const vc2_seq *vc2_dec_seq(const vc2_dec *d);
const vc2_pic *vc2_dec_pic(const vc2_dec *d);
// Per-phase time of the last frame in seconds (summed over calls): [0] entropy decode, [1] inverse wavelet.
void vc2_dec_times(const vc2_dec *d, double t[2]);
// Output in the display's NV20 layout instead (4:2:2 10-bit, 4 samples in 5 bytes, Y plane then CbCr plane).
// Call before decode with the target (pitch in bytes for both planes); NULL switches back to planar.
void vc2_dec_set_nv20(vc2_dec *d, uint8_t *y, uint8_t *cbcr, ptrdiff_t pitch);

// --- decoder: the last wavelet level on a GPU (optional backends: make GPU=cl | vk) ----------------------
// The CPU threads then do the entropy decoding and the deep levels, the GPU the last level (3/4 of the
// samples), launched in pieces while the CPU is still decoding the frame. Same output, bit for bit.
// Applies to 10-bit streams with LeGall 5,3 or Deslauriers-Dubuc 9,7; others stay on the CPU, with a note
// on stderr. With a GPU, a frame's planes are valid until the next begin / decode call, as before.
enum { VC2_GPU_OFF = 0, VC2_GPU_OPENCL = 1, VC2_GPU_VULKAN = 2 };
// Selects the path for the following frames. cpus: cores for the GPU driver's own threads (NULL = wherever
// the caller runs). Returns -1 if the backend is not compiled in.
int vc2_dec_set_gpu(vc2_dec *d, int backend, const int *cpus, int ncpus);
int vc2_dec_gpu_active(const vc2_dec *d);  // VC2_GPU_* in use for the last frame
double vc2_dec_gpu_time(const vc2_dec *d); // time the GPU was busy with the last frame, in seconds
// NV20 target as a dma-buf (a scanout buffer) the GPU writes directly; fd < 0 = none. Without it the GPU
// path fills the pointers of vc2_dec_set_nv20 through a copy. Used by the GPU path only: also call
// vc2_dec_set_nv20 with the buffer's mapping, so that the CPU path has its target.
void vc2_dec_set_nv20_dmabuf(vc2_dec *d, int fd, size_t y_offset, size_t cbcr_offset, ptrdiff_t pitch);

// --- encoder --------------------------------------------------------------------------------------------
typedef struct {
    int width, height, chroma, bit_depth;
    int fr_num, fr_den;
    int wavelet, depth;
    int slice_w, slice_h;     // luma samples per slice (the picture is padded to a multiple of 2^depth)
    double bpp;               // bits per pixel (luma sample count), per frame budget = w*h*bpp/8 bytes
    int rc_rows;              // rate control unit in slice rows; 0 = the whole frame
    double tail_lambda;       // tail truncation strength, 0 = only all-zero tails are dropped
    int fixed_q;              // > 0: constant quantiser index, no rate control (analysis)
    int no_truncate;          // write the zeros of a dropped tail as bits (for decoders that mishandle short components)
    int max_slice_bytes;      // > 0: no slice larger than this (RTP: a slice must fit one packet)
    int qm_custom;            // 0 = the default matrix of the wavelet (still written into the stream)
    uint8_t qm[VC2_MAX_DEPTH + 1][4];
    int threads;
    const int *cpus;
} vc2_enc_cfg;
typedef struct vc2_enc vc2_enc;
void vc2_enc_defaults(vc2_enc_cfg *c, int width, int height);
vc2_enc *vc2_enc_new(const vc2_enc_cfg *c);
void vc2_enc_free(vc2_enc *e);
size_t vc2_enc_max_size(const vc2_enc *e);
// The two finest forward levels on the GPU (VC2_GPU_OPENCL, make GPU=cl: Mali / libmali), the same bytes out.
// LeGall 5,3, depth 4, 10 bit, component widths padded to multiples of 64. 0 if active, -1 if not (CPU then).
int vc2_enc_set_gpu(vc2_enc *e, int backend);
double vc2_enc_gpu_ms(const vc2_enc *e); // GPU time of the last frame, -1 without the GPU
// Encodes one frame of planar samples (LSB-aligned) into out; returns the access unit's size.
size_t vc2_enc_encode(vc2_enc *e, const uint16_t *const plane[3], const ptrdiff_t stride[3], uint8_t *out);
// End of sequence (13 bytes), to close a file or a stream.
size_t vc2_enc_end(vc2_enc *e, uint8_t *out);
int vc2_enc_last_q(const vc2_enc *e); // mean quantiser index of the last frame, x100

// --- RTP (RFC 8450) -------------------------------------------------------------------------------------
#define VC2_RTP_OVERHEAD 32 // RTP header + payload header of a slice fragment: the largest slice is mtu - 32
typedef int (*vc2_rtp_send_fn)(void *ctx, const uint8_t *pkt, size_t len);
typedef struct vc2_rtp_tx vc2_rtp_tx;
// mtu = the largest packet handed to send (RTP header included, i.e. the UDP payload size)
vc2_rtp_tx *vc2_rtp_tx_new(int payload_type, uint32_t ssrc, int mtu);
void vc2_rtp_tx_free(vc2_rtp_tx *t);
// Sends one access unit (sequence header + HQ picture, as vc2_enc_encode writes it) as a sequence header
// packet, a transform parameters packet and slice fragments; the marker is set on the last packet.
// ts = RTP timestamp (90 kHz). Returns the number of packets, or -1 if a slice does not fit the mtu.
int vc2_rtp_tx_frame(vc2_rtp_tx *t, const uint8_t *au, size_t len, uint32_t ts, vc2_rtp_send_fn send, void *ctx);
// SDP media section for a stream (ST 2110-22 style attributes included); returns the length written.
int vc2_rtp_sdp(char *out, size_t cap, const char *dst_ip, int port, int payload_type, const vc2_seq *s, double bpp, const char *ts_refclk);

typedef struct vc2_rtp_rx vc2_rtp_rx;
typedef struct { uint64_t packets, late, lost, frames, concealed_slices, dropped_frames; } vc2_rtp_stats;
// The receiver feeds the decoder slice row by slice row as packets arrive. reorder = how many packets a
// missing one is waited for (0 = none). Slices lost are replaced by the same slices of the previous frame.
vc2_rtp_rx *vc2_rtp_rx_new(vc2_dec *d, int reorder);
void vc2_rtp_rx_free(vc2_rtp_rx *r);
// Returns 1 when this packet completed a frame (*out is valid until the next packet), 0 otherwise.
int vc2_rtp_rx_packet(vc2_rtp_rx *r, const uint8_t *pkt, size_t len, vc2_frame *out);
// Gives up waiting for missing packets and closes the open frame (call on a timeout). Returns like rx_packet.
int vc2_rtp_rx_flush(vc2_rtp_rx *r, vc2_frame *out);
int vc2_rtp_rx_lines(const vc2_rtp_rx *r); // finished lines of the frame in progress
const vc2_rtp_stats *vc2_rtp_rx_stats(const vc2_rtp_rx *r);
// Hook for ST 2022-7: merge the two legs before vc2_rtp_rx_packet; duplicates (same sequence number) are dropped here.
#endif
