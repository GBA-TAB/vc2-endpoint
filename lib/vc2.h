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
// Encodes one frame of planar samples (LSB-aligned) into out; returns the access unit's size.
size_t vc2_enc_encode(vc2_enc *e, const uint16_t *const plane[3], const ptrdiff_t stride[3], uint8_t *out);
// End of sequence (13 bytes), to close a file or a stream.
size_t vc2_enc_end(vc2_enc *e, uint8_t *out);
int vc2_enc_last_q(const vc2_enc *e); // mean quantiser index of the last frame, x100
#endif
