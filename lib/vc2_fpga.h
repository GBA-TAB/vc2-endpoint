// vc2-endpoint: the host side of a hardware VC-2 decoder (docs/FPGA.md section 9).
// The host parses the stream headers and sends the card, per picture, one descriptor followed by the slice bytes
// (the H2C stream). The card parses only the slices. Format version 0: a proposal, until the card side agrees.
#ifndef VC2_FPGA_H
#define VC2_FPGA_H
#include "vc2.h"

#define VC2_FPGA_DESC_LEN 64 // bytes, little endian (layout in docs/FPGA.md section 9)
#define VC2_FPGA_ALIGN 16    // every descriptor starts on a 16-byte (128-bit) boundary; slice bytes are padded with 0
#define VC2_FPGA_NEW_SEQ 1   // flags bit 0: first picture, or the sequence / fixed transform parameters changed (flush)

typedef struct {
    uint32_t picture_number, slice_bytes, seq_count, au_index;
    int flags, width, height, chroma, bit_depth, wavelet, depth, slices_x, slices_y, prefix_bytes, size_scaler;
    uint8_t qm[16]; // LL, then HL, LH, HH for levels 1..depth
} vc2_fpga_desc;

typedef struct {
    vc2_seq seq;
    vc2_pic last;   // the fixed parameters of the previous picture, to detect a change
    int have_seq, have_pic, new_seq;
    uint32_t seq_count, au_index;
} vc2_fpga_tx;

// Packs one access unit (as vc2_access_unit returns it) into a descriptor + slice bytes + padding at out.
// Returns the bytes written (a multiple of VC2_FPGA_ALIGN), 0 if the access unit is outside the hardware subset
// (*why says what) or malformed (*why = "malformed"). cap must be at least len + VC2_FPGA_DESC_LEN + VC2_FPGA_ALIGN.
size_t vc2_fpga_pack(vc2_fpga_tx *t, const uint8_t *au, size_t len, uint8_t *out, size_t cap, const char **why);
// NULL if the hardware subset covers seq + pic, else the reason.
const char *vc2_fpga_subset(const vc2_seq *s, const vc2_pic *p);

void vc2_fpga_desc_write(uint8_t *out, const vc2_fpga_desc *d);
int vc2_fpga_desc_read(const uint8_t *in, size_t len, vc2_fpga_desc *d); // 0, or -1 if not a version-0 descriptor

// The card model: rebuilds a VC-2 access unit from a descriptor and its slice bytes alone, so that the ordinary
// decoder shows the descriptor carries everything the card needs. Returns the access unit's length (cap must be
// at least slice_bytes + 256), 0 on error.
size_t vc2_fpga_model_au(const uint8_t *in, size_t len, uint8_t *au, size_t cap);
#endif
