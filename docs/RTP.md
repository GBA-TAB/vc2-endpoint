# VC-2 HQ over RTP (RFC 8450): design note

Status: design only. Nothing here has been run on a network; the repo's tools are file-based. The packet
layout below is written from memory of RFC 8450: check it against the RFC before building on it.

## What RFC 8450 carries

RFC 8450 defines an RTP payload for VC-2 HQ profile streams. Every RTP packet carries one of:

- a **sequence header** (parse code 0x00), sent before the pictures it describes and repeated often enough
  for receivers that join late (we send it with every picture, 12 bytes at 1080p50);
- a **picture fragment** of an HQ picture (parse code 0xEC): either the transform parameters of the picture
  (wavelet, depth, slice counts, prefix bytes, size scaler, quantisation matrix), or a run of **whole
  slices** with the coordinates of the first one (slice x and y offset) and the slice count;
- an **end of sequence** (0x10), or padding / auxiliary data, which a receiver may ignore.

The RTP timestamp (90 kHz) is the same for all packets of a picture, the marker bit is set on the last
packet of a picture, and the payload header carries the picture number, so a receiver can assign every
slice to its picture and its place without parsing the packets before it.

Our files and library use the ordinary stream syntax (parse info + sequence header + HQ picture, parse code
0xE8). The sender splits a picture into fragments; the receiver can feed the decoder either the rebuilt
picture or, for sub-frame latency, the slice rows as they arrive (see below). Both ends keep the
transform parameters and slice bytes untouched, so the conversion is a re-framing, not a transcode.

## Consequences for the encoder

1. **A slice must fit one packet.** Fragments hold whole slices, so the largest slice (1 byte quantiser
   index + 3 length bytes + the three components) has to stay under the path MTU minus headers: about
   1400 bytes on a 1500-byte Ethernet path. Our rate control gives every slice the same quantiser and lets
   the size follow the content, so a busy slice can be several times the mean. For RTP the encoder needs
   a per-slice cap (raise the index of a slice until it fits). Not implemented yet. At 3 bpp a 128x16 slice
   is 768 bytes on average and a 64x16 slice 384 bytes, so 64x16 leaves the cap rarely active.
2. **Truncated slices are plain VC-2**, but some receivers get them wrong (ffmpeg's decoder reads one byte
   past a short component and can produce a stray coefficient). `--no-truncate` writes every coefficient
   and costs 5 dB at 3 bpp (it cannot go below about 2.1 bpp at all); use it only towards receivers known
   to need it.
3. **Constant bit rate per picture, or per group of slice rows.** `--rc-rows N` spends the budget per N
   slice rows instead of per picture (0.23 dB at 3 bpp for N = 1). With N = 1 the sender never has to hold
   more than one slice row before sending, and a ST 2110-21-style narrow sender is possible: packets leave at a
   constant rate while the picture is still being encoded.

## Sub-frame latency end to end

- Encoder: the forward wavelet of a slice row needs the lines of that row plus the filter reach below it
  (about 30 lines for LeGall 5,3 at depth 4, 60 for Deslauriers-Dubuc 9,7). With `--rc-rows 1` a slice row
  can be sent as soon as those lines are in. The encoder in this repo works on whole frames; the row-wise
  variant is the same code driven line by line (open).
- Network: slice rows arrive in order; with a constant packet rate a 1080p50 picture of 68 slice rows
  delivers one row every 0.29 ms.
- Decoder: `vc2_dec_begin` / `vc2_dec_feed` decode every complete slice row on arrival and return the number
  of finished picture lines. Output trails the input by the same filter reach. On the Firefly the last
  line is ready 0.65 ms after the last byte of a 1080p50 picture (measured, see the README).
- Display: a receiver that starts scan-out one filter reach (plus a safety margin) after the first slice
  row arrives never waits for the whole picture. The player in this repo shows whole frames; a
  line-locked output needs a genlocked display timing (PTP) and is open.

## Packet loss

Slices are independent in the bit stream, but not in the picture: a lost slice removes its coefficients at
every wavelet level, so the damage spreads by the filter reach around the slice (wider at the deep levels).
The decoder zeroes missing slices if told so (not implemented: the index of slices is built from the byte
stream today; an RTP receiver would pass slice positions explicitly). For ST 2022-7 the two legs merge per
packet before the decoder (the `dup-merge` crate of srt-2110-gateway does that for our other receivers).

## SDP

    m=video 5004 RTP/AVP 96
    a=rtpmap:96 vc2/90000
    a=fmtp:96 profile-level-id=...   ; RFC 8450 has no required parameters; the stream describes itself

For NMOS (IS-04) the flow is `video/vc2`; there is no ST 2110 part for VC-2, so it runs as an RTP stream
beside ST 2110-22 (JPEG XS), with the sender's timing (2110-21) declared as for -22 if a receiver needs it.

## Open

- per-slice size cap in the rate control;
- fragmenter / reassembler (sender and receiver side), with slice positions passed to the decoder;
- concealment of lost slices;
- any run on the 2110 network: needs a heads-up to its owner and the user's go.
