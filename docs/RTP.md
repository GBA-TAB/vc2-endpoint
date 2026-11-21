# VC-2 HQ over RTP (RFC 8450)

Implemented in `lib/vc2_rtp.c`, with the tools `vc2send` and `vc2recv`. Tested on a loopback interface only
(see the README for the figures); it has not been run on a real network. Section numbers below refer to
RFC 8450.

## Packets (section 4)

Every packet carries one data unit, after the RTP header and a 4-byte payload header (extended sequence
number, flags, parse code). We send three kinds per picture:

1. the **sequence header** (parse code 0x00, figure 1), with every picture, so a receiver can join at any
   picture;
2. a **fragment with the transform parameters** (0xEC, no slices, figure 2): picture number, slice prefix
   bytes and size scaler in the payload header, then the coded transform parameters;
3. **fragments with whole slices** (0xEC, figure 3): the same header fields, the fragment length, the number
   of slices and the position of the first one (slice offset x / y), then the slices as they are in the
   stream. A packet takes as many whole slices as fit the MTU.

Header usage (sections 4.1, 4.2): the sequence number is 32 bits (low half in the RTP header, high half in
the payload header); the timestamp is the picture's sampling instant on a 90 kHz clock, the same on all
packets of a picture including its sequence header; the marker bit is set on the packet with the last slice.
I / F are 0 (progressive only). End of sequence, auxiliary data and padding packets are not sent and are
ignored on reception.

Files and the encoder use the ordinary stream form (version 2, HQ picture, parse code 0xE8). Fragments are
a version 3 feature (section 7.1), so the packetiser sends the sequence header as version 3 and writes the
transform parameters in their version 3 form (two extra flags, both 0); the slice bytes are not touched. The
receiver rebuilds a picture from the fragments for the decoder. Section 4.4 allows exactly this conversion.

## What the stream must satisfy (section 4.4)

A slice must fit a packet. `vc2enc --max-slice N` / `vc2_enc_cfg.max_slice_bytes` caps every slice (a
slice over the cap gets a coarser quantiser); use MTU minus 32 (12 RTP + 4 + 16 payload header). At 3 bpp
with 128x16 slices and a 1200-byte packet the cap costs 0.06 dB. `vc2_rtp_tx_frame` fails if a slice does
not fit. Prefix bytes and size scaler stay far below the 16-bit limit.

## Receiver

- **Reorder window**: packets ahead of the expected sequence number are held; a missing packet is given up
  when a packet more than `reorder` (default 8) sequence numbers ahead arrives, or on
  `vc2_rtp_rx_flush` (the caller's timeout). In-order streams are not delayed.
- **Loss**: slices are placed by their position. Missing slices are replaced by the same slices of the
  previous picture (their coefficients, re-padded if the size scaler changed), or by empty slices if there
  is no previous picture. Because slices overlap in the wavelet domain, the replaced area blends into its
  surroundings. A lost transform parameters packet is rebuilt from the previous picture's and the two fields
  every fragment carries (section 4.2 allows assuming the parameters unchanged). If the last packets of a
  picture are lost, the picture is closed by the first packet of the next one or by the flush.
- **Sub-frame**: every packet's slices go to the cascade decoder at once (`vc2_dec_feed`), so picture
  lines are ready about 30 lines (LeGall 5,3) behind the slice row received.
- ST 2022-7: not implemented. The place for it is in front of `vc2_rtp_rx_packet`: packets of both legs can
  be passed in as they come, since a sequence number already seen is dropped as late.

## SDP (section 7.2) and ST 2110-22

`vc2_rtp_sdp` (`vc2send --sdp`) writes:

    m=video 5004 RTP/AVP 96
    c=IN IP4 127.0.0.1
    b=AS:310773
    a=rtpmap:96 vc2/90000
    a=fmtp:96 profile=HQ;version=3;level=0;sampling=YCbCr-4:2:2;depth=10;width=1920;height=1080;exactframerate=50;colorimetry=BT709;TCS=SDR;PM=2110GPM;SSN=ST2110-22:2019;TP=2110TPN
    a=framerate:50
    a=mediaclk:direct=0

`profile`, `version` and `level` are RFC 8450's parameters. The others are what an ST 2110-22 receiver
expects, as we use them for our JPEG XS streams: the bit rate in `b=AS` (kbit/s), `a=framerate`, the sender
type `TP`, the standard number `SSN`, the video description in the style of ST 2110-20 (sampling, depth,
width, height, exactframerate, colorimetry, TCS, PM), and from ST 2110-10 `a=mediaclk:direct=0` and
`a=ts-refclk` (passed in by the caller; left out by `vc2send`, which has no PTP clock). These names were
written from our working notes, not checked against the SMPTE documents: check them before interop. The
colorimetry and `TP=2110TPN` are fixed in the code today; the sender does not shape its packet timing to the
ST 2110-21 narrow model beyond spreading a picture's packets evenly over the frame period.

## Open

- A run on a real network (needs the network owner's agreement), multicast, IGMP.
- ST 2022-7 merge; forward error correction (section 5).
- A sender that packetises while encoding (`--rc-rows 1` makes the encoder's budget row-wise; the encoder
  itself still works on whole frames).
- Timestamps from a PTP media clock (the tools use the system clock) and RTP into the on-screen player.
