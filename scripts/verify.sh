#!/bin/bash
# Bit-exactness of the decoder against ffmpeg's dirac decoder.
#   scripts/verify.sh <build dir> <clip.yuv> <width> <height> [work dir]
# (a) streams from ffmpeg's vc2 encoder, (b) streams from our encoder with every coefficient written out
# (--no-truncate), (c) our truncated streams against the same content written out in full: ffmpeg's decoder
# parses one byte past a short slice component, so truncated streams are checked against ourselves.
# Each stream is decoded three ways by us: one thread, three threads, and fed slice row by slice row.
# GPU=cl (or vk): the same three ways again with vc2bench --gpu, which must give the same picture; the last
# line says how many of these decodes ran on the GPU (wavelets it does not do stay on the CPU).
# FFSYNC: ffmpeg's option against frame duplication ("-vsync passthrough" for ffmpeg 4).
B=$1; CLIP=$2; W=$3; H=$4; WORK=${5:-$(mktemp -d)}
RUN=${RUN:-}   # e.g. RUN="nice -n 19 taskset -c 10,11"
GPU=${GPU:-}
FFSYNC=${FFSYNC:--fps_mode passthrough}
: > $WORK/gpu.count
pass=0; fail=0
# -cpuflags 0: ffmpeg's x86 SIMD dequantisation is wrong for slices whose subband width is not a multiple of 8
# (e.g. 240-pixel slices); its C code, which is what arm64 runs, is right.
ffdec() { $RUN ffmpeg -hide_banner -loglevel error -y -cpuflags 0 -i "$1" $FFSYNC -f rawvideo -pix_fmt yuv422p10le "$2"; }
ours() { # stream -> prints the md5 if all three ways agree
    local a b c
    a=$($RUN $B/vc2bench --frames 4 --out $WORK/o1.yuv "$1" >/dev/null && md5sum < $WORK/o1.yuv)
    b=$($RUN $B/vc2bench --frames 4 --threads 3 --out $WORK/o2.yuv "$1" >/dev/null && md5sum < $WORK/o2.yuv)
    c=$($RUN $B/vc2bench --frames 4 --subframe 40 --out $WORK/o3.yuv "$1" >/dev/null && md5sum < $WORK/o3.yuv)
    [ -n "$a" ] && [ "$a" = "$b" ] && [ "$a" = "$c" ] || return
    if [ -n "$GPU" ]; then
        local way g
        for way in "" "--threads 3" "--subframe 40"; do
            $RUN $B/vc2bench --frames 4 $way --gpu $GPU --out $WORK/o4.yuv "$1" > $WORK/o4.txt 2>/dev/null || return
            g=$(md5sum < $WORK/o4.yuv)
            [ "$g" = "$a" ] || return
            grep -q " gpu " $WORK/o4.txt && echo >> $WORK/gpu.count
        done
    fi
    echo "$a"
}
check() { # name stream reference.yuv
    local m r
    m=$(ours "$2"); r=$(md5sum < "$3")
    if [ -n "$m" ] && [ "$m" = "$r" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $1"; fi
    [ -n "$VERBOSE" ] && echo "ok $1"
}
# (a) ffmpeg's encoder
for wl in 9_7 5_3 haar; do for sl in "32 16" "64 32" "128 16"; do for bpp in 3 6; do for depth in 3 4; do
    set -- $sl
    br=$(( W * H * bpp * 50 ))
    $RUN ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt yuv422p10le -s ${W}x${H} -r 50 -i $CLIP -c:v vc2 -strict -1 -b:v $br \
        -wavelet_type $wl -wavelet_depth $depth -slice_width $1 -slice_height $2 -f dirac $WORK/ff.vc2 || continue
    ffdec $WORK/ff.vc2 $WORK/ref.yuv
    check "ffmpeg-encoded $wl ${1}x$2 bpp $bpp depth $depth" $WORK/ff.vc2 $WORK/ref.yuv
done; done; done; done
# (b) and (c) our encoder
for wl in 0 1 2 3 4; do for sl in 32x16 128x16 64x32 240x8; do for q in 8 20 32; do for depth in 2 4; do
    [ $sl = 240x8 ] && [ $depth = 4 ] && continue
    $RUN $B/vc2enc -q -s ${W}x${H} --wavelet $wl --depth $depth --slice $sl --q $q --tail 0 --no-truncate $CLIP $WORK/full.vc2
    $RUN $B/vc2enc -q -s ${W}x${H} --wavelet $wl --depth $depth --slice $sl --q $q --tail 0 $CLIP $WORK/trunc.vc2
    ffdec $WORK/full.vc2 $WORK/ref.yuv
    check "ours, written out, wavelet $wl $sl q $q depth $depth" $WORK/full.vc2 $WORK/ref.yuv
    check "ours, truncated, wavelet $wl $sl q $q depth $depth" $WORK/trunc.vc2 $WORK/ref.yuv
done; done; done; done
# rate-controlled streams as used in practice (truncated, tail dropping): the three ways must agree
for wl in 0 1; do for bpp in 1.5 2 3 5; do
    $RUN $B/vc2enc -q -s ${W}x${H} --wavelet $wl --bpp $bpp $CLIP $WORK/rc.vc2
    if [ -n "$(ours $WORK/rc.vc2)" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL rate-controlled wavelet $wl bpp $bpp"; fi
done; done
echo "pass $pass fail $fail"
[ -n "$GPU" ] && echo "decodes on the GPU ($GPU): $(wc -l < $WORK/gpu.count)"
[ $fail = 0 ]
