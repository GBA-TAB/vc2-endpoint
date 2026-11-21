#!/bin/bash
# Reference vectors for a hardware decoder (docs/FPGA.md), from tools/vc2dump.
#   scripts/fpga-vectors.sh check <build dir> <clip.yuv> <width> <height> [work dir]
#       vc2dump's final picture against vc2dec's, for streams across the parameter space (both encoders,
#       every wavelet the subset names and more, depths 1-4, slice sizes, quantisers, truncated or not).
#   scripts/fpga-vectors.sh make <build dir> <clip.yuv> <width> <height> <out dir> [frames]
#       the vector set: the streams of the decoder subset, each dumped (default 2 frames), checked, packed
#       per stream as <name>.tar.xz, plus ranges.txt (largest values over all of them) and an index.
# ffmpeg is needed for the ffmpeg-encoded streams; FFSYNC as in verify.sh.
set -u
MODE=$1; B=$2; CLIP=$3; W=$4; H=$5
RUN=${RUN:-}
FFSYNC=${FFSYNC:--fps_mode passthrough}
ffenc() { # out wavelet depth slice_w slice_h bpp
    $RUN ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt yuv422p10le -s ${W}x${H} -r 50 -i $CLIP -c:v vc2 -strict -1 \
        -b:v $(( W * H * $6 * 50 )) -wavelet_type $2 -wavelet_depth $3 -slice_width $4 -slice_height $5 -f dirac $1
}
same() { # stream dumpdir: vc2dump's pictures = vc2dec's
    local a b
    a=$(for f in "$2"/f*/; do cat "$f/picture.yuv"; done | md5sum)
    $RUN $B/vc2dec "$1" "$2.ref.yuv" > /dev/null 2>&1 || return 1
    b=$(head -c $(( $(ls "$2"/f*/picture.yuv | wc -l) * W * H * 4 )) "$2.ref.yuv" | md5sum)
    rm -f "$2.ref.yuv"
    [ "$a" = "$b" ]
}

if [ "$MODE" = check ]; then
    WORK=${6:-$(mktemp -d)}
    pass=0; fail=0
    one() { # name stream
        rm -rf $WORK/d
        if $RUN $B/vc2dump "$2" $WORK/d 1 > /dev/null 2>&1 && same "$2" $WORK/d; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $1"; fi
    }
    for wl in 5_3 9_7 haar; do for sl in "32 16" "128 16" "64 32"; do for bpp in 2 3 6; do for depth in 3 4; do
        set -- $sl
        ffenc $WORK/s.vc2 $wl $depth $1 $2 $bpp || continue
        one "ffmpeg $wl ${1}x$2 bpp $bpp depth $depth" $WORK/s.vc2
    done; done; done; done
    for wl in 0 1 2 3 4; do for sl in 32x16 128x16 64x32 240x8; do for q in 8 20 32; do for depth in 1 2 4; do
        [ $sl = 240x8 ] && [ $depth = 4 ] && continue
        for tr in "--tail 0 --no-truncate" "--tail 0.2"; do
            $RUN $B/vc2enc -q -s ${W}x${H} --wavelet $wl --depth $depth --slice $sl --q $q $tr $CLIP $WORK/s.vc2 || continue
            one "ours wavelet $wl $sl q $q depth $depth $tr" $WORK/s.vc2
        done
    done; done; done; done
    for bpp in 1.5 2 3 5; do
        $RUN $B/vc2enc -q -s ${W}x${H} --bpp $bpp $CLIP $WORK/s.vc2 && one "ours default bpp $bpp" $WORK/s.vc2
    done
    rm -rf $WORK/d $WORK/s.vc2
    echo "vc2dump vs vc2dec: $pass passed, $fail failed"
    [ $fail = 0 ]
    exit
fi

[ "$MODE" = make ] || { echo "usage: $0 check|make ..."; exit 2; }
OUT=$6; FR=${7:-2}
mkdir -p $OUT
: > $OUT/index.txt
add() { # name stream description
    rm -rf $OUT/$1
    $RUN $B/vc2dump "$2" $OUT/$1 $FR > /dev/null 2>&1 || { echo "FAIL dump $1"; return 1; }
    [ $(ls -d $OUT/$1/f*/ | wc -l) = $FR ] || { echo "FAIL $1: not $FR frames"; return 1; }
    same "$2" $OUT/$1 || { echo "FAIL compare $1"; return 1; }
    cp "$2" $OUT/$1/stream.vc2
    echo "$1: $3; $(sed -n 3p $OUT/$1/f000/params.txt)" >> $OUT/index.txt
    (cd $OUT && tar cJf $1.tar.xz $1)
    echo "ok $1"
}
T=$OUT/tmp.vc2
# the subset (docs/FPGA.md): LeGall 5,3, depth 4, 4:2:2 10-bit, our default slices
for bpp in 2 3 5; do $RUN $B/vc2enc -q -s ${W}x${H} --bpp $bpp $CLIP $T && add ours_53_d4_b$bpp $T "our encoder, defaults, $bpp bpp, truncated slices"; done
$RUN $B/vc2enc -q -s ${W}x${H} --q 20 --tail 0 --no-truncate $CLIP $T && add ours_53_d4_q20_full $T "our encoder, fixed quantiser 20, every coefficient written"
$RUN $B/vc2enc -q -s ${W}x${H} --bpp 3 --max-slice 1168 $CLIP $T && add ours_53_d4_b3_cap1168 $T "our encoder, 3 bpp, slices capped at 1168 bytes (one RTP packet)"
$RUN $B/vc2enc -q -s ${W}x${H} --slice 64x32 --bpp 3 $CLIP $T && add ours_53_d4_b3_64x32 $T "our encoder, 3 bpp, 64x32 slices (another slice geometry)"
ffenc $T 5_3 4 128 16 3 && add ffmpeg_53_d4_128x16_b3 $T "ffmpeg's encoder, LeGall 5,3 depth 4, 128x16 slices, 3 bpp"
# stress: a sharp test pattern with noise (ffmpeg testsrc2), very high bitrate and the finest quantiser,
# for the bit widths
$RUN ffmpeg -nostdin -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=${W}x${H}:rate=50,noise=alls=40:allf=t" \
    -frames:v $FR -pix_fmt yuv422p10le -f rawvideo $OUT/stress.yuv &&
for spec in "stress_53_d4_b8|--bpp 8" "stress_53_d4_q1_full|--q 1 --tail 0 --no-truncate"; do
    $RUN $B/vc2enc -q -s ${W}x${H} ${spec#*|} $OUT/stress.yuv $T && add ${spec%%|*} $T "stress: testsrc2 + noise, ${spec#*|}"
done
rm -f $OUT/stress.yuv
# outside the subset, for later: DD 9,7 and depth 3
$RUN $B/vc2enc -q -s ${W}x${H} --wavelet 0 --bpp 3 $CLIP $T && add ours_dd97_d4_b3 $T "outside the subset: Deslauriers-Dubuc 9,7"
$RUN $B/vc2enc -q -s ${W}x${H} --depth 3 --bpp 3 $CLIP $T && add ours_53_d3_b3 $T "outside the subset: depth 3"
rm -f $T
# largest values over the subset's streams (names *_53_d4*, stress included), per component and subband / level output
cat $OUT/*_53_d4*/f*/ranges.txt | grep -vE '^(#|quantiser)' | awk '
    $2 == "synth" { k = $1 " synth " $3; if ($4 > s[k]) s[k] = $4; next }
    { k = $1 " " $2 " " $3; if ($5 > q[k]) q[k] = $5; if ($7 > dq[k]) dq[k] = $7 }
    END { for (k in q) print k, "q", q[k], "dq", dq[k]; for (k in s) print k, s[k] }' | sort > $OUT/ranges.txt
grep -h '^quantiser' $OUT/*_53_d4*/f*/ranges.txt | awk '{ if (lo == "" || $2 < lo) lo = $2; if ($4 > hi) hi = $4 } END { print "quantiser index over the subset streams:", lo, "..", hi }' >> $OUT/ranges.txt
echo "vector set in $OUT ($(ls $OUT/*.tar.xz | wc -l) streams)"
