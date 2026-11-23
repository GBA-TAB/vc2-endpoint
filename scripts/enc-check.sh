#!/bin/sh
# enc-check.sh: the encoder's fast paths against its plain ones, byte for byte. Every optimisation of the encoder
# is meant to leave the output unchanged; this encodes a grid of sizes, wavelets, rates and modes with each fast
# path switched off through its environment switch, and compares.
#   scripts/enc-check.sh BUILD clip.yuv [more clips...]   (planar 16-bit 4:2:2, as vc2enc reads them; 1080p or larger)
# With a GPU=cl build also the GPU front end (skipped where it is not available). Exit status = cases that differ.
set -u
B=${1:?build directory}; shift
[ $# -ge 1 ] || { echo "usage: $0 BUILD clip.yuv [clip.yuv ...]" >&2; exit 2; }
ENC=$B/vc2enc
T=$(mktemp -d "${TMPDIR:-/tmp}/enc-check.XXXXXX") && trap 'rm -rf "$T"' EXIT
bad=0; n=0
# switch: the plain path to compare with (each fast path has one)
SWITCHES="VC2_NO16=1 VC2_NO16C=1 VC2_WV_STRIPS=1 VC2_NO_RAWIN=1"
one() { # one() name env... -- args
    ref=$1; shift
    env $ref "$ENC" "$@" "$T/a.vc2" >/dev/null 2>&1 || return 0
    "$ENC" "$@" "$T/b.vc2" >/dev/null 2>&1 || return 0
    n=$((n + 1))
    cmp -s "$T/a.vc2" "$T/b.vc2" || { bad=$((bad + 1)); echo "differs ($ref): $*"; }
}
for clip in "$@"; do
    for sz in 1920x1080 1280x720 998x562; do
        for wl in 1 0; do
            for bpp in 1.5 3 8; do
                for mode in "" "--no-truncate" "--rc-rows 1" "--max-slice 1168"; do
                    for sw in $SWITCHES; do
                        one $sw -s $sz --wavelet $wl --bpp $bpp --threads 4 $mode "$clip"
                    done
                done
            done
        done
    done
    for q in 0 1 4 11 20 40; do
        for sw in $SWITCHES; do one $sw -s 1920x1080 --q $q --no-truncate --threads 4 "$clip"; done
    done
    # the GPU front end against the CPU path (a GPU=cl build; refused sizes are skipped)
    if "$ENC" --gpu cl -s 1920x1080 --bpp 3 --loops 1 "$clip" "$T/g.vc2" >/dev/null 2>&1; then
        for sz in 1920x1080 1280x720; do
            for bpp in 1.5 3 8; do
                for mode in "" "--no-truncate" "--rc-rows 1"; do
                    "$ENC" -s $sz --bpp $bpp --threads 3 $mode "$clip" "$T/a.vc2" >/dev/null 2>&1
                    for gsw in X=0 VC2_NO_GPU_COUNT=1 VC2_GPU_NO_IMPORT=1; do
                        env $gsw "$ENC" --gpu cl -s $sz --bpp $bpp --threads 3 $mode "$clip" "$T/b.vc2" >/dev/null 2>&1
                        n=$((n + 1))
                        cmp -s "$T/a.vc2" "$T/b.vc2" || { bad=$((bad + 1)); echo "differs (GPU, $gsw): -s $sz --bpp $bpp $mode $clip"; }
                    done
                done
            done
        done
    fi
done
echo "$bad of $n cases differ"
exit $bad
