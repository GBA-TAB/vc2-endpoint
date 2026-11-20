#!/bin/bash
# play.sh <build dir> <file.vc2> [vc2play options]: desktop off, play, desktop back on (run as root on the Firefly)
build=$1; clip=$2; shift 2
systemctl stop lightdm; sleep 2
"$build/vc2play" "$@" "$clip"
rc=$?
systemctl start lightdm
exit $rc
