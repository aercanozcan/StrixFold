#!/usr/bin/env bash
# package_golden.sh GOLDEN_DIR OUT_DIR: OUT_DIR/vision-golden-1.tar.zst (+ .sha256) of SHA256SUMS's files.
set -Eeuo pipefail
# GOLDEN_DIR is make_golden.sh's output for the published cases; it must match this folder's SHA256SUMS first.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 2 ]] || { sed -n 2p "$0"; exit 2; }
GOLDEN=$(realpath "$1")
OUT=$(realpath -m "$2")
NAME=vision-golden-1
python3 -I "$HERE/golden.py" check "$GOLDEN"
cmp -s "$GOLDEN/SHA256SUMS" "$HERE/SHA256SUMS" || { echo "$GOLDEN/SHA256SUMS is not this folder's" >&2; exit 1; }
mkdir -p "$OUT"
# the listed files and SHA256SUMS under NAME/, in byte order, with fixed owner, mode and mtime
{ echo SHA256SUMS; cut -d' ' -f3- "$HERE/SHA256SUMS"; } | LC_ALL=C sort > "$OUT/.files"
LC_ALL=C tar -C "$GOLDEN" --format=gnu --no-recursion --owner=0 --group=0 --numeric-owner --mode=u=rw,go=r \
  --mtime=@0 --transform="s,^,$NAME/," -T "$OUT/.files" -cf - | zstd -19 --long=27 -T0 -q -f -o "$OUT/$NAME.tar.zst"
rm -f "$OUT/.files"
(cd "$OUT" && sha256sum "$NAME.tar.zst" > "$NAME.tar.zst.sha256" && cat "$NAME.tar.zst.sha256")
ls -l "$OUT/$NAME.tar.zst"
