#!/usr/bin/env bash
# fetch_golden.sh DEST [URL]: the published golden, checked, in DEST/vision-golden-1 (URL: the release, or a file).
set -Eeuo pipefail
# the tarball's sha256 is pinned here; its contents must then match this folder's SHA256SUMS
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=vision-golden-1.tar.zst
SHA256=ef3cbbe0b2c69f29b22c25d7cfa90080c9f50aecd1b70db7c477ca041fbb9058
[[ $# -ge 1 && $# -le 2 ]] || { sed -n 2p "$0"; exit 2; }
DEST=$(realpath -m "$1")
URL=${2:-https://github.com/dturini12/StrixFold/releases/download/vision-golden-1/$NAME}
mkdir -p "$DEST"
if [[ -f $URL ]]; then cp "$URL" "$DEST/$NAME.part"; else curl -fL --retry 3 -o "$DEST/$NAME.part" "$URL"; fi
echo "$SHA256  $DEST/$NAME.part" | sha256sum -c --quiet - || { echo "$URL: not the published $NAME" >&2; exit 1; }
mv "$DEST/$NAME.part" "$DEST/$NAME"
tar --zstd -C "$DEST" -xf "$DEST/$NAME"
cmp -s "$DEST/vision-golden-1/SHA256SUMS" "$HERE/SHA256SUMS" || { echo "the tarball's SHA256SUMS is not this folder's" >&2; exit 1; }
python3 -I "$HERE/golden.py" check "$DEST/vision-golden-1"
echo "golden at $DEST/vision-golden-1 (tf-dsv41-vision-check --golden $DEST/vision-golden-1)"
