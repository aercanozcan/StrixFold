#!/usr/bin/env bash
# make_golden.sh SIDECAR_DIR OUT_DIR [CASE...]: golden dumps (default: the published cases), OUT_DIR/SHA256SUMS.
set -Eeuo pipefail
# SIDECAR_DIR holds vision.safetensors (tools/dsv41-zig/kit/make_vision_sidecar.py). THREADS (default: all CPUs).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { sed -n 2p "$0"; exit 2; }
SIDECAR=$(realpath "$1")
OUT=$(realpath -m "$2")
shift 2
IMAGE=${STRIXFOLD_BUILDER:-localhost/strixfold-builder:rocm10.0}
BIN=${VISION_REFERENCE_BIN:-$HERE/vision-reference}
[[ -f $SIDECAR/vision.safetensors ]] || { echo "no vision.safetensors in $SIDECAR" >&2; exit 1; }
(cd "$SIDECAR" && sha256sum --quiet -c SHA256SUMS) || { echo "$SIDECAR fails its SHA256SUMS" >&2; exit 1; }
[[ -x $BIN ]] || "$HERE/build.sh" "$BIN"
mkdir -p "$OUT/.inputs"
LIST=$(python3 -I "$HERE/golden.py" pictures "$OUT/.inputs" "$@")
mapfile -t SPECS <<< "$LIST"
# the reference runs in the builder image (its glibc's pow, cos and sin make the RoPE tables) on the CPU only
podman run --rm --network=none -v "$SIDECAR:$SIDECAR:ro" -v "$OUT:$OUT:z" -v "$(dirname "$BIN"):/bin-ref:ro" "$IMAGE" \
  "/bin-ref/$(basename "$BIN")" "$SIDECAR" "$OUT" "${THREADS:-$(nproc)}" "${SPECS[@]}"
rm -rf "$OUT/.inputs"
python3 -I "$HERE/golden.py" finalize "$OUT" "${SPECS[@]%%=*}"
