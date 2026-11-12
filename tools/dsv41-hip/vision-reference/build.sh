#!/usr/bin/env bash
# build.sh [OUT_BINARY]: vision-reference (default ./vision-reference), built by the builder image's clang++ (podman).
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$(realpath -m "${1:-$HERE/vision-reference}")
IMAGE=${STRIXFOLD_BUILDER:-localhost/strixfold-builder:rocm10.0}
mkdir -p "$(dirname "$OUT")"
# host code exactly as the reference was built: no contraction, no fast math, AVX2 + FMA (std::fma is one instruction)
FLAGS=(-O3 -DNDEBUG -std=gnu++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
       -ffp-contract=off -fno-fast-math -mavx2 -mfma -pthread)
SOURCES=(main.cpp ds41_vision.cpp ds41_vision_prep.cpp ds41_vision_reference.cpp)
if [[ ${VISION_REFERENCE_NATIVE:-0} == 1 ]]; then
  cd "$HERE" && exec "${CXX:-clang++}" "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT"
fi
podman run --rm --network=none -v "$HERE:/src:ro,z" -v "$(dirname "$OUT"):/out:z" -w /src "$IMAGE" \
  /opt/rocm/lib/llvm/bin/clang++ "${FLAGS[@]}" "${SOURCES[@]}" -o "/out/$(basename "$OUT")"
echo "built $OUT"
