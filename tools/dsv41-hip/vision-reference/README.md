# DeepSeek-V4.1 vision host reference

The CPU reference of the DeepSeek-V4.1 vision tower (ViT + aligner) that `tf-dsv41-vision-check`
(`zig/tests/dsv41_hip/vision_check.zig`) compares the HIP tower against, byte for byte, step by step. It writes the
golden dumps that check reads; the published dumps are its output for three synthetic pictures.

The reference does the tower's declared arithmetic: every linear one FP32 `fma` chain over k ascending per output (BF16
after the bias), RMSNorm a wave-order sum with a correctly rounded root, attention an online softmax over 64-key blocks
with a declared exp, GELU through fdlibm's `erff`, and the 2D RoPE tables in double precision through libm's `pow`,
`cos` and `sin`. `zig/kernels/hip/dsv41/vision.hip` repeats the same operations in the same order.

## Where it comes from

Daniel Turini wrote these sources for strix-halo-engine-lab (branch `pr/ds41-vision`, commit 481141258, Apache-2.0).
The commit that adds this folder brings them in unchanged, and the commit after it adapts them. The arithmetic is unchanged.
The adaptation:

- `ds41_vision.cpp` reads the 266 tower tensors from the safetensors sidecar that StrixFold's HIP tower also reads
  (`vision.safetensors`, made by `tools/dsv41-zig/kit/make_vision_sidecar.py`), instead of engine-lab's GGUF file. The
  configuration is the release's `vision_config` (the GGUF's metadata held the same values), checked against the
  tensor shapes.
- The image decoders, the prepared-image SHA-256 digest and `prepare_bytes` are gone. Pictures come in as raw RGB.
- `declared.hpp` holds `declared_exp`, `declared_sqrt` and `STRIX_DS41_HD` from hyukudan/strix-halo-engine-lab's
  `include/strix/ds41_moe.hpp` (Apache-2.0, see `THIRD_PARTY_NOTICES.md`) and nothing else from that header.
- The comments are shorter, and `ds41_vision_math.hpp` now carries the notice of the fdlibm `erff` it follows.

| File | What |
|---|---|
| `ds41_vision.hpp`, `ds41_vision_math.hpp` | API, configuration, element-wise declared arithmetic |
| `ds41_vision_prep.cpp` | grid plan and prep (Pillow 11.3 BICUBIC `ImageOps.pad`, BF16 patches) |
| `ds41_vision_reference.cpp` | `encode_reference`: patch embed, 32 blocks, norm, 3x3 unfold, aligner |
| `ds41_vision.cpp` | sidecar reader, RoPE tables, span rows |
| `declared.hpp` | hyukudan's `declared_exp` / `declared_sqrt` |
| `main.cpp` | `vision-reference`: one folder of dumps per picture |
| `golden.py` | the pictures, `index.json` / `SHA256SUMS`, `check` |
| `build.sh`, `make_golden.sh`, `package_golden.sh`, `fetch_golden.sh` | build, regenerate, package, download |
| `SHA256SUMS` | the published golden: every file of `syn64x48`, `syn700x250`, `tall` and `index.json` |

## Build

    tools/dsv41-hip/vision-reference/build.sh [OUT_BINARY]

`build.sh` compiles with the builder image's clang++ (`localhost/strixfold-builder:rocm10.0`, or set
`STRIXFOLD_BUILDER`), using `-O3 -std=gnu++20 -ffp-contract=off -fno-fast-math -mavx2 -mfma`. The default output is
`./vision-reference`. To build with the host's `clang++` instead, set `VISION_REFERENCE_NATIVE=1`. The host needs AVX2
and FMA.

## Regenerate

    tools/dsv41-hip/vision-reference/make_golden.sh SIDECAR_DIR OUT_DIR [CASE...]

`SIDECAR_DIR` is a vision sidecar (`vision.safetensors` with its `SHA256SUMS`). The weights are the model's and are not
published, so make the sidecar from a checkpoint you hold with `make_vision_sidecar.py`.

The script builds `vision-reference` if needed. It runs it in the builder image (CPU only, no network; the RoPE tables
use that image's glibc 2.39 libm), writes `OUT_DIR/<case>/` for each case, then writes `OUT_DIR/index.json` (each case's
grid and the sha256 of its rows and span) and `OUT_DIR/SHA256SUMS`. One case takes about 10 s on 32 threads. Each case
folder holds `grid.json`, `rgb.u8`, `patches.bf16`, `cos.f32`, `sin.f32`, `embed.bf16`, `block00.bf16` through
`block31.bf16`, `norm.bf16`, `rows.bf16` and `span.bf16`, all raw little-endian, about 112 MB.

| Case | Picture | ViT grid | Cells | Published |
|---|---|---|---|---|
| `syn64x48` | the device test's synthetic 64x48 (seed 64) | 34 x 45 | 12 x 15 | yes |
| `syn700x250` | synthetic 700x250 (seed 700) | 24 x 65 | 8 x 22 | yes |
| `tall` | 40x900 gradient | 185 x 9 | 62 x 3 | yes |
| `odd` | 431x257 pattern | 30 x 51 | 10 x 17 | no |
| `rgba` | 300x200 RGBA shapes through Pillow's `convert("RGB")` (needs Pillow) | 32 x 48 | 11 x 16 | no |

For another picture, run `vision-reference SIDECAR_DIR OUT_DIR THREADS NAME=raw:W:H:FILE` with `W*H*3` RGB bytes in
`FILE`.

## Verify

    python3 tools/dsv41-hip/vision-reference/golden.py check OUT_DIR

This compares every file in this folder's `SHA256SUMS` with `OUT_DIR`. Regenerate exactly the published cases for this
check, because `index.json` lists every generated case. The reference built here gave the same bytes as engine-lab's
own build for `syn64x48`, `syn700x250`, `odd`, `tall` and `rgba`: all 41 files of each case matched.

## Download

    tools/dsv41-hip/vision-reference/fetch_golden.sh DEST [URL]

This downloads `vision-golden-1.tar.zst` (257 MB; the sha256 is pinned in the script) from the release, or from the
URL or local file you give it. It unpacks the tarball to `DEST/vision-golden-1` and checks the contents against
`SHA256SUMS`. `package_golden.sh GOLDEN_DIR OUT_DIR` makes the tarball from `make_golden.sh`'s output. It is
reproducible: sorted names, fixed owner, mode and mtime, `zstd -19 --long=27` (zstd 1.5.7).

## Run tf-dsv41-vision-check against it

    zig build dsv41-hip-gpu-test        # with BUILD_AND_RUN.md's options; zig-out/native-hip/test/{tf-dsv41-vision-check,dsv41-kit}
    zig-out/native-hip/test/tf-dsv41-vision-check --golden DEST/vision-golden-1 --prep-only
    zig-out/native-hip/test/tf-dsv41-vision-check --golden DEST/vision-golden-1 \
        --kit zig-out/native-hip/test/dsv41-kit --weights SIDECAR_DIR [--case NAME]...

`--prep-only` compares `picture.zig`'s grid and patches with each case's `grid.json` and `patches.bf16`, and needs no
GPU. The full run also encodes each case on the GPU and checks 37 probes against the dumps: `cos`, `sin`, `embed`,
the 32 blocks, `norm` and `rows`. Then it checks `span.bf16`, and that two more encodes give the same span. Give the
check a GPU that nothing else is using, because it times the encodes.
