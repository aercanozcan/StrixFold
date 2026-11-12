#pragma once
// DeepSeek V4.1 vision host reference: grid plan, prep (Pillow BICUBIC + ImageOps.pad, BF16 patches), ViT + aligner.
#include "ds41_vision_math.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::image {

// A picture or prep error with a client-facing message.
class ImageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// 8-bit RGB, rows top to bottom, pixels interleaved R, G, B.
struct Rgb8 {
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> pixels;
};

}  // namespace strix::image

namespace strix::deepseek41::vision {

// vision_config of the release's config.json (the 1017 tower; the sidecar's tensor shapes are checked against it).
struct VisionConfig {
    std::uint32_t layers = 32, dim = 1024, heads = 16, inter = 2816, patch = 14, merge = 3, out_dim = 5120;
    std::uint32_t max_tokens = 1024, min_pixels = 295936, image_token_id = 129264;
    float theta = 10000.0f, eps = 1e-6f;
    std::optional<double> max_wh_ratio;  // null in the release
    [[nodiscard]] std::uint32_t head_dim() const noexcept { return dim / heads; }
    [[nodiscard]] std::uint32_t patch_in() const noexcept { return 3U * patch * patch; }
    [[nodiscard]] std::uint32_t merged_in() const noexcept { return dim * merge * merge; }
};

// plan_image_grid: the pixel size the picture is padded to, its ViT patch grid and the aligner's token grid.
struct GridPlan {
    std::uint32_t best_w = 0, best_h = 0, vit_w = 0, vit_h = 0, llm_w = 0, llm_h = 0;
    [[nodiscard]] std::uint32_t patches() const noexcept { return vit_w * vit_h; }
    [[nodiscard]] std::uint32_t cells() const noexcept { return llm_w * llm_h; }
    // [IMAGE_START] + ([IMAGE] * llm_w + [IMAGE_NEW_LINE]) * llm_h + [IMAGE_END]
    [[nodiscard]] std::uint32_t span_tokens() const noexcept { return llm_h * (llm_w + 1U) + 2U; }
    friend bool operator==(const GridPlan&, const GridPlan&) = default;
};

[[nodiscard]] GridPlan plan_grid(std::uint32_t width, std::uint32_t height, const VisionConfig& config);

// Pillow 11.3 Image.resize(size, BICUBIC) on an RGB image (8-bit fixed-point two-pass resample).
[[nodiscard]] image::Rgb8 resize_bicubic(const image::Rgb8& in, std::uint32_t width, std::uint32_t height);
// ImageOps.pad(image, (width, height), color=(fill, fill, fill)): contain() then centre on the fill colour.
[[nodiscard]] image::Rgb8 pad(const image::Rgb8& in, std::uint32_t width, std::uint32_t height, std::uint8_t fill = 127);

// One picture ready for the tower: its plan and BF16 patches [vit_h * vit_w][3][patch][patch], ((v / 255) - 0.5) / 0.5.
struct PreparedImage {
    GridPlan plan;
    std::vector<std::uint16_t> patches;
};

[[nodiscard]] PreparedImage prepare(const image::Rgb8& picture, const VisionConfig& config);

// Span position kinds (image_processor.py: IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END).
enum class SpanKind : std::uint8_t { start = 0, cell = 1, newline = 2, end = 3 };
[[nodiscard]] SpanKind span_kind(const GridPlan& plan, std::uint32_t index, std::uint32_t* cell = nullptr) noexcept;

// --- the vision weights (checkpoint tensor names, BF16) ---
struct BlockWeights {
    const std::uint16_t *norm1 = nullptr, *wqkv = nullptr, *bqkv = nullptr, *wo = nullptr, *bo = nullptr;
    const std::uint16_t *norm2 = nullptr, *w1 = nullptr, *w2 = nullptr;
};
// All tensors in one BF16 blob, each [out][in] row-major (torch nn.Linear layout).
struct VisionWeights {
    VisionConfig config;
    std::uint64_t file_bytes = 0;
    std::vector<std::uint16_t> blob;
    struct Tensor {
        std::size_t offset = 0, elements = 0;
    };
    std::vector<std::pair<std::string, Tensor>> tensors;  // checked order
    // views into blob
    const std::uint16_t *patch_w = nullptr, *patch_b = nullptr, *norm = nullptr;
    const std::uint16_t *a1_w = nullptr, *a1_b = nullptr, *a2_w = nullptr, *a2_b = nullptr;
    const std::uint16_t *image_start = nullptr, *image_newline = nullptr, *image_end = nullptr;
    std::vector<BlockWeights> blocks;
    [[nodiscard]] const Tensor* find(const std::string& name) const;
};
// The 266 tower tensors of DIR/vision.safetensors (make_vision_sidecar.py), names, shapes and BF16 checked.
[[nodiscard]] VisionWeights load_vision_sidecar(const std::filesystem::path& dir);

// The ViT's 2D rotary tables for a patch grid: cos/sin [vit_h * vit_w][head_dim / 2], FP32, double rounded to FP32.
void rope_tables(const GridPlan& plan, const VisionConfig& config, std::vector<float>& cos, std::vector<float>& sin);

// The host needs FMA (x86-64: AVX2 + FMA) for encode_reference's std::fma to be one instruction.
[[nodiscard]] bool host_has_fma() noexcept;
// The observer (optional) sees each named step's BF16 output: "embed", "block" (per layer), "norm", "rows".
using StepObserver = std::function<void(const char* step, std::uint32_t layer, std::span<const std::uint16_t> values)>;
// Host reference of the tower: patches -> aligner rows [cells][out_dim] (BF16), FP32 FMA chains, BF16 rounding points.
[[nodiscard]] std::vector<std::uint16_t> encode_reference(const VisionWeights& weights, const PreparedImage& image,
                                                          const StepObserver& observer = {}, unsigned threads = 0);
// The span rows [span_tokens][out_dim] (BF16): image_start, each row of cells then image_newline, image_end.
[[nodiscard]] std::vector<std::uint16_t> span_rows(const VisionWeights& weights, const GridPlan& plan,
                                                   std::span<const std::uint16_t> cell_rows);

// --- bf16 helpers (round to nearest even, as torch's .to(torch.bfloat16)) ---
[[nodiscard]] std::uint16_t bf16_bits(float v) noexcept;
[[nodiscard]] float bf16_value(std::uint16_t bits) noexcept;

}  // namespace strix::deepseek41::vision
