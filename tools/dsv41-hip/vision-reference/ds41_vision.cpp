// DeepSeek V4.1 vision tower, host side: the deepseek4-vision GGUF (configuration and BF16 weights), the 2D rotary
// tables and the host reference of the ViT + aligner (the release's inference/vision.py) with the declared
// arithmetic of docs/models/deepseek-v4.1/vision.md. Every dot product is one FP32 accumulator per output, summed
// with fused multiply-adds in ascending index order from 0; the device kernels reproduce it bit for bit.
// Built with -ffp-contract=off.
#include "strix/ds41_vision.hpp"

#include "strix/gguf_reader.hpp"
#include "strix/model_input.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strix::deepseek41::vision {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("vision file: " + what); }

const gguf::MetadataEntry& meta(const gguf::FileInfo& f, const std::string& key) {
    const auto* e = f.find_metadata(key);
    if (!e) fail("missing metadata " + key);
    return *e;
}
std::uint64_t meta_u(const gguf::FileInfo& f, const std::string& key) {
    const auto& e = meta(f, key);
    if (e.unsigned_value) return *e.unsigned_value;
    if (e.signed_value && *e.signed_value >= 0) return static_cast<std::uint64_t>(*e.signed_value);
    fail(key + " is not a non-negative integer");
}
double meta_f(const gguf::FileInfo& f, const std::string& key) {
    const auto& e = meta(f, key);
    if (e.floating_value) return *e.floating_value;
    if (e.unsigned_value) return static_cast<double>(*e.unsigned_value);
    if (e.signed_value) return static_cast<double>(*e.signed_value);
    fail(key + " is not a number");
}
std::uint32_t u32(std::uint64_t v, const char* what) {
    if (v == 0U || v > 1'000'000U) fail(std::string("implausible ") + what);
    return static_cast<std::uint32_t>(v);
}

}  // namespace

bool host_has_fma() noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("fma") && __builtin_cpu_supports("avx2");
#else
    return true;
#endif
}

const VisionWeights::Tensor* VisionWeights::find(const std::string& name) const {
    for (const auto& [n, t] : tensors)
        if (n == name) return &t;
    return nullptr;
}

VisionWeights load_vision_file(const std::filesystem::path& path, bool metadata_only) {
    gguf::Limits limits;
    const auto f = gguf::read_file(path, limits);
    VisionWeights w;
    w.file_bytes = f.file_size;
    const auto arch = meta(f, "general.architecture").string_value;
    if (!arch || *arch != "deepseek4-vision") fail("general.architecture is not deepseek4-vision");
    if (const auto* rev = f.find_metadata("general.source.revision"); rev && rev->string_value) w.source_revision = *rev->string_value;
    auto& c = w.config;
    const std::string p = "deepseek4-vision.";
    c.layers = u32(meta_u(f, p + "block_count"), "block_count");
    c.dim = u32(meta_u(f, p + "embedding_length"), "embedding_length");
    c.inter = u32(meta_u(f, p + "feed_forward_length"), "feed_forward_length");
    c.heads = u32(meta_u(f, p + "attention.head_count"), "head_count");
    c.out_dim = u32(meta_u(f, p + "projection_length"), "projection_length");
    c.patch = u32(meta_u(f, p + "patch_size"), "patch_size");
    c.merge = u32(meta_u(f, p + "downsample_ratio"), "downsample_ratio");
    c.max_tokens = u32(meta_u(f, p + "image.max_tokens"), "image.max_tokens");
    c.min_pixels = static_cast<std::uint32_t>(meta_u(f, p + "image.min_pixels"));
    c.image_token_id = u32(meta_u(f, p + "image.token_id"), "image.token_id");
    c.theta = static_cast<float>(meta_f(f, p + "rope.freq_base"));
    c.eps = static_cast<float>(meta_f(f, p + "attention.layer_norm_rms_epsilon"));
    if (const double r = meta_f(f, p + "image.max_width_height_ratio"); r > 0.0) c.max_wh_ratio = r;
    if (c.dim % c.heads != 0U || c.head_dim() % 4U != 0U) fail("head geometry");
    if (c.max_wh_ratio) fail("image.max_width_height_ratio is not supported (null in the release)");

    // expected tensors and shapes ([out][in] as GGUF ne0 = in, ne1 = out)
    struct Want {
        std::string name;
        std::uint64_t in, out;  // out = 0: a vector of `in`
    };
    std::vector<Want> want{{"vision.patch_embed.proj.weight", c.patch_in(), c.dim}, {"vision.patch_embed.proj.bias", c.dim, 0},
                           {"vision.norm.weight", c.dim, 0},
                           {"aligner.w1.weight", c.merged_in(), c.out_dim},     {"aligner.w1.bias", c.out_dim, 0},
                           {"aligner.w2.weight", c.out_dim, c.out_dim},         {"aligner.w2.bias", c.out_dim, 0},
                           {"image_start", c.out_dim, 0},                       {"image_newline", c.out_dim, 0},
                           {"image_end", c.out_dim, 0}};
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string b = "vision.blocks." + std::to_string(l) + ".";
        want.push_back({b + "norm1.weight", c.dim, 0});
        want.push_back({b + "attn.wqkv.weight", c.dim, 3ULL * c.dim});
        want.push_back({b + "attn.wqkv.bias", 3ULL * c.dim, 0});
        want.push_back({b + "attn.wo.weight", c.dim, c.dim});
        want.push_back({b + "attn.wo.bias", c.dim, 0});
        want.push_back({b + "norm2.weight", c.dim, 0});
        want.push_back({b + "mlp.w1.weight", c.dim, 2ULL * c.inter});
        want.push_back({b + "mlp.w2.weight", c.inter, c.dim});
    }
    if (f.tensors.size() != want.size()) fail("expected " + std::to_string(want.size()) + " tensors, the file has " + std::to_string(f.tensors.size()));
    std::size_t total = 0;
    std::vector<const gguf::TensorInfo*> infos;
    for (const auto& wt : want) {
        const auto it = std::find_if(f.tensors.begin(), f.tensors.end(), [&](const auto& t) { return t.name == wt.name; });
        if (it == f.tensors.end()) fail("missing tensor " + wt.name);
        if (it->ggml_type != 30U) fail(wt.name + " is not BF16");
        const std::vector<std::uint64_t> dims = wt.out ? std::vector<std::uint64_t>{wt.in, wt.out} : std::vector<std::uint64_t>{wt.in};
        if (it->dimensions != dims) fail(wt.name + " has an unexpected shape");
        const std::size_t elements = static_cast<std::size_t>(wt.out ? wt.in * wt.out : wt.in);
        if (it->byte_size != elements * 2U) fail(wt.name + " byte size");
        w.tensors.push_back({wt.name, {total, elements}});
        total += elements;
        infos.push_back(&*it);
    }
    if (metadata_only) return w;
    w.blob.resize(total);
    ModelInput in(path);
    if (!in) fail("cannot open " + path.string());
    for (std::size_t i = 0; i < infos.size(); ++i) {
        in.seekg(static_cast<std::streamoff>(infos[i]->absolute_offset));
        in.read(reinterpret_cast<char*>(w.blob.data() + w.tensors[i].second.offset), static_cast<std::streamsize>(infos[i]->byte_size));
        if (!in) fail("short read of " + w.tensors[i].first);
    }
    const auto view = [&](const std::string& name) { return w.blob.data() + w.find(name)->offset; };
    w.patch_w = view("vision.patch_embed.proj.weight");
    w.patch_b = view("vision.patch_embed.proj.bias");
    w.norm = view("vision.norm.weight");
    w.a1_w = view("aligner.w1.weight");
    w.a1_b = view("aligner.w1.bias");
    w.a2_w = view("aligner.w2.weight");
    w.a2_b = view("aligner.w2.bias");
    w.image_start = view("image_start");
    w.image_newline = view("image_newline");
    w.image_end = view("image_end");
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string b = "vision.blocks." + std::to_string(l) + ".";
        w.blocks.push_back({view(b + "norm1.weight"), view(b + "attn.wqkv.weight"), view(b + "attn.wqkv.bias"), view(b + "attn.wo.weight"),
                            view(b + "attn.wo.bias"), view(b + "norm2.weight"), view(b + "mlp.w1.weight"), view(b + "mlp.w2.weight")});
    }
    return w;
}

void rope_tables(const GridPlan& g, const VisionConfig& c, std::vector<float>& cos, std::vector<float>& sin) {
    const std::uint32_t half = c.head_dim() / 2U, quarter = half / 2U, n = g.patches();
    std::vector<float> inv(quarter);
    for (std::uint32_t j = 0; j < quarter; ++j)
        inv[j] = static_cast<float>(1.0 / std::pow(static_cast<double>(c.theta), static_cast<double>(2U * j) / static_cast<double>(half)));
    cos.resize(std::size_t{n} * half);
    sin.resize(std::size_t{n} * half);
    for (std::uint32_t r = 0; r < n; ++r) {
        const float row = static_cast<float>(r / g.vit_w), col = static_cast<float>(r % g.vit_w);
        for (std::uint32_t j = 0; j < half; ++j) {
            const float angle = (j < quarter ? row : col) * inv[j % quarter];
            cos[std::size_t{r} * half + j] = static_cast<float>(std::cos(static_cast<double>(angle)));
            sin[std::size_t{r} * half + j] = static_cast<float>(std::sin(static_cast<double>(angle)));
        }
    }
}

std::vector<std::uint16_t> span_rows(const VisionWeights& w, const GridPlan& g, std::span<const std::uint16_t> cell_rows) {
    const std::size_t D = w.config.out_dim;
    if (cell_rows.size() != std::size_t{g.cells()} * D) throw std::runtime_error("cell rows size");
    std::vector<std::uint16_t> out(std::size_t{g.span_tokens()} * D);
    for (std::uint32_t i = 0; i < g.span_tokens(); ++i) {
        std::uint32_t cell = 0;
        const std::uint16_t* src = nullptr;
        switch (span_kind(g, i, &cell)) {
            case SpanKind::start: src = w.image_start; break;
            case SpanKind::newline: src = w.image_newline; break;
            case SpanKind::end: src = w.image_end; break;
            case SpanKind::cell: src = cell_rows.data() + std::size_t{cell} * D; break;
        }
        std::memcpy(out.data() + std::size_t{i} * D, src, D * 2U);
    }
    return out;
}

}  // namespace strix::deepseek41::vision
