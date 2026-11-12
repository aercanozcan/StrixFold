// DeepSeek V4.1 image preprocessing, equal to the release's inference/image_processor.py (load_image) with Pillow
// 11.3: plan_image_grid / safe_resize / solve_resize_ratio in double precision with Python's operation order,
// ImageOps.pad (contain + paste on (127, 127, 127)) with Pillow's 8-bit BICUBIC resample (Resample.c), and
// ((x / 255) - 0.5) / 0.5 in FP32 rounded to BF16. Built with -ffp-contract=off.
#include "strix/ds41_vision.hpp"

#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace strix::deepseek41::vision {
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;

// Python's round(x) for a float: nearest, ties to even.
std::int64_t py_round(double x) { return static_cast<std::int64_t>(std::nearbyint(x)); }

double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

// precompute_coeffs + normalize_coeffs_8bpc for in0 = 0, in1 = in_size (Image.resize's default box).
struct Coeffs {
    int ksize = 0;
    std::vector<int> bounds;      // [out][2] = xmin, count
    std::vector<std::int32_t> k;  // [out][ksize]
};

Coeffs coeffs(int in_size, int out_size) {
    Coeffs c;
    const float in0 = 0.0f, in1 = static_cast<float>(in_size);
    double filterscale = static_cast<double>(in1 - in0) / out_size;
    const double scale = filterscale;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = 2.0 * filterscale;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.bounds.resize(static_cast<std::size_t>(out_size) * 2U);
    c.k.assign(static_cast<std::size_t>(out_size) * static_cast<std::size_t>(c.ksize), 0);
    std::vector<double> kk(static_cast<std::size_t>(c.ksize));
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; ++x) {
            const double w = bicubic((x + xmin - center + 0.5) * ss);
            kk[static_cast<std::size_t>(x)] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
            if (ww != 0.0) kk[static_cast<std::size_t>(x)] /= ww;
        for (; x < c.ksize; ++x) kk[static_cast<std::size_t>(x)] = 0;
        for (x = 0; x < c.ksize; ++x) {
            const double v = kk[static_cast<std::size_t>(x)];
            c.k[static_cast<std::size_t>(xx) * static_cast<std::size_t>(c.ksize) + static_cast<std::size_t>(x)] =
                v < 0 ? static_cast<int>(-0.5 + v * (1 << kPrecisionBits)) : static_cast<int>(0.5 + v * (1 << kPrecisionBits));
        }
        c.bounds[static_cast<std::size_t>(xx) * 2U] = xmin;
        c.bounds[static_cast<std::size_t>(xx) * 2U + 1U] = xmax;
    }
    return c;
}

std::uint8_t clip8(int v) {
    const int s = v >> kPrecisionBits;  // arithmetic shift, as Pillow's lookup index
    return static_cast<std::uint8_t>(s < 0 ? 0 : s > 255 ? 255 : s);
}

}  // namespace

std::uint16_t bf16_bits(float v) noexcept {
    std::uint32_t u = std::bit_cast<std::uint32_t>(v);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
    u += 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}

float bf16_value(std::uint16_t bits) noexcept { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U); }

GridPlan plan_grid(std::uint32_t width_in, std::uint32_t height_in, const VisionConfig& cfg) {
    const auto p = static_cast<std::int64_t>(cfg.patch), r = static_cast<std::int64_t>(cfg.merge);
    const auto max_n = static_cast<std::int64_t>(cfg.max_tokens);
    double width = width_in, height = height_in;  // Python ints, but max_wh_ratio may make them floats
    std::int64_t wi = width_in, hi = height_in;
    bool is_float = false;
    if (cfg.max_wh_ratio && width > height * *cfg.max_wh_ratio) {
        width = height * *cfg.max_wh_ratio;
        is_float = true;
    }
    const double area = is_float ? width * height : static_cast<double>(wi * hi);
    if (0 < area && area < cfg.min_pixels) {
        const double ratio = std::pow(cfg.min_pixels / area, 0.5);
        wi = static_cast<std::int64_t>((is_float ? width : static_cast<double>(wi)) * ratio);
        hi = static_cast<std::int64_t>(static_cast<double>(hi) * ratio);
        width = static_cast<double>(wi);
        height = static_cast<double>(hi);
        is_float = false;
    }
    if (!is_float) {
        width = static_cast<double>(wi);
        height = static_cast<double>(hi);
    }
    std::int64_t best_w = static_cast<std::int64_t>(std::ceil(width / static_cast<double>(p))) * p;
    std::int64_t best_h = static_cast<std::int64_t>(std::ceil(height / static_cast<double>(p))) * p;
    const auto grid = [&](std::int64_t bh, std::int64_t bw) {
        return std::pair<std::int64_t, std::int64_t>{(bh / p + r - 1) / r, (bw / p + r - 1) / r};
    };
    auto [lh, lw] = grid(best_h, best_w);
    if (lh * (lw + 1) + 2 > max_n) {  // solve_resize_ratio(height, width, ...)
        const double rr = height / width;
        const double max_w_float = std::sqrt(static_cast<double>(max_n - 2) / rr + 0.25) - 0.5;
        const double max_h_float = max_w_float * rr;
        const std::int64_t cell = p * r;
        if (max_w_float < 1.0) {
            best_h = (max_n - 2) / 2 * cell;
            best_w = cell;
        } else if (max_h_float < 1.0) {
            best_h = cell;
            best_w = (max_n - 3) * cell;
        } else {
            const double beta = std::min(static_cast<double>(static_cast<std::int64_t>(std::floor(max_w_float)) * cell) / width,
                                         static_cast<double>(static_cast<std::int64_t>(std::floor(max_h_float)) * cell) / height);
            best_h = static_cast<std::int64_t>(std::floor(height * beta / static_cast<double>(p))) * p;
            best_w = static_cast<std::int64_t>(std::floor(width * beta / static_cast<double>(p))) * p;
        }
        std::tie(lh, lw) = grid(best_h, best_w);
        if (lh * (lw + 1) + 2 > max_n) throw image::ImageError("image input: no grid fits the token budget");
    }
    if (best_w <= 0 || best_h <= 0) throw image::ImageError("image input: the picture is too small");
    GridPlan g;
    g.best_w = static_cast<std::uint32_t>(best_w);
    g.best_h = static_cast<std::uint32_t>(best_h);
    g.vit_w = g.best_w / cfg.patch;
    g.vit_h = g.best_h / cfg.patch;
    g.llm_w = static_cast<std::uint32_t>(lw);
    g.llm_h = static_cast<std::uint32_t>(lh);
    return g;
}

image::Rgb8 resize_bicubic(const image::Rgb8& in, std::uint32_t width, std::uint32_t height) {
    if (width == in.width && height == in.height) return in;
    const int iw = static_cast<int>(in.width), ih = static_cast<int>(in.height);
    const int ow = static_cast<int>(width), oh = static_cast<int>(height);
    const image::Rgb8* src = &in;
    image::Rgb8 tmp;
    if (ow != iw) {  // horizontal pass over every row (Pillow's row window gives the same values)
        const auto c = coeffs(iw, ow);
        tmp.width = width;
        tmp.height = in.height;
        tmp.pixels.resize(std::size_t{width} * in.height * 3U);
        for (int y = 0; y < ih; ++y) {
            const std::uint8_t* row = in.pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(iw) * 3U;
            std::uint8_t* out = tmp.pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ow) * 3U;
            for (int xx = 0; xx < ow; ++xx) {
                const int xmin = c.bounds[static_cast<std::size_t>(xx) * 2U], xmax = c.bounds[static_cast<std::size_t>(xx) * 2U + 1U];
                const std::int32_t* k = c.k.data() + static_cast<std::size_t>(xx) * static_cast<std::size_t>(c.ksize);
                int s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xmax; ++x) {
                    const std::uint8_t* px = row + static_cast<std::size_t>(x + xmin) * 3U;
                    s0 += px[0] * k[x];
                    s1 += px[1] * k[x];
                    s2 += px[2] * k[x];
                }
                out[static_cast<std::size_t>(xx) * 3U] = clip8(s0);
                out[static_cast<std::size_t>(xx) * 3U + 1U] = clip8(s1);
                out[static_cast<std::size_t>(xx) * 3U + 2U] = clip8(s2);
            }
        }
        src = &tmp;
    }
    if (oh == ih) return src == &tmp ? tmp : in;
    const auto c = coeffs(ih, oh);
    image::Rgb8 out;
    out.width = src->width;
    out.height = height;
    out.pixels.resize(std::size_t{src->width} * height * 3U);
    const std::size_t stride = std::size_t{src->width} * 3U;
    for (int yy = 0; yy < oh; ++yy) {
        const int ymin = c.bounds[static_cast<std::size_t>(yy) * 2U], ymax = c.bounds[static_cast<std::size_t>(yy) * 2U + 1U];
        const std::int32_t* k = c.k.data() + static_cast<std::size_t>(yy) * static_cast<std::size_t>(c.ksize);
        std::uint8_t* o = out.pixels.data() + static_cast<std::size_t>(yy) * stride;
        for (std::size_t xx = 0; xx < stride; ++xx) {
            int s = 1 << (kPrecisionBits - 1);
            for (int y = 0; y < ymax; ++y) s += src->pixels[static_cast<std::size_t>(y + ymin) * stride + xx] * k[y];
            o[xx] = clip8(s);
        }
    }
    return out;
}

image::Rgb8 pad(const image::Rgb8& in, std::uint32_t width, std::uint32_t height, std::uint8_t fill) {
    // contain()
    const double im_ratio = static_cast<double>(in.width) / in.height;
    const double dest_ratio = static_cast<double>(width) / height;
    std::uint32_t cw = width, ch = height;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            const auto nh = py_round(static_cast<double>(in.height) / in.width * width);
            if (nh != static_cast<std::int64_t>(height)) ch = static_cast<std::uint32_t>(nh);
        } else {
            const auto nw = py_round(static_cast<double>(in.width) / in.height * height);
            if (nw != static_cast<std::int64_t>(width)) cw = static_cast<std::uint32_t>(nw);
        }
    }
    if (cw == 0U || ch == 0U) throw image::ImageError("image input: the picture is too thin to resize");
    auto resized = resize_bicubic(in, cw, ch);
    if (cw == width && ch == height) return resized;
    image::Rgb8 out;
    out.width = width;
    out.height = height;
    out.pixels.assign(std::size_t{width} * height * 3U, fill);
    std::uint32_t x0 = 0, y0 = 0;
    if (cw != width) x0 = static_cast<std::uint32_t>(py_round(static_cast<double>(width - cw) * 0.5));
    else y0 = static_cast<std::uint32_t>(py_round(static_cast<double>(height - ch) * 0.5));
    for (std::uint32_t y = 0; y < ch && y0 + y < height; ++y)
        std::memcpy(out.pixels.data() + (std::size_t{y0 + y} * width + x0) * 3U, resized.pixels.data() + std::size_t{y} * cw * 3U,
                    std::size_t{std::min(cw, width - x0)} * 3U);
    return out;
}

std::array<std::uint8_t, 32> prepared_digest(const GridPlan& g, std::span<const std::uint16_t> patches) {
    Sha256Fast h;
    std::uint8_t head[24];
    const std::uint32_t f[6] = {g.best_h, g.best_w, g.vit_h, g.vit_w, g.llm_h, g.llm_w};
    for (int i = 0; i < 6; ++i)
        for (int b = 0; b < 4; ++b) head[i * 4 + b] = static_cast<std::uint8_t>(f[i] >> (8 * b));
    h.update(std::as_bytes(std::span(head)));
    std::vector<std::uint8_t> le(patches.size() * 2U);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        le[2U * i] = static_cast<std::uint8_t>(patches[i]);
        le[2U * i + 1U] = static_cast<std::uint8_t>(patches[i] >> 8U);
    }
    h.update(std::as_bytes(std::span(le)));
    return h.finalize().bytes;
}

PreparedImage prepare(const image::Rgb8& picture, const VisionConfig& cfg) {
    if (picture.width == 0U || picture.height == 0U) throw image::ImageError("image input: empty picture");
    PreparedImage out;
    out.plan = plan_grid(picture.width, picture.height, cfg);
    const auto& g = out.plan;
    if (cfg.max_wh_ratio && static_cast<double>(picture.width) >= *cfg.max_wh_ratio * picture.height)
        throw image::ImageError("image input: max_wh_ratio resizing is not implemented");
    const auto padded = pad(picture, g.best_w, g.best_h, 127);
    const std::uint32_t p = cfg.patch;
    out.patches.resize(std::size_t{g.patches()} * 3U * p * p);
    std::array<std::uint16_t, 256> lut{};
    for (int v = 0; v < 256; ++v) {
        const float x = static_cast<float>(v) / 255.0f;  // FP32 division, then subtraction and division by 0.5
        const float y = (x - 0.5f) / 0.5f;
        lut[static_cast<std::size_t>(v)] = bf16_bits(y);
    }
    std::size_t o = 0;
    for (std::uint32_t py = 0; py < g.vit_h; ++py)
        for (std::uint32_t px = 0; px < g.vit_w; ++px)
            for (std::uint32_t c = 0; c < 3U; ++c)
                for (std::uint32_t ky = 0; ky < p; ++ky)
                    for (std::uint32_t kx = 0; kx < p; ++kx)
                        out.patches[o++] = lut[padded.pixels[(std::size_t{py * p + ky} * g.best_w + px * p + kx) * 3U + c]];
    out.digest = prepared_digest(g, out.patches);
    return out;
}

PreparedImage prepare_bytes(std::span<const std::uint8_t> encoded, const VisionConfig& cfg, const image::DecodeLimits& limits) {
    return prepare(image::decode_image(encoded, limits), cfg);
}

SpanKind span_kind(const GridPlan& g, std::uint32_t i, std::uint32_t* cell) noexcept {
    if (i == 0U) return SpanKind::start;
    if (i + 1U == g.span_tokens()) return SpanKind::end;
    const std::uint32_t k = i - 1U, row = k / (g.llm_w + 1U), col = k % (g.llm_w + 1U);
    if (col == g.llm_w) return SpanKind::newline;
    if (cell) *cell = row * g.llm_w + col;
    return SpanKind::cell;
}

}  // namespace strix::deepseek41::vision
