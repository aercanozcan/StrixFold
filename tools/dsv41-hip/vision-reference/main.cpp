// vision-reference: the golden dumps tf-dsv41-vision-check reads, every step of encode_reference for each picture.
#include "ds41_vision.hpp"

#include <bit>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace v = strix::deepseek41::vision;
namespace fs = std::filesystem;
static_assert(std::endian::native == std::endian::little);

namespace {

const char* const kUsage =
    "usage: vision-reference SIDECAR_DIR OUT_DIR THREADS CASE...\n"
    "  CASE: NAME=syn:W:H (the device test's synthetic picture, seeded with W) or NAME=raw:W:H:FILE (W*H*3 RGB bytes)";

template <class T>
void dump(const fs::path& path, std::span<const T> values) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
    out.close();
}

std::vector<std::uint8_t> read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// The synthetic picture of engine-lab's tests/hip_ds41_vision_tests.hip, seeded (as there) with the width.
strix::image::Rgb8 picture(std::uint32_t w, std::uint32_t h) {
    strix::image::Rgb8 im;
    im.width = w;
    im.height = h;
    im.pixels.resize(std::size_t{w} * h * 3U);
    std::uint32_t s = w * 2654435761U;
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x) {
            s = s * 1103515245U + 12345U;
            auto* p = im.pixels.data() + (std::size_t{y} * w + x) * 3U;
            p[0] = static_cast<std::uint8_t>((7U * x + 3U * y) ^ (s >> 24U));
            p[1] = static_cast<std::uint8_t>(((x * y + 17U * x) >> 3U) & 255U);
            p[2] = static_cast<std::uint8_t>((((x - w / 2U) * (x - w / 2U) + (y - h / 2U) * (y - h / 2U)) < (w * h / 9U)) ? 250U : (x ^ y) * 5U);
        }
    return im;
}

struct Case {
    std::string name;
    strix::image::Rgb8 rgb;
};

std::uint32_t dimension(const std::string& text) {
    const unsigned long n = std::stoul(text);
    if (n == 0U || n > 65535U) throw std::runtime_error("picture dimension out of range: " + text);
    return static_cast<std::uint32_t>(n);
}

// NAME=syn:W:H or NAME=raw:W:H:FILE.
Case parse_case(const std::string& spec) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos || eq == 0U) throw std::runtime_error("bad case " + spec);
    Case c{spec.substr(0, eq), {}};
    if (c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos) throw std::runtime_error("bad case name " + c.name);
    std::vector<std::string> parts;  // kind, W, H[, FILE (the rest, colons and all)]
    std::string rest = spec.substr(eq + 1U);
    for (int k = 0; k < 3; ++k) {
        const auto colon = rest.find(':');
        if (colon == std::string::npos) break;
        parts.push_back(rest.substr(0, colon));
        rest = rest.substr(colon + 1U);
    }
    parts.push_back(rest);
    if (parts.size() == 3U && parts[0] == "syn") {
        c.rgb = picture(dimension(parts[1]), dimension(parts[2]));
    } else if (parts.size() == 4U && parts[0] == "raw") {
        c.rgb.width = dimension(parts[1]);
        c.rgb.height = dimension(parts[2]);
        c.rgb.pixels = read(parts[3]);
        if (c.rgb.pixels.size() != std::size_t{c.rgb.width} * c.rgb.height * 3U) throw std::runtime_error(c.name + ": FILE is not W*H*3 bytes");
    } else {
        throw std::runtime_error("bad case " + spec);
    }
    return c;
}

void write_grid(const fs::path& path, const strix::image::Rgb8& rgb, const v::GridPlan& g) {
    std::ofstream grid(path);
    grid.exceptions(std::ios::failbit | std::ios::badbit);
    grid << "{\n  \"width\": " << rgb.width << ", \"height\": " << rgb.height << ",\n  \"best_w\": " << g.best_w
         << ", \"best_h\": " << g.best_h << ",\n  \"vit_w\": " << g.vit_w << ", \"vit_h\": " << g.vit_h << ", \"n_vit_w\": " << g.vit_w
         << ", \"n_vit_h\": " << g.vit_h << ",\n  \"llm_w\": " << g.llm_w << ", \"llm_h\": " << g.llm_h << ", \"n_llm_w\": " << g.llm_w
         << ", \"n_llm_h\": " << g.llm_h << ",\n  \"patches\": " << g.patches() << ", \"cells\": " << g.cells()
         << ", \"span_tokens\": " << g.span_tokens() << "\n}\n";
    grid.close();
}

void run_case(const v::VisionWeights& w, const Case& cs, const fs::path& root, unsigned threads) {
    const auto img = v::prepare(cs.rgb, w.config);
    const auto& g = img.plan;
    const fs::path dir = root / cs.name;
    fs::create_directories(dir);
    write_grid(dir / "grid.json", cs.rgb, g);
    dump(dir / "rgb.u8", std::span(cs.rgb.pixels));
    dump(dir / "patches.bf16", std::span(img.patches));
    std::vector<float> cos, sin;
    v::rope_tables(g, w.config, cos, sin);
    dump(dir / "cos.f32", std::span<const float>(cos));
    dump(dir / "sin.f32", std::span<const float>(sin));
    std::set<std::string> seen;
    std::cout << cs.name << ": " << cs.rgb.width << "x" << cs.rgb.height << ", " << g.patches() << " patches, " << g.cells() << " cells" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    const auto rows = v::encode_reference(w, img, [&](const char* step, std::uint32_t layer, std::span<const std::uint16_t> values) {
        std::string name = step;
        if (name == "block") {
            if (layer >= w.config.layers) throw std::runtime_error("unexpected block");
            char block[16];
            std::snprintf(block, sizeof(block), "block%02u", layer);
            name = block;
        } else if (name != "embed" && name != "norm" && name != "rows") {
            throw std::runtime_error("unexpected step");
        }
        if (!seen.insert(name).second) throw std::runtime_error("duplicate step");
        const std::size_t expected = name == "rows" ? std::size_t(g.cells()) * w.config.out_dim : std::size_t(g.patches()) * w.config.dim;
        if (values.size() != expected) throw std::runtime_error("step size mismatch");
        dump(dir / (name + ".bf16"), values);
    }, threads);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (seen.size() != w.config.layers + 3U) throw std::runtime_error("missing steps");
    const auto span = v::span_rows(w, g, rows);
    dump(dir / "span.bf16", std::span(span));
    std::cout << cs.name << ": " << seen.size() << " steps and the span written, encode " << seconds << " s" << std::endl;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 5) throw std::runtime_error(kUsage);
    if (!v::host_has_fma()) throw std::runtime_error("the host needs AVX2 and FMA");
    std::vector<Case> cases;
    for (int i = 4; i < argc; ++i) cases.push_back(parse_case(argv[i]));
    const unsigned threads = static_cast<unsigned>(std::stoul(argv[3]));
    const auto w = v::load_vision_sidecar(argv[1]);
    if (w.config.layers != 32 || w.tensors.size() != 266) throw std::runtime_error("unexpected vision model");
    for (const auto& cs : cases) run_case(w, cs, argv[2], threads);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
