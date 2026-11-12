// Vision tower host side: the sidecar's BF16 weights (vision.safetensors), the 2D rotary tables and the span rows.
#include "ds41_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace strix::deepseek41::vision {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("vision sidecar: " + what); }

// The little JSON a safetensors header holds: objects, arrays, strings, non-negative integers (enough for it).
struct Json {
    enum Kind { null, number, string, array, object } kind = null;
    std::uint64_t n = 0;
    std::string s;
    std::vector<Json> a;
    std::map<std::string, Json> o;
};

struct Parser {
    const std::string& t;
    std::size_t i = 0;
    void ws() {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) ++i;
    }
    char peek() {
        ws();
        if (i >= t.size()) fail("truncated header");
        return t[i];
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("header: expected '") + c + "' at byte " + std::to_string(i));
        ++i;
    }
    std::string str() {
        expect('"');
        std::string out;
        while (i < t.size() && t[i] != '"') {
            if (t[i] == '\\') {
                if (++i >= t.size()) break;
                const char e = t[i];
                if (e == 'u') {
                    utf8(out, codepoint());
                    continue;
                }
                out += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e;
            } else {
                out += t[i];
            }
            ++i;
        }
        expect('"');
        return out;
    }
    // the four hex digits after "\\u" (i at the 'u'); leaves i on the last digit
    std::uint32_t hex4() {
        if (i + 4 >= t.size()) fail("header: short \\u escape");
        std::uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = t[++i];
            const int d = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
            if (d < 0) fail("header: bad \\u escape");
            v = v * 16U + static_cast<std::uint32_t>(d);
        }
        return v;
    }
    // a \\u escape's code point, a surrogate pair joined; i ends past the escape
    std::uint32_t codepoint() {
        std::uint32_t cp = hex4();
        ++i;
        if (cp >= 0xD800U && cp <= 0xDBFFU) {
            if (i + 1 >= t.size() || t[i] != '\\' || t[i + 1] != 'u') fail("header: unpaired surrogate");
            ++i;
            const std::uint32_t lo = hex4();
            ++i;
            if (lo < 0xDC00U || lo > 0xDFFFU) fail("header: unpaired surrogate");
            cp = 0x10000U + ((cp - 0xD800U) << 10U) + (lo - 0xDC00U);
        } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
            fail("header: unpaired surrogate");
        }
        return cp;
    }
    static void utf8(std::string& out, std::uint32_t cp) {
        if (cp < 0x80U) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800U) {
            out += static_cast<char>(0xC0U | (cp >> 6U));
            out += static_cast<char>(0x80U | (cp & 0x3FU));
        } else if (cp < 0x10000U) {
            out += static_cast<char>(0xE0U | (cp >> 12U));
            out += static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU));
            out += static_cast<char>(0x80U | (cp & 0x3FU));
        } else {
            out += static_cast<char>(0xF0U | (cp >> 18U));
            out += static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU));
            out += static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU));
            out += static_cast<char>(0x80U | (cp & 0x3FU));
        }
    }
    Json value(int depth) {
        if (depth > 8) fail("header nests too deep");
        Json v;
        const char c = peek();
        if (c == '{') {
            v.kind = Json::object;
            ++i;
            if (peek() == '}') return ++i, v;
            for (;;) {
                std::string k = str();
                expect(':');
                v.o[k] = value(depth + 1);
                if (peek() == '}') return ++i, v;
                expect(',');
            }
        }
        if (c == '[') {
            v.kind = Json::array;
            ++i;
            if (peek() == ']') return ++i, v;
            for (;;) {
                v.a.push_back(value(depth + 1));
                if (peek() == ']') return ++i, v;
                expect(',');
            }
        }
        if (c == '"') {
            v.kind = Json::string;
            v.s = str();
            return v;
        }
        if (c >= '0' && c <= '9') {
            v.kind = Json::number;
            while (i < t.size() && t[i] >= '0' && t[i] <= '9') {
                const auto d = static_cast<std::uint64_t>(t[i++] - '0');
                if (v.n > (UINT64_MAX - d) / 10U) fail("header: integer too large");
                v.n = v.n * 10U + d;
            }
            return v;
        }
        if (t.compare(i, 4, "null") == 0) return i += 4, v;
        fail("header: unexpected byte at " + std::to_string(i));
    }
};

std::vector<std::uint64_t> integers(const Json& v, const std::string& what) {
    if (v.kind != Json::array) fail(what + " is not an array");
    std::vector<std::uint64_t> out;
    for (const auto& x : v.a) {
        if (x.kind != Json::number) fail(what + " holds a non-integer");
        out.push_back(x.n);
    }
    return out;
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

VisionWeights load_vision_sidecar(const std::filesystem::path& dir) {
    const auto path = dir / "vision.safetensors";
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path.string());
    VisionWeights w;
    w.file_bytes = std::filesystem::file_size(path);
    std::uint8_t len_le[8];
    in.read(reinterpret_cast<char*>(len_le), 8);
    if (!in) fail("bad header length");
    std::uint64_t hlen = 0;
    for (int b = 7; b >= 0; --b) hlen = (hlen << 8U) | len_le[b];
    if (!in || hlen > (64ULL << 20U) || 8U + hlen > w.file_bytes) fail("bad header length");
    std::string text(static_cast<std::size_t>(hlen), '\0');
    in.read(text.data(), static_cast<std::streamsize>(hlen));
    if (!in) fail("short header");
    Parser parser{text};
    const Json head = parser.value(0);
    if (head.kind != Json::object) fail("header is not an object");
    const std::uint64_t base = 8U + hlen;
    const auto& c = w.config;
    if (c.dim % c.heads != 0U || c.head_dim() % 4U != 0U) fail("head geometry");

    // expected tensors and shapes (safetensors shape [out, in], as torch's)
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
    if (head.o.size() - head.o.count("__metadata__") != want.size())
        fail("expected " + std::to_string(want.size()) + " tensors, the file has " + std::to_string(head.o.size() - head.o.count("__metadata__")));
    std::size_t total = 0;
    std::vector<std::uint64_t> starts;
    for (const auto& wt : want) {
        const auto it = head.o.find(wt.name);
        if (it == head.o.end() || it->second.kind != Json::object) fail("missing tensor " + wt.name);
        const auto& t = it->second.o;
        const auto dtype = t.find("dtype"), shape = t.find("shape"), offsets = t.find("data_offsets");
        if (dtype == t.end() || dtype->second.s != "BF16") fail(wt.name + " is not BF16");
        const std::vector<std::uint64_t> dims = wt.out ? std::vector<std::uint64_t>{wt.out, wt.in} : std::vector<std::uint64_t>{wt.in};
        if (shape == t.end() || integers(shape->second, wt.name + " shape") != dims) fail(wt.name + " has an unexpected shape");
        const std::size_t elements = static_cast<std::size_t>(wt.out ? wt.in * wt.out : wt.in);
        const auto span = offsets == t.end() ? std::vector<std::uint64_t>{} : integers(offsets->second, wt.name + " data_offsets");
        if (span.size() != 2U || span[1] < span[0] || span[1] - span[0] != elements * 2U) fail(wt.name + " byte size");
        if (span[1] > w.file_bytes - base) fail(wt.name + " lies past the end of the file");
        w.tensors.push_back({wt.name, {total, elements}});
        total += elements;
        starts.push_back(base + span[0]);
    }
    w.blob.resize(total);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        in.seekg(static_cast<std::streamoff>(starts[i]));
        in.read(reinterpret_cast<char*>(w.blob.data() + w.tensors[i].second.offset), static_cast<std::streamsize>(w.tensors[i].second.elements * 2U));
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
