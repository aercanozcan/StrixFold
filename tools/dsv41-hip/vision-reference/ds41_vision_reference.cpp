// DeepSeek V4.1 vision tower: the host reference of the ViT + aligner (the release's inference/vision.py) with the
// declared arithmetic of docs/models/deepseek-v4.1/vision.md. Every dot product is one FP32 accumulator per output,
// summed with fused multiply-adds in ascending index order from 0; the device kernels reproduce it bit for bit.
// Built with -ffp-contract=off and, on x86-64, -mavx2 -mfma (std::fma is a hardware instruction; consecutive outputs
// vectorise without changing any output's chain). Linked only by tests and tools (strix_ds41_vision_reference);
// callers check cpu_has_fma() first.
#include "strix/ds41_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strix::deepseek41::vision {
namespace {

constexpr std::size_t kKeyBlock = kVisionKeyBlock;

// Sum of one key block's probabilities (unused slots 0): 16 groups of 4 consecutive keys, each ((p0 + p1) + p2) + p3,
// combined by the butterfly g = g + g[c ^ d] for d = 8, 4, 2, 1 (the device's 16-lane shuffle order).
float block_sum(const float* p) {
    float g[16], h[16];
    for (int c = 0; c < 16; ++c) g[c] = ((p[4 * c] + p[4 * c + 1]) + p[4 * c + 2]) + p[4 * c + 3];
    for (int d = 8; d != 0; d /= 2) {
        for (int c = 0; c < 16; ++c) h[c] = g[c] + g[c ^ d];
        for (int c = 0; c < 16; ++c) g[c] = h[c];
    }
    return g[0];
}
static_assert(kKeyBlock == 64, "block_sum assumes 64 keys per block");

// Run fn(begin, end) over [0, n) in contiguous chunks on `threads` workers.
template <class F>
void parallel_rows(std::size_t n, unsigned threads, F&& fn) {
    if (threads <= 1U || n < 2U) {
        fn(std::size_t{0}, n);
        return;
    }
    const std::size_t parts = std::min<std::size_t>(threads, n);
    std::vector<std::thread> pool;
    pool.reserve(parts);
    for (std::size_t p = 0; p < parts; ++p) {
        const std::size_t b = n * p / parts, e = n * (p + 1U) / parts;
        pool.emplace_back([&fn, b, e] { fn(b, e); });
    }
    for (auto& t : pool) t.join();
}

// y[n][o] = bf16(acc + bias[o]) with acc = fma(x[n][k], w[o][k], acc) over k = 0..K-1 (bias optional). x and w BF16.
// The weights are transposed to FP32 [k][o] so that consecutive outputs vectorise; each output's chain is unchanged.
void linear(const std::uint16_t* x, std::size_t rows, std::size_t K, const std::uint16_t* w, const std::uint16_t* bias, std::size_t O,
            std::uint16_t* y, unsigned threads, std::size_t x_stride = 0) {
    if (x_stride == 0U) x_stride = K;
    std::vector<float> wt(K * O);
    parallel_rows(O, threads, [&](std::size_t b, std::size_t e) {
        for (std::size_t o = b; o < e; ++o)
            for (std::size_t k = 0; k < K; ++k) wt[k * O + o] = bf16_float(w[o * K + k]);
    });
    constexpr std::size_t OB = 64;
    parallel_rows(rows, threads, [&](std::size_t rb, std::size_t re) {
        std::vector<float> xr(K);
        float acc[OB];
        for (std::size_t n = rb; n < re; ++n) {
            for (std::size_t k = 0; k < K; ++k) xr[k] = bf16_float(x[n * x_stride + k]);
            for (std::size_t o0 = 0; o0 < O; o0 += OB) {
                const std::size_t ob = std::min(OB, O - o0);
                for (std::size_t j = 0; j < OB; ++j) acc[j] = 0.0F;
                for (std::size_t k = 0; k < K; ++k) {
                    const float a = xr[k];
                    const float* wr = wt.data() + k * O + o0;
                    for (std::size_t j = 0; j < ob; ++j) acc[j] = std::fma(a, wr[j], acc[j]);
                }
                for (std::size_t j = 0; j < ob; ++j) {
                    const float v = bias ? acc[j] + bf16_float(bias[o0 + j]) : acc[j];
                    y[n * O + o0 + j] = bf16_round(v);
                }
            }
        }
    });
}

// RMSNorm per row: ss = wave-order sum of fma(x, x, lane partial), r = 1 / sqrt(ss / D + eps), out = bf16(w * (x * r)).
void rms_norm(const std::uint16_t* x, std::size_t rows, std::size_t D, const std::uint16_t* w, float eps, std::uint16_t* out, unsigned threads) {
    parallel_rows(rows, threads, [&](std::size_t b, std::size_t e) {
        for (std::size_t n = b; n < e; ++n) {
            const std::uint16_t* xr = x + n * D;
            float part[32] = {};
            for (std::size_t k = 0; k < D; ++k) {
                const float v = bf16_float(xr[k]);
                part[k % 32U] = std::fma(v, v, part[k % 32U]);
            }
            const float ss = wave_sum_host(part);
            const float r = vision_inv_root(ss, static_cast<float>(D), eps);
            for (std::size_t k = 0; k < D; ++k) out[n * D + k] = vision_norm(xr[k], r, w[k]);
        }
    });
}

// x = bf16(x + y), both BF16.
void add_residual(std::uint16_t* x, const std::uint16_t* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) x[i] = bf16_round(bf16_float(x[i]) + bf16_float(y[i]));
}

}  // namespace

std::vector<std::uint16_t> encode_reference(const VisionWeights& w, const PreparedImage& img, const StepObserver& observer, unsigned threads) {
    const auto& c = w.config;
    if (w.blob.empty()) throw std::runtime_error("vision weights not loaded");
    if (threads == 0U) threads = std::max(1U, std::thread::hardware_concurrency());
    const std::size_t N = img.plan.patches(), D = c.dim, H = c.heads, HD = c.head_dim(), I = c.inter;
    if (img.patches.size() != N * c.patch_in()) throw std::runtime_error("patch tensor size");
    std::vector<float> cosv, sinv;
    rope_tables(img.plan, c, cosv, sinv);
    const std::size_t half = HD / 2U;

    std::vector<std::uint16_t> x(N * D), h(N * D), qkv(N * 3U * D), attn(N * D), tmp(N * D), gu(N * 2U * I), mid(N * I);
    linear(img.patches.data(), N, c.patch_in(), w.patch_w, w.patch_b, D, x.data(), threads);
    if (observer) observer("embed", 0, x);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const auto& b = w.blocks[l];
        rms_norm(x.data(), N, D, b.norm1, c.eps, h.data(), threads);
        linear(h.data(), N, D, b.wqkv, b.bqkv, 3U * D, qkv.data(), threads);
        // rotary on q and k (rotate-half within each head), rounded to BF16
        parallel_rows(N, threads, [&](std::size_t rb, std::size_t re) {
            for (std::size_t n = rb; n < re; ++n)
                for (std::size_t part = 0; part < 2U; ++part)
                    for (std::size_t hh = 0; hh < H; ++hh) {
                        std::uint16_t* v = qkv.data() + n * 3U * D + part * D + hh * HD;
                        for (std::size_t j = 0; j < half; ++j) {
                            float o1, o2;
                            vision_rotate(bf16_float(v[j]), bf16_float(v[j + half]), cosv[n * half + j], sinv[n * half + j], o1, o2);
                            v[j] = bf16_round(o1);
                            v[j + half] = bf16_round(o2);
                        }
                    }
        });
        // attention per head and query row, keys in blocks of kKeyBlock (online softmax, declared order):
        //   s_j = (sum_d fma(q[d], k_j[d])) * 0.125;  m' = max(m, max_block s);  a = exp(m - m');  p_j = exp(s_j - m');
        //   l = l * a;  l = l + block_sum(p);  o = o * a;  for j in the block ascending: o[d] = fma(p_j, v_j[d], o[d]);
        //   out = bf16(o / l)
        for (std::size_t hh = 0; hh < H; ++hh) {
            std::vector<float> kk(N * HD), vv(N * HD);
            for (std::size_t j = 0; j < N; ++j)
                for (std::size_t d = 0; d < HD; ++d) {
                    kk[j * HD + d] = bf16_float(qkv[j * 3U * D + D + hh * HD + d]);
                    vv[j * HD + d] = bf16_float(qkv[j * 3U * D + 2U * D + hh * HD + d]);
                }
            parallel_rows(N, threads, [&](std::size_t rb, std::size_t re) {
                std::vector<float> q(HD), o(HD), sblk(kKeyBlock);
                for (std::size_t i = rb; i < re; ++i) {
                    for (std::size_t d = 0; d < HD; ++d) q[d] = bf16_float(qkv[i * 3U * D + hh * HD + d]);
                    float m = -__builtin_huge_valf(), lsum = 0.0F;
                    std::fill(o.begin(), o.end(), 0.0F);
                    for (std::size_t j0 = 0; j0 < N; j0 += kKeyBlock) {
                        const std::size_t kn = std::min(kKeyBlock, N - j0);
                        float mb = -__builtin_huge_valf();
                        for (std::size_t j = 0; j < kn; ++j) {
                            const float* kr = kk.data() + (j0 + j) * HD;
                            float acc = 0.0F;
                            for (std::size_t d = 0; d < HD; ++d) acc = std::fma(q[d], kr[d], acc);
                            sblk[j] = acc * 0.125F;
                            mb = std::max(mb, sblk[j]);
                        }
                        const float mn = std::max(m, mb);
                        const float alpha = ds41::declared_exp(m - mn);
                        float pblk[kKeyBlock] = {};
                        for (std::size_t j = 0; j < kn; ++j) pblk[j] = ds41::declared_exp(sblk[j] - mn);
                        lsum = lsum * alpha;
                        lsum = lsum + block_sum(pblk);
                        for (std::size_t d = 0; d < HD; ++d) o[d] = o[d] * alpha;
                        for (std::size_t j = 0; j < kn; ++j) {
                            const float* vr = vv.data() + (j0 + j) * HD;
                            for (std::size_t d = 0; d < HD; ++d) o[d] = std::fma(pblk[j], vr[d], o[d]);
                        }
                        m = mn;
                    }
                    for (std::size_t d = 0; d < HD; ++d) attn[i * D + hh * HD + d] = bf16_round(o[d] / lsum);
                }
            });
        }
        linear(attn.data(), N, D, b.wo, b.bo, D, tmp.data(), threads);
        add_residual(x.data(), tmp.data(), N * D);
        rms_norm(x.data(), N, D, b.norm2, c.eps, h.data(), threads);
        linear(h.data(), N, D, b.w1, nullptr, 2U * I, gu.data(), threads);
        parallel_rows(N, threads, [&](std::size_t rb, std::size_t re) {
            for (std::size_t n = rb; n < re; ++n)
                for (std::size_t k = 0; k < I; ++k) mid[n * I + k] = vision_silu_mul(gu[n * 2U * I + k], gu[n * 2U * I + I + k]);
        });
        linear(mid.data(), N, I, b.w2, nullptr, D, tmp.data(), threads);
        add_residual(x.data(), tmp.data(), N * D);
        if (observer) observer("block", l, x);
    }
    rms_norm(x.data(), N, D, w.norm, c.eps, h.data(), threads);
    if (observer) observer("norm", 0, h);
    // 3 x 3 unfold (zero beyond the grid): feature ch * 9 + ki * 3 + kj of cell (lh, lw) reads patch (3 lh + ki, 3 lw + kj)
    const auto& g = img.plan;
    const std::size_t L = g.cells(), M = c.merge, F = c.merged_in();
    std::vector<std::uint16_t> unf(L * F, 0);
    for (std::size_t lh = 0; lh < g.llm_h; ++lh)
        for (std::size_t lw = 0; lw < g.llm_w; ++lw)
            for (std::size_t ki = 0; ki < M; ++ki)
                for (std::size_t kj = 0; kj < M; ++kj) {
                    const std::size_t py = lh * M + ki, px = lw * M + kj;
                    if (py >= g.vit_h || px >= g.vit_w) continue;
                    const std::uint16_t* src = h.data() + (py * g.vit_w + px) * D;
                    std::uint16_t* dst = unf.data() + (lh * g.llm_w + lw) * F + ki * M + kj;
                    for (std::size_t ch = 0; ch < D; ++ch) dst[ch * M * M] = src[ch];
                }
    std::vector<std::uint16_t> a1(L * c.out_dim), rows(L * c.out_dim);
    linear(unf.data(), L, F, w.a1_w, w.a1_b, c.out_dim, a1.data(), threads);
    for (auto& v : a1) v = bf16_round(vision_gelu(bf16_float(v)));
    linear(a1.data(), L, c.out_dim, w.a2_w, w.a2_b, c.out_dim, rows.data(), threads);
    if (observer) observer("rows", 0, rows);
    return rows;
}

}  // namespace strix::deepseek41::vision
