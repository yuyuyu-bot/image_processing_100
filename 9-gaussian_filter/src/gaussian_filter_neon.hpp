#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <memory>

#include "NEON_2_SSE.h"
#include "neon_utils.hpp"


namespace neon {

void gaussian_filter_separate(const std::uint8_t* const src, std::uint8_t* const dst,
                              const std::size_t width, const std::size_t height,
                              const std::size_t ksize, const float sigma) {
    const auto vector_size = sizeof(uint8x8_t) / sizeof(std::uint8_t);
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;

    const auto kernel = std::make_unique<float[]>(ksize);
    for (int k = -khalf; k <= khalf; k++) {
        kernel[(k + khalf)] = std::exp(-(k * k) / (2 * sigma * sigma));
    }
    auto kernel_sum = 0.f;
    for (std::size_t k1 = 0; k1 < ksize; k1++) {
        for (std::size_t k2 = 0; k2 < ksize; k2++) {
            kernel_sum += kernel[k1] * kernel[k2];
        }
    }

    const auto vertical_sum = std::make_unique<float[]>(width * height * 3);

    for (std::size_t y = 0; y < height; y++) {
        std::size_t x = 0;
        for (; x + vector_size < width; x += vector_size) {
            float32x4x2_t v_rsum{ vdupq_n_f32(0.f), vdupq_n_f32(0.f) };
            float32x4x2_t v_gsum{ vdupq_n_f32(0.f), vdupq_n_f32(0.f) };
            float32x4x2_t v_bsum{ vdupq_n_f32(0.f), vdupq_n_f32(0.f) };
            int kidx = 0;
            for (int k = -khalf; k <= khalf; k++) {
                const auto sy = std::clamp(static_cast<int>(y) + k, 0, static_cast<int>(height) - 1);
                const auto v_rgb_u8x8x3 = vld3_u8(&src[stride * sy + x * 3]);
                const auto v_r_u32x4x2 = u8x8_to_u32x4x2(v_rgb_u8x8x3.val[0]);
                const auto v_g_u32x4x2 = u8x8_to_u32x4x2(v_rgb_u8x8x3.val[1]);
                const auto v_b_u32x4x2 = u8x8_to_u32x4x2(v_rgb_u8x8x3.val[2]);
                const auto v_kernel_value = vdupq_n_f32(kernel[kidx]);

                v_rsum.val[0] = vmlaq_f32(v_rsum.val[0], vcvtq_f32_u32(v_r_u32x4x2.val[0]), v_kernel_value);
                v_rsum.val[1] = vmlaq_f32(v_rsum.val[1], vcvtq_f32_u32(v_r_u32x4x2.val[1]), v_kernel_value);
                v_gsum.val[0] = vmlaq_f32(v_gsum.val[0], vcvtq_f32_u32(v_g_u32x4x2.val[0]), v_kernel_value);
                v_gsum.val[1] = vmlaq_f32(v_gsum.val[1], vcvtq_f32_u32(v_g_u32x4x2.val[1]), v_kernel_value);
                v_bsum.val[0] = vmlaq_f32(v_bsum.val[0], vcvtq_f32_u32(v_b_u32x4x2.val[0]), v_kernel_value);
                v_bsum.val[1] = vmlaq_f32(v_bsum.val[1], vcvtq_f32_u32(v_b_u32x4x2.val[1]), v_kernel_value);
            }

            const float32x4x3_t v_rgb_low{ v_rsum.val[0], v_gsum.val[0], v_bsum.val[0] };
            const float32x4x3_t v_rgb_high{ v_rsum.val[1], v_gsum.val[1], v_bsum.val[1] };
            vst3q_f32(&vertical_sum[stride * y + (x + 0) * 3], v_rgb_low);
            vst3q_f32(&vertical_sum[stride * y + (x + 4) * 3], v_rgb_high);
        }
        // remainder
        for (; x < width; x++) {
            auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
            int kidx = 0;
            for (int k = -khalf; k <= khalf; k++) {
                const auto sy =
                    std::clamp(static_cast<int>(y) + k, 0, static_cast<int>(height) - 1);
                rsum += kernel[kidx] * src[stride * sy + x * 3 + 0];
                gsum += kernel[kidx] * src[stride * sy + x * 3 + 1];
                bsum += kernel[kidx] * src[stride * sy + x * 3 + 2];
                kidx++;
            }
            vertical_sum[stride * y + x * 3 + 0] = rsum;
            vertical_sum[stride * y + x * 3 + 1] = gsum;
            vertical_sum[stride * y + x * 3 + 2] = bsum;
        }
    }

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
            int kidx = 0;
            for (int k = -khalf; k <= khalf; k++) {
                const auto sx =
                    std::clamp(static_cast<int>(x) + k, 0, static_cast<int>(width) - 1);
                rsum += kernel[kidx] * vertical_sum[stride * y + sx * 3 + 0];
                gsum += kernel[kidx] * vertical_sum[stride * y + sx * 3 + 1];
                bsum += kernel[kidx] * vertical_sum[stride * y + sx * 3 + 2];
                kidx++;
            }
            dst[stride * y + x * 3 + 0] = rsum / kernel_sum;
            dst[stride * y + x * 3 + 1] = gsum / kernel_sum;
            dst[stride * y + x * 3 + 2] = bsum / kernel_sum;
        }
    }
}

}  // namespace neon
