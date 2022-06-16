#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <memory>

#include "NEON_2_SSE.h"


namespace neon {

void gaussian_filter_separate(const std::uint8_t* const src, std::uint8_t* const dst,
                              const std::size_t width, const std::size_t height,
                              const std::size_t ksize, const float sigma) {
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;
    constexpr auto vector_size = sizeof(uint8x16_t);

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
            float32x4x4_t vrsum, vgsum, vbsum;
            for (int i = 0; i < 4; i++) {
                vrsum.val[i] = vdupq_n_f32(0);
                vgsum.val[i] = vdupq_n_f32(0);
                vbsum.val[i] = vdupq_n_f32(0);
            }

            int kidx = 0;
            for (int k = -khalf; k <= khalf; k++) {
                const auto sy =
                    std::clamp(static_cast<int>(y) + k, 0, static_cast<int>(height) - 1);
                const auto vrgb = vld3q_u8(&src[stride * sy + x * 3]);
                uint16x8x2_t vr_u16, vg_u16, vb_u16;
                vr_u16.val[0] = vmovl_u8(vget_high_u8(vrgb.val[0]));
                vr_u16.val[1] = vmovl_u8(vget_low_u8(vrgb.val[0]));

                const auto vkernel = vdupq_n_f32(kernel[kidx]);

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
