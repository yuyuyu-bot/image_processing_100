#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "NEON_2_SSE.h"


namespace neon {

void mean_filter_separate(const std::uint8_t* const src, std::uint8_t* const dst,
                          const std::size_t width, const std::size_t height,
                          const std::size_t ksize) {
    const auto stride = width * 3;
    const auto k_half = static_cast<int>(ksize >> 1);

    using VerticalBufferType = std::uint16_t;
    assert(std::numeric_limits<std::uint8_t>::max() * ksize
            < std::numeric_limits<VerticalBufferType>::max());

    const auto vertical_buffer = std::make_unique<VerticalBufferType[]>(width * height * 3);

    constexpr auto load_src_size = sizeof(uint8x8_t) / sizeof(std::uint8_t);
    for (std::size_t y = 0; y < height; y++) {
        std::size_t x = 0;
        for (; x + load_src_size < width; x += load_src_size) {
            auto Rsum = vdupq_n_u16(0);
            auto Gsum = vdupq_n_u16(0);
            auto Bsum = vdupq_n_u16(0);
            for (int dy = -k_half; dy <= k_half; dy++) {
                const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                const auto vRGB = vld3_u8(&src[sy * stride + x * 3]);
                Rsum = vaddw_u8(Rsum, vRGB.val[0]);
                Gsum = vaddw_u8(Gsum, vRGB.val[1]);
                Bsum = vaddw_u8(Bsum, vRGB.val[2]);
            }
            const auto vDst = uint16x8x3_t{Rsum, Gsum, Bsum};
            vst3q_u16(&vertical_buffer[y * stride + x * 3], vDst);
        }

        for (; x < width; x++) {
            VerticalBufferType Rsum = 0, Bsum = 0, Gsum = 0;
            for (int dy = -k_half; dy <= k_half; dy++) {
                const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                Rsum += src[sy * stride + x * 3 + 0];
                Gsum += src[sy * stride + x * 3 + 1];
                Bsum += src[sy * stride + x * 3 + 2];
            }
            vertical_buffer[y * stride + x * 3 + 0] = Rsum;
            vertical_buffer[y * stride + x * 3 + 1] = Gsum;
            vertical_buffer[y * stride + x * 3 + 2] = Bsum;
        }
    }

    // TODO: NEON horizontal summation
    for (std::size_t y = 0; y < height; y++) {
        int Rsum = 0, Bsum = 0, Gsum = 0;
        for (int dx = -k_half; dx <= k_half; dx++) {
            const auto sx = std::max(dx, 0);
            Rsum += vertical_buffer[y * stride + sx * 3 + 0];
            Gsum += vertical_buffer[y * stride + sx * 3 + 1];
            Bsum += vertical_buffer[y * stride + sx * 3 + 2];
        }

        for (std::size_t x = 0; x < width; x++) {
            dst[y * stride + x * 3 + 0]
                = static_cast<std::uint16_t>(Rsum / static_cast<float>(ksize * ksize));
            dst[y * stride + x * 3 + 1]
                = static_cast<std::uint16_t>(Gsum / static_cast<float>(ksize * ksize));
            dst[y * stride + x * 3 + 2]
                = static_cast<std::uint16_t>(Bsum / static_cast<float>(ksize * ksize));

            const auto sub_x = std::max(static_cast<int>(x) - k_half, 0);
            const auto add_x = std::min(static_cast<int>(x) + k_half + 1, static_cast<int>(width) - 1);

            Rsum -= vertical_buffer[y * stride + sub_x * 3 + 0];
            Gsum -= vertical_buffer[y * stride + sub_x * 3 + 1];
            Bsum -= vertical_buffer[y * stride + sub_x * 3 + 2];
            Rsum += vertical_buffer[y * stride + add_x * 3 + 0];
            Gsum += vertical_buffer[y * stride + add_x * 3 + 1];
            Bsum += vertical_buffer[y * stride + add_x * 3 + 2];
        }
    }
}

}  // namespace neon
