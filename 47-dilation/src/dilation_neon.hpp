#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "NEON_2_SSE.h"


namespace neon {

void dilation(const std::uint8_t* const src, std::uint8_t* const dst,
              const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto vector_size = sizeof(uint8x16_t) / sizeof(uint8_t);
    const auto k_half = static_cast<int>(ksize >> 1);

    for (std::size_t y = 0; y < height; y++) {
        std::size_t x = 0;
        for (; x < static_cast<std::size_t>(k_half); x++) {
            uint8_t max_value = 0;
            for (int dy = -k_half; dy <= k_half; dy++) {
                for (int dx = -k_half; dx <= k_half; dx++) {
                    const auto sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
                    const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                    max_value = std::max(max_value, src[sy * width + sx]);
                }
            }
            dst[y * width + x] = max_value;
        }
        for (; x + vector_size + k_half < width; x += vector_size) {
            uint8x16_t max_values = vdupq_n_u8(0);

            for (int dy = -k_half; dy <= k_half; dy++) {
                for (int dx = -k_half; dx <= k_half; dx++) {
                    const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                    const auto sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
                    const auto pixels = vld1q_u8(&src[sy * width + sx]);
                    max_values = vmaxq_u8(max_values, pixels);
                }
            }

            vst1q_u8(&dst[y * width + x], max_values);
        }
        for (; x < width; x++) {
            uint8_t max_value = 0;
            for (int dy = -k_half; dy <= k_half; dy++) {
                for (int dx = -k_half; dx <= k_half; dx++) {
                    const auto sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
                    const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                    max_value = std::max(max_value, src[sy * width + sx]);
                }
            }
            dst[y * width + x] = max_value;
        }
    }
}

}  // namespace neon
