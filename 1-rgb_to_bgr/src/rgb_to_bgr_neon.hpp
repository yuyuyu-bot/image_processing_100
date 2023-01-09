#include <cstddef>
#include <cstdint>
#include <utility>

#include "NEON_2_SSE.h"


namespace neon {

void rgb_to_bgr(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto vector_size = sizeof(uint8x16_t) / sizeof(std::uint8_t);

    std::size_t i = 0;
    const auto end = width * height - vector_size;
    for (i = 0; i < end; i += vector_size) {
        const auto v_rgb = vld3q_u8(&src[i * 3]);
        const uint8x16x3_t v_bgr{v_rgb.val[2], v_rgb.val[1], v_rgb.val[0]};
        vst3q_u8(&dst[i * 3], v_bgr);
    }
    // remainder
    for (; i < width * height; i++) {
        dst[i * 3 + 0] = src[i * 3 + 2];
        dst[i * 3 + 1] = src[i * 3 + 1];
        dst[i * 3 + 2] = src[i * 3 + 0];
    }
}

}  // namespace neon
