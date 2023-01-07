#include <cstddef>
#include <cstdint>
#include <algorithm>

#include "NEON_2_SSE.h"


namespace neon {

void color_reduction(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height) {
    constexpr auto shr = 6u;
    constexpr auto offset = 1 << (shr - 1);
    constexpr auto vector_size = sizeof(uint8x16_t) / sizeof(std::uint8_t);
    const auto offset_vector = vdupq_n_u8(offset);

    std::size_t i = 0;
    for (; i + vector_size < width * height * 3; i += vector_size) {
        const auto src_vector = vld1q_u8(&src[i]);
        const auto dst_vector
            = vaddq_u8(vshlq_n_u8(vshrq_n_u8(src_vector, shr), shr), offset_vector);
        vst1q_u8(&dst[i], dst_vector);
    }
    for (; i < width * height * 3; i++) {
        dst[i] = ((src[i] >> shr) << shr) + offset;
    }
}

}  // namespace neon
