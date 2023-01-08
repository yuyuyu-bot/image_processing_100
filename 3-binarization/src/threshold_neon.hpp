#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>

#include "NEON_2_SSE.h"


namespace neon {

void threshold(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height, const std::uint8_t thresh) {
    constexpr auto min_value = std::numeric_limits<std::uint8_t>::min();
    constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();
    constexpr auto vector_size = sizeof(uint8x16_t) / sizeof(std::uint8_t);

    auto src_ptr = src;
    auto dst_ptr = dst;

    std::size_t i = 0;
    for (; i + vector_size < width * height; i += vector_size) {
        const auto v_src = vld1q_u8(src_ptr);
        vst1q_u8(dst_ptr, vcgtq_u8(v_src, vdupq_n_u8(thresh)));

        src_ptr += vector_size;
        dst_ptr += vector_size;
    }

    for (; i < width * height; i++) {
        dst[i] = src[i] <= thresh ? min_value : max_value;
    }
}

}  // namespace neon
