#include <cstddef>
#include <cstdint>
#include <utility>

#include "NEON_2_SSE.h"


namespace neon {

void rgb_to_bgr(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto vector_size = sizeof(uint8x16x3_t) / sizeof(std::uint8_t);

    auto src_ptr = src;
    auto dst_ptr = dst;

    std::size_t i = 0;
    for (; i + vector_size < width * height * 3; i += vector_size) {
        const auto v_rgb = vld3q_u8(src_ptr);
        const auto&& v_bgr = uint8x16x3_t{v_rgb.val[2], v_rgb.val[1], v_rgb.val[0]};
        vst3q_u8(dst_ptr, v_bgr);

        src_ptr += vector_size;
        dst_ptr += vector_size;
    }

    for (; i < width * height * 3; i += 3) {
        dst_ptr[0] = src_ptr[2];
        dst_ptr[1] = src_ptr[1];
        dst_ptr[2] = src_ptr[0];
        src_ptr += 3;
        dst_ptr += 3;
    }
}

}  // namespace neon
