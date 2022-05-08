#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "NEON_2_SSE.h"


namespace neon {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto vector_size = sizeof(uint8x16x3_t) / sizeof(std::uint8_t);

    auto src_ptr = src;
    auto dst_ptr = dst;

    for (std::size_t i = 0; i < width * height * 3; i += vector_size) {
        const auto&& v_RGB = vld3q_u8(src_ptr);
        const uint8x16_t& v_R = v_RGB.val[0];
        const uint8x16_t& v_G = v_RGB.val[1];
        const uint8x16_t& v_B = v_RGB.val[2];
        const uint8x16_t&& v_vmax = vmaxq_u8(v_R, vmaxq_u8(v_G, v_B));
        const uint8x16_t&& v_vmin = vminq_u8(v_R, vminq_u8(v_G, v_B));

        std::uint8_t a_R[16], a_G[16], a_B[16], a_vmax[16], a_vmin[16];
        std::memcpy(a_R, &v_R, sizeof(uint8x16_t));
        std::memcpy(a_G, &v_G, sizeof(uint8x16_t));
        std::memcpy(a_B, &v_B, sizeof(uint8x16_t));
        std::memcpy(a_vmax, &v_vmax, sizeof(uint8x16_t));
        std::memcpy(a_vmin, &v_vmin, sizeof(uint8x16_t));
        std::uint8_t a_H[16], a_S[16], a_V[16];
        for (int j = 0; j < 16; j++) {
            int H;
            if (a_vmin[j] == a_vmax[j]) {
                H = 0;
            } else if (a_vmin[j] == a_B[j]) {
                H = 60 * (a_G[j] - a_R[j]) / (a_vmax[j] - a_vmin[j]) + 60;
            } else if (a_vmin[j] == a_R[j]) {
                H = 60 * (a_B[j] - a_G[j]) / (a_vmax[j] - a_vmin[j]) + 180;
            } else {
                H = 60 * (a_R[j] - a_B[j]) / (a_vmax[j] - a_vmin[j]) + 300;
            }

            a_H[j] = static_cast<std::uint8_t>(H / 360.f * 255.f);
            a_S[j] = a_vmax[j] - a_vmin[j];
            a_V[j] = a_vmax[j];
        }

        uint8x16_t v_H, v_S, v_V;
        std::memcpy(&v_H, a_H, sizeof(uint8x16_t));
        std::memcpy(&v_S, a_S, sizeof(uint8x16_t));
        std::memcpy(&v_V, a_V, sizeof(uint8x16_t));
        const uint8x16x3_t v_HSV{v_H, v_S, v_V};
        vst3q_u8(dst_ptr, v_HSV);

        src_ptr += vector_size;
        dst_ptr += vector_size;
    }
}

}  // namespace neon
