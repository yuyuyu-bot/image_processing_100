#include <cstddef>
#include <cstdint>
#include <utility>

#include "NEON_2_SSE.h"
#include "neon_utils.hpp"


namespace neon {

void rgb_to_gray(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height) {
    constexpr auto r_coeff = 4899;
    constexpr auto g_coeff = 9617;
    constexpr auto b_coeff = 1868;
    constexpr auto normalize_shift_bits = 14;
    constexpr auto vector_size = sizeof(uint8x8_t) / sizeof(std::uint8_t);

    auto src_ptr = src;
    auto dst_ptr = dst;

    std::size_t i = 0;
    for (; i + vector_size < width * height; i += vector_size) {
        const auto v_rgb = vld3_u8(src_ptr);
        const auto v_r_f32 = u8x8_to_u32x4x2(v_rgb.val[0]);
        const auto v_g_f32 = u8x8_to_u32x4x2(v_rgb.val[1]);
        const auto v_b_f32 = u8x8_to_u32x4x2(v_rgb.val[2]);

        const auto v_rc_u32_l = vmulq_n_u32(v_r_f32.val[0], r_coeff);
        const auto v_rc_u32_h = vmulq_n_u32(v_r_f32.val[1], r_coeff);
        const auto v_gc_u32_l = vmulq_n_u32(v_g_f32.val[0], g_coeff);
        const auto v_gc_u32_h = vmulq_n_u32(v_g_f32.val[1], g_coeff);
        const auto v_bc_u32_l = vmulq_n_u32(v_b_f32.val[0], b_coeff);
        const auto v_bc_u32_h = vmulq_n_u32(v_b_f32.val[1], b_coeff);

        const auto v_gray_u32_l = vaddq_u32(v_rc_u32_l, vaddq_u32(v_gc_u32_l, v_bc_u32_l));
        const auto v_gray_u32_h = vaddq_u32(v_rc_u32_h, vaddq_u32(v_gc_u32_h, v_bc_u32_h));

        const auto v_gray = u32x4x2_to_u8x8(
            uint32x4x2_t{
                vshrq_n_u32(v_gray_u32_l, normalize_shift_bits),
                vshrq_n_u32(v_gray_u32_h, normalize_shift_bits) });
        vst1_u8(dst_ptr, v_gray);

        src_ptr += 3 * vector_size;
        dst_ptr += vector_size;
    }

    for (; i < width * height; i++) {
        *dst_ptr = (r_coeff * src_ptr[0] + g_coeff * src_ptr[1] + b_coeff * src_ptr[2])
                    >> normalize_shift_bits;
        src_ptr += 3;
        dst_ptr++;
    }
}

}  // namespace neon
