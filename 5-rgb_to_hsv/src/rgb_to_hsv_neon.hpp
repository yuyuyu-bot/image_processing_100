#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "NEON_2_SSE.h"
#include "neon_utils.hpp"


namespace neon {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto vector_size = sizeof(uint8x16x3_t) / sizeof(std::uint8_t);
    const auto v_0_f32 = vdupq_n_f32(0.f);
    const auto v_1_s32 = vdupq_n_s32(1);
    const auto v_360_f32 = vdupq_n_f32(360.f);
    const auto v_255_f32 = vdupq_n_f32(255.f);
    const auto v_0p5_f32 = vdupq_n_f32(0.5f);

    auto src_ptr = src;
    auto dst_ptr = dst;

    std::size_t i = 0;
    for (; i + vector_size < width * height * 3; i += vector_size) {
        const auto&& v_RGB = vld3q_u8(src_ptr);
        const uint8x16_t& v_R = v_RGB.val[0];
        const uint8x16_t& v_G = v_RGB.val[1];
        const uint8x16_t& v_B = v_RGB.val[2];
        const uint8x16_t&& v_vmax = vmaxq_u8(v_R, vmaxq_u8(v_G, v_B));
        const uint8x16_t&& v_vmin = vminq_u8(v_R, vminq_u8(v_G, v_B));

        const uint8x16_t&& v_zero_flag = vceqq_u8(v_vmax, v_vmin);
        const uint8x16_t&& v_R_flag = vceqq_u8(v_vmax, v_R);
        const uint8x16_t&& v_G_flag = vceqq_u8(v_vmax, v_G);

        const uint8x16_t&& v_vrange = vsubq_u8(v_vmax, v_vmin);
        const auto&& v_vrange_u16_l = vmovl_u8(vget_low_u8(v_vrange));
        const auto&& v_vrange_u16_h = vmovl_u8(vget_high_u8(v_vrange));
        const auto&& v_vrange_recp_f32_l_l = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_vrange_u16_l))));
        const auto&& v_vrange_recp_f32_l_h = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_vrange_u16_l))));
        const auto&& v_vrange_recp_f32_h_l = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_vrange_u16_h))));
        const auto&& v_vrange_recp_f32_h_h = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_vrange_u16_h))));

        const auto compute_H =
            [&v_0_f32, &v_1_s32, &v_360_f32, &v_0p5_f32, &v_vrange_recp_f32_l_l, &v_vrange_recp_f32_l_h,
             &v_vrange_recp_f32_h_l, &v_vrange_recp_f32_h_h]
            (const uint8x16_t v1, const uint8x16_t v2, const float offset) {
                const auto&& v1_u16_l = vmovl_u8(vget_low_u8(v1));
                const auto&& v1_u16_h = vmovl_u8(vget_high_u8(v1));
                const auto&& v2_u16_l = vmovl_u8(vget_low_u8(v2));
                const auto&& v2_u16_h = vmovl_u8(vget_high_u8(v2));

                const auto&& diff_s16_l = vmulq_n_s16(vsubq_s16(v1_u16_l, v2_u16_l), 60);
                const auto&& diff_s16_h = vmulq_n_s16(vsubq_s16(v1_u16_h, v2_u16_h), 60);
                const auto&& diff_f32_l_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(diff_s16_l)));
                const auto&& diff_f32_l_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(diff_s16_l)));
                const auto&& diff_f32_h_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(diff_s16_h)));
                const auto&& diff_f32_h_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(diff_s16_h)));

                const auto compute_H_s32 =
                [&v_0_f32, &v_1_s32, &v_360_f32, &v_0p5_f32](const float32x4_t& v1, const float32x4_t& v2, const float32x4_t& v3) {
                    auto&& v_H_f32 = vaddq_f32(vmulq_f32(v1, v2), v3);
                    v_H_f32 = vbslq_f32(vcltq_f32(v_H_f32, v_0_f32), vaddq_f32(v_H_f32, v_360_f32), v_H_f32);
                    const auto&& v_H_s32 = vaddq_s32(vcvtq_s32_f32(v_H_f32), v_1_s32);
                    return vshrq_n_s32(v_H_s32, 1);
                };

                const auto&& v_offset = vdupq_n_f32(offset);
                const auto&& v_H_s32_l_l = compute_H_s32(diff_f32_l_l, v_vrange_recp_f32_l_l, v_offset);
                const auto&& v_H_s32_l_h = compute_H_s32(diff_f32_l_h, v_vrange_recp_f32_l_h, v_offset);
                const auto&& v_H_s32_h_l = compute_H_s32(diff_f32_h_l, v_vrange_recp_f32_h_l, v_offset);
                const auto&& v_H_s32_h_h = compute_H_s32(diff_f32_h_h, v_vrange_recp_f32_h_h, v_offset);

                return s32x4_4_to_u8x16(v_H_s32_l_l, v_H_s32_l_h, v_H_s32_h_l, v_H_s32_h_h);
            };

        const uint8x16_t&& v_zero_value = vdupq_n_u8(0);
        const uint8x16_t&& v_R_value = compute_H(v_G, v_B, 0);
        const uint8x16_t&& v_G_value = compute_H(v_B, v_R, 120);
        const uint8x16_t&& v_B_value = compute_H(v_R, v_G, 240);

        uint8x16_t&& v_H = vbslq_u8(v_zero_flag, v_zero_value, v_B_value);
        v_H = vbslq_u8(v_R_flag, v_R_value, v_H);
        v_H = vbslq_u8(v_G_flag, v_G_value, v_H);

        // compute S
        uint8x16_t v_S;
        {
            const uint32x4x2_t&& v_diff_u32_low  = u8x8_to_u32x4x2(vget_low_u8(v_vrange));
            const uint32x4x2_t&& v_diff_u32_high = u8x8_to_u32x4x2(vget_high_u8(v_vrange));
            float32x4x2_t v_S_f32_low { vcvtq_f32_u32(v_diff_u32_low.val[0]),  vcvtq_f32_u32(v_diff_u32_low.val[1]) };
            float32x4x2_t v_S_f32_high{ vcvtq_f32_u32(v_diff_u32_high.val[0]), vcvtq_f32_u32(v_diff_u32_high.val[1]) };

            const uint32x4x2_t&& v_vmax_u32_low  = u8x8_to_u32x4x2(vget_low_u8(v_vmax));
            const uint32x4x2_t&& v_vmax_u32_high = u8x8_to_u32x4x2(vget_high_u8(v_vmax));
            const float32x4x2_t v_vmax_inv_low{
                vrecpeq_f32(vcvtq_f32_u32(v_vmax_u32_low.val[0])), vrecpeq_f32(vcvtq_f32_u32(v_vmax_u32_low.val[1])) };
            const float32x4x2_t v_vmax_inv_high{
                vrecpeq_f32(vcvtq_f32_u32(v_vmax_u32_high.val[0])), vrecpeq_f32(vcvtq_f32_u32(v_vmax_u32_high.val[1])) };

            v_S_f32_low.val[0]  = vmulq_f32(v_vmax_inv_low.val[0], vmulq_f32(v_255_f32, v_S_f32_low.val[0]));
            v_S_f32_low.val[1]  = vmulq_f32(v_vmax_inv_low.val[1], vmulq_f32(v_255_f32, v_S_f32_low.val[1]));
            v_S_f32_high.val[0] = vmulq_f32(v_vmax_inv_high.val[0], vmulq_f32(v_255_f32, v_S_f32_high.val[0]));
            v_S_f32_high.val[1] = vmulq_f32(v_vmax_inv_high.val[1], vmulq_f32(v_255_f32, v_S_f32_high.val[1]));

            v_S_f32_low.val[0]  = vaddq_f32(v_S_f32_low.val[0], v_0p5_f32);
            v_S_f32_low.val[1]  = vaddq_f32(v_S_f32_low.val[1], v_0p5_f32);
            v_S_f32_high.val[0] = vaddq_f32(v_S_f32_high.val[0], v_0p5_f32);
            v_S_f32_high.val[1] = vaddq_f32(v_S_f32_high.val[1], v_0p5_f32);

            v_S = f32x4_4_to_u8x16(v_S_f32_low.val[0], v_S_f32_low.val[1], v_S_f32_high.val[0], v_S_f32_high.val[1]);
        }

        const uint8x16_t v_V = v_vmax;

        const uint8x16x3_t v_HSV{v_H, v_S, v_V};
        vst3q_u8(dst_ptr, v_HSV);

        src_ptr += vector_size;
        dst_ptr += vector_size;
    }
    // remainder
    for (; i < width * height * 3; i += 3) {
        const auto R = src[i + 0];
        const auto G = src[i + 1];
        const auto B = src[i + 2];

        const auto vmax = std::max({R, G, B});
        const auto vmin = std::min({R, G, B});

        float H;
        if (vmin == vmax) {
            H = 0.f;
        } else if (vmax == R) {
            H = 60.f * (G - B) / (vmax - vmin);
            H = H < 0 ? H + 360.f : H;
        } else if (vmax == G) {
            H = 60.f * (B - R) / (vmax - vmin) + 120.f;
        } else { // vmax == B
            H = 60.f * (R - G) / (vmax - vmin) + 240.f;
        }

        const auto S = vmax != 0 ? 255.f * (vmax - vmin) / vmax : 0;
        const auto V = vmax;

        dst[i + 0] = static_cast<std::uint8_t>(H / 2.f + 0.5f);
        dst[i + 1] = static_cast<std::uint8_t>(S + 0.5f);
        dst[i + 2] = V;
    }
}

}  // namespace neon
