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

    std::size_t i = 0;
    for (; i + vector_size < width * height * 3; i += vector_size) {
        const auto&& v_RGB = vld3q_u8(src_ptr);
        const uint8x16_t& v_R = v_RGB.val[0];
        const uint8x16_t& v_G = v_RGB.val[1];
        const uint8x16_t& v_B = v_RGB.val[2];
        const uint8x16_t&& v_vmax = vmaxq_u8(v_R, vmaxq_u8(v_G, v_B));
        const uint8x16_t&& v_vmin = vminq_u8(v_R, vminq_u8(v_G, v_B));

        const uint8x16_t&& v_zero_flag = vceqq_u8(v_vmax, v_vmin);
        const uint8x16_t&& v_B_flag = vceqq_u8(v_vmin, v_B);
        const uint8x16_t&& v_R_flag = vceqq_u8(v_vmin, v_R);

        const uint8x16_t&& v_vrange = vsubq_u8(v_vmax, v_vmin);
        const auto&& v_vrange_u16_l = vmovl_u8(vget_low_u8(v_vrange));
        const auto&& v_vrange_u16_h = vmovl_u8(vget_high_u8(v_vrange));
        const auto&& v_vrange_recp_f32_l_l =
            vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_vrange_u16_l))));
        const auto&& v_vrange_recp_f32_l_h =
            vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_vrange_u16_l))));
        const auto&& v_vrange_recp_f32_h_l =
            vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_vrange_u16_h))));
        const auto&& v_vrange_recp_f32_h_h =
            vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_vrange_u16_h))));

        const auto compute_H = [
            &v_vrange_recp_f32_l_l, &v_vrange_recp_f32_l_h, &v_vrange_recp_f32_h_l,
            &v_vrange_recp_f32_h_h](const uint8x16_t v1, const uint8x16_t v2, const float offset) {
                const auto&& v1_u16_l = vmovl_u8(vget_low_u8(v1));
                const auto&& v1_u16_h = vmovl_u8(vget_high_u8(v1));
                const auto&& v2_u16_l = vmovl_u8(vget_low_u8(v2));
                const auto&& v2_u16_h = vmovl_u8(vget_high_u8(v2));

                const auto&& numer_s16_l = vmulq_n_s16(vsubq_s16(v1_u16_l, v2_u16_l), 60);
                const auto&& numer_s16_h = vmulq_n_s16(vsubq_s16(v1_u16_h, v2_u16_h), 60);
                const auto&& numer_f32_l_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(numer_s16_l)));
                const auto&& numer_f32_l_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(numer_s16_l)));
                const auto&& numer_f32_h_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(numer_s16_h)));
                const auto&& numer_f32_h_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(numer_s16_h)));

                constexpr auto normalize_term = 255.f / 360.f;
                const auto compute_H_s32 = [](const float32x4_t& v1, const float32x4_t& v2,
                                              const float32x4_t& v3) {
                    return vcvtq_s32_f32(
                        vmulq_n_f32(vaddq_f32(vmulq_f32(v1, v2), v3), normalize_term));
                };

                const auto&& v_offset = vdupq_n_f32(offset);
                const auto&& v_H_s32_l_l = compute_H_s32(numer_f32_l_l, v_vrange_recp_f32_l_l,
                                                         v_offset);
                const auto&& v_H_s32_l_h = compute_H_s32(numer_f32_l_h, v_vrange_recp_f32_l_h,
                                                         v_offset);
                const auto&& v_H_s32_h_l = compute_H_s32(numer_f32_h_l, v_vrange_recp_f32_h_l,
                                                         v_offset);
                const auto&& v_H_s32_h_h = compute_H_s32(numer_f32_h_h, v_vrange_recp_f32_h_h,
                                                         v_offset);

                const auto&& v_H_s16_l = vcombine_s16(vmovn_s32((v_H_s32_l_l)),
                                                      vmovn_s32((v_H_s32_l_h)));
                const auto&& v_H_s16_h = vcombine_s16(vmovn_s32((v_H_s32_h_l)),
                                                      vmovn_s32((v_H_s32_h_h)));

                return vcombine_u8(vmovn_u16(v_H_s16_l), vmovn_u16(v_H_s16_h));
            };

        const uint8x16_t&& v_zero_value = vdupq_n_u8(0);
        const uint8x16_t&& v_B_value = compute_H(v_G, v_R, 60);
        const uint8x16_t&& v_R_value = compute_H(v_B, v_G, 180);
        const uint8x16_t&& v_G_value = compute_H(v_R, v_B, 300);

        uint8x16_t&& v_H = vbslq_u8(v_zero_flag, v_zero_value, v_G_value);
        v_H = vbslq_u8(v_B_flag, v_B_value, v_H);
        v_H = vbslq_u8(v_R_flag, v_R_value, v_H);

        const uint8x16_t v_S = vsubq_u16(v_vmax, v_vmin);
        const uint8x16_t v_V = v_vmax;
        const uint8x16x3_t v_HSV{v_H, v_S, v_V};
        vst3q_u8(dst_ptr, v_HSV);

        src_ptr += vector_size;
        dst_ptr += vector_size;
    }

    for (; i < width * height * 3; i += 3) {
        const auto R = src[i + 0];
        const auto G = src[i + 1];
        const auto B = src[i + 2];

        const auto vmax = std::max({R, G, B});
        const auto vmin = std::min({R, G, B});

        int H;
        if (vmin == vmax) {
            H = 0;
        } else if (vmin == B) {
            H = 60 * (G - R) / (vmax - vmin) + 60;
        } else if (vmin == R) {
            H = 60 * (B - G) / (vmax - vmin) + 180;
        } else {
            H = 60 * (R - B) / (vmax - vmin) + 300;
        }

        const auto S = vmax - vmin;
        const auto V = vmax;

        dst[i + 0] = static_cast<std::uint8_t>(H / 360.f * 255.f);
        dst[i + 1] = S;
        dst[i + 2] = V;
    }
}

}  // namespace neon
