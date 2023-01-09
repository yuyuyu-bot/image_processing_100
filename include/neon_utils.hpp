#ifndef NEON_UTILS_HPP
#define NEON_UTILS_HPP

#include "NEON_2_SSE.h"

namespace {

inline auto u8x8_to_u32x4x2(const uint8x8_t& v_u8x8) {
    const auto&& v_u16x8 = vmovl_u8(v_u8x8);
    return uint32x4x2_t{vmovl_u16(vget_low_u16(v_u16x8)), vmovl_u16(vget_high_u16(v_u16x8))};
}

inline auto u8x8_to_s32x4x2(const uint8x8_t& v_u8x8) {
    const auto&& v_u32x4x2 = u8x8_to_u32x4x2(v_u8x8);
    return int32x4x2_t{vreinterpretq_s32_u32(v_u32x4x2.val[0]), vreinterpretq_s32_u32(v_u32x4x2.val[1])};
}

inline auto u32x4x2_to_u8x8(const uint32x4x2_t& v_u32x4x2) {
    const auto&& v_u16 = vcombine_u16(vmovn_u32(v_u32x4x2.val[0]), vmovn_u32(v_u32x4x2.val[1]));
    return vmovn_u16(v_u16);
}

inline auto s32x4_4_to_u8x16(const int32x4_t& v_s32x4_1, const int32x4_t& v_s32x4_2, const int32x4_t& v_s32x4_3, const int32x4_t& v_s32x4_4) {
    const auto&& v_u16x8_1 = vcombine_u16(vmovn_u32((v_s32x4_1)), vmovn_u32((v_s32x4_2)));
    const auto&& v_u16x8_2 = vcombine_u16(vmovn_u32((v_s32x4_3)), vmovn_u32((v_s32x4_4)));
    return vcombine_u8(vmovn_u16(v_u16x8_1), vmovn_u16(v_u16x8_2));
}

inline auto u32x4_4_to_u8x16(const uint32x4_t& v_u32x4_1, const uint32x4_t& v_u32x4_2, const uint32x4_t& v_u32x4_3, const uint32x4_t& v_u32x4_4) {
    const auto&& v_u16x8_1 = vcombine_u16(vmovn_u32((v_u32x4_1)), vmovn_u32((v_u32x4_2)));
    const auto&& v_u16x8_2 = vcombine_u16(vmovn_u32((v_u32x4_3)), vmovn_u32((v_u32x4_4)));
    return vcombine_u8(vmovn_u16(v_u16x8_1), vmovn_u16(v_u16x8_2));
}

inline auto f32x4_4_to_u8x16(const float32x4_t& v_f32x4_1, const float32x4_t& v_f32x4_2, const float32x4_t& v_f32x4_3, const float32x4_t& v_f32x4_4) {
    const auto&& v_u32x4_1 = vcvtq_u32_f32(v_f32x4_1);
    const auto&& v_u32x4_2 = vcvtq_u32_f32(v_f32x4_2);
    const auto&& v_u32x4_3 = vcvtq_u32_f32(v_f32x4_3);
    const auto&& v_u32x4_4 = vcvtq_u32_f32(v_f32x4_4);

    const auto&& v_u16x8_1 = vcombine_u16(vmovn_u32((v_u32x4_1)), vmovn_u32((v_u32x4_2)));
    const auto&& v_u16x8_2 = vcombine_u16(vmovn_u32((v_u32x4_3)), vmovn_u32((v_u32x4_4)));

    return vcombine_u8(vmovn_u16(v_u16x8_1), vmovn_u16(v_u16x8_2));
}

} // anonymous namespace

#endif // NEON_UTILS_HPP
