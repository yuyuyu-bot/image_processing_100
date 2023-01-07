#ifndef NEON_UTILS_HPP
#define NEON_UTILS_HPP

#include "NEON_2_SSE.h"

namespace {

inline auto u8x8_to_u32x4x2(const uint8x8_t& v_u8x8) {
    const auto&& v_u16x8 = vmovl_u8(v_u8x8);
    return uint32x4x2_t{ vmovl_u16(vget_low_u16(v_u16x8)), vmovl_u16(vget_high_u16(v_u16x8)) };
}

} // anonymous namespace

#endif // NEON_UTILS_HPP
