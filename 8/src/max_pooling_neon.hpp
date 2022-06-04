#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "NEON_2_SSE.h"


namespace neon {

void max_pooling(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;
    constexpr int vector_size = sizeof(uint8x16_t) / sizeof(std::uint8_t);

    const auto pad_width = (width / ksize + 1) * ksize;
    const auto pad_height = (height / ksize + 1) * ksize;
    const auto pad_stride = pad_width * 3;
    const auto padded_src = std::make_unique<std::uint8_t[]>(pad_width * pad_height * 3);

    for (std::size_t y = 0; y < pad_height; y++) {
        const auto src_y = std::min(y, height - 1);
        std::copy_n(&src[src_y * stride], stride, &padded_src[y * pad_stride]);
        for (std::size_t x = width; x < pad_width; x++) {
            padded_src[y * pad_stride + x * 3 + 0] = src[src_y * stride + (width - 1) * 3 + 0];
            padded_src[y * pad_stride + x * 3 + 1] = src[src_y * stride + (width - 1) * 3 + 1];
            padded_src[y * pad_stride + x * 3 + 2] = src[src_y * stride + (width - 1) * 3 + 2];
        }
    }

    for (std::size_t y = khalf; y < pad_height; y += ksize) {
        for (std::size_t x = khalf; x < pad_width; x += ksize) {
            auto vrmax = vdupq_n_u8(0);
            auto vgmax = vdupq_n_u8(0);
            auto vbmax = vdupq_n_u8(0);
            std::uint8_t rmax = 0, gmax = 0, bmax = 0;
            for (int ky = -khalf; ky <= khalf; ky++) {
                int kx = -khalf;
                for (; kx + vector_size <= khalf; kx += vector_size) {
                    const auto vrgb = vld3q_u8(&padded_src[(y + ky) * pad_stride + (x + kx) * 3]);
                    vrmax = vmaxq_u8(vrmax, vrgb.val[0]);
                    vgmax = vmaxq_u8(vgmax, vrgb.val[1]);
                    vbmax = vmaxq_u8(vbmax, vrgb.val[2]);
                }
                for (; kx <= khalf; kx++) {
                    rmax = std::max(rmax, padded_src[(y + ky) * pad_stride + (x + kx) * 3 + 0]);
                    gmax = std::max(gmax, padded_src[(y + ky) * pad_stride + (x + kx) * 3 + 1]);
                    bmax = std::max(bmax, padded_src[(y + ky) * pad_stride + (x + kx) * 3 + 2]);
                }
            }

            std::uint8_t armax[vector_size] = { 0 };
            std::uint8_t agmax[vector_size] = { 0 };
            std::uint8_t abmax[vector_size] = { 0 };
            vst1q_u8(armax, vrmax);
            vst1q_u8(agmax, vgmax);
            vst1q_u8(abmax, vbmax);
            for (int i = 0; i < vector_size; i++) {
                rmax = std::max(rmax, armax[i]);
                gmax = std::max(gmax, agmax[i]);
                bmax = std::max(bmax, abmax[i]);
            }

            const auto vdst = uint8x16x3_t{ vdupq_n_u8(rmax), vdupq_n_u8(gmax), vdupq_n_u8(bmax) };

            for (int ky = -khalf; ky <= khalf && y + ky < height; ky++) {
                int kx = -khalf;
                for (; kx + vector_size <= khalf && x + kx + vector_size < width; kx += vector_size) {
                    vst3q_u8(&dst[(y + ky) * stride + (x + kx) * 3], vdst);
                }
                for (; kx <= khalf && x + kx < width; kx++) {
                    dst[(y + ky) * stride + (x + kx) * 3 + 0] = rmax;
                    dst[(y + ky) * stride + (x + kx) * 3 + 1] = gmax;
                    dst[(y + ky) * stride + (x + kx) * 3 + 2] = bmax;
                }
            }
        }
    }
}

}  // namespace neon
