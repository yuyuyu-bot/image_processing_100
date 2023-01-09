#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "NEON_2_SSE.h"


namespace neon {

void average_pooling(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;
    constexpr int vector_size = sizeof(uint8x8_t) / sizeof(std::uint8_t);

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
            auto vrsuml = vdupq_n_u32(0);
            auto vrsumh = vdupq_n_u32(0);
            auto vgsuml = vdupq_n_u32(0);
            auto vgsumh = vdupq_n_u32(0);
            auto vbsuml = vdupq_n_u32(0);
            auto vbsumh = vdupq_n_u32(0);
            auto rsum = 0, gsum = 0, bsum = 0;
            for (int ky = -khalf; ky <= khalf; ky++) {
                int kx = -khalf;
                for (; kx + vector_size <= khalf; kx += vector_size) {
                    const auto vrgb = vld3_u8(&padded_src[(y + ky) * pad_stride + (x + kx) * 3]);
                    const auto vr_u16 = vmovl_u8(vrgb.val[0]);
                    const auto vg_u16 = vmovl_u8(vrgb.val[1]);
                    const auto vb_u16 = vmovl_u8(vrgb.val[2]);
                    vrsuml = vaddw_u16(vrsuml, vget_low_u16(vr_u16));
                    vrsumh = vaddw_u16(vrsumh, vget_high_u16(vr_u16));
                    vgsuml = vaddw_u16(vgsuml, vget_low_u16(vg_u16));
                    vgsumh = vaddw_u16(vgsumh, vget_high_u16(vg_u16));
                    vbsuml = vaddw_u16(vbsuml, vget_low_u16(vb_u16));
                    vbsumh = vaddw_u16(vbsumh, vget_high_u16(vb_u16));
                }
                for (; kx <= khalf; kx++) {
                    rsum += padded_src[(y + ky) * pad_stride + (x + kx) * 3 + 0];
                    gsum += padded_src[(y + ky) * pad_stride + (x + kx) * 3 + 1];
                    bsum += padded_src[(y + ky) * pad_stride + (x + kx) * 3 + 2];
                }
            }

            std::uint32_t arsum[vector_size] = { 0 };
            std::uint32_t agsum[vector_size] = { 0 };
            std::uint32_t absum[vector_size] = { 0 };
            vst1q_u32(arsum, vrsuml);
            vst1q_u32(&arsum[vector_size / 2], vrsumh);
            vst1q_u32(agsum, vgsuml);
            vst1q_u32(&agsum[vector_size / 2], vgsumh);
            vst1q_u32(absum, vbsuml);
            vst1q_u32(&absum[vector_size / 2], vbsumh);
            for (int i = 0; i < vector_size; i++) {
                rsum += arsum[i];
                gsum += agsum[i];
                bsum += absum[i];
            }

            const auto rave = static_cast<std::uint8_t>(rsum / (ksize * ksize));
            const auto gave = static_cast<std::uint8_t>(gsum / (ksize * ksize));
            const auto bave = static_cast<std::uint8_t>(bsum / (ksize * ksize));
            const auto vdst = uint8x8x3_t{vdup_n_u8(rave), vdup_n_u8(gave), vdup_n_u8(bave)};

            for (int ky = -khalf; ky <= khalf && y + ky < height; ky++) {
                int kx = -khalf;
                for (; kx + vector_size <= khalf && x + kx + vector_size < width; kx += vector_size) {
                    vst3_u8(&dst[(y + ky) * stride + (x + kx) * 3], vdst);
                }
                for (; kx <= khalf && x + kx < width; kx++) {
                    dst[(y + ky) * stride + (x + kx) * 3 + 0] = rave;
                    dst[(y + ky) * stride + (x + kx) * 3 + 1] = gave;
                    dst[(y + ky) * stride + (x + kx) * 3 + 2] = bave;
                }
            }
        }
    }
}

}  // namespace neon
