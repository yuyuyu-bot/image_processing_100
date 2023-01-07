#include <algorithm>
#include <cstddef>
#include <cstdint>


namespace cpp {

void average_pooling(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;

    for (std::size_t y = khalf; y < height + khalf; y += ksize) {
        for (std::size_t x = khalf; x < width + khalf; x += ksize) {
            int rsum = 0, bsum = 0, gsum = 0;
            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    const auto sx = std::min(static_cast<int>(x) + kx, static_cast<int>(width) - 1);
                    const auto sy = std::min(static_cast<int>(y) + ky, static_cast<int>(height) - 1);
                    rsum += src[sy * stride + sx * 3 + 0];
                    gsum += src[sy * stride + sx * 3 + 1];
                    bsum += src[sy * stride + sx * 3 + 2];
                }
            }

            const auto rave = rsum / (ksize * ksize);
            const auto gave = gsum / (ksize * ksize);
            const auto bave = bsum / (ksize * ksize);

            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    const auto sx = std::min(static_cast<int>(x) + kx, static_cast<int>(width) - 1);
                    const auto sy = std::min(static_cast<int>(y) + ky, static_cast<int>(height) - 1);
                    dst[sy * stride + sx * 3 + 0] = rave;
                    dst[sy * stride + sx * 3 + 1] = gave;
                    dst[sy * stride + sx * 3 + 2] = bave;
                }
            }
        }
    }
}

}  // namespace cpp
