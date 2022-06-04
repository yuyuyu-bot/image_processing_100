#include <algorithm>
#include <cstddef>
#include <cstdint>


namespace cpp {

void max_pooling(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;

    for (std::size_t y = khalf; y < height + khalf; y += ksize) {
        for (std::size_t x = khalf; x < width + khalf; x += ksize) {
            std::uint8_t rmax = 0;
            std::uint8_t gmax = 0;
            std::uint8_t bmax = 0;
            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    const auto sx = std::min(static_cast<int>(x) + kx, static_cast<int>(width) - 1);
                    const auto sy = std::min(static_cast<int>(y) + ky, static_cast<int>(height) - 1);
                    rmax = std::max(rmax, src[sy * stride + sx * 3 + 0]);
                    gmax = std::max(gmax, src[sy * stride + sx * 3 + 1]);
                    bmax = std::max(bmax, src[sy * stride + sx * 3 + 2]);
                }
            }

            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    const auto sx = std::min(static_cast<int>(x) + kx, static_cast<int>(width) - 1);
                    const auto sy = std::min(static_cast<int>(y) + ky, static_cast<int>(height) - 1);
                    dst[sy * stride + sx * 3 + 0] = rmax;
                    dst[sy * stride + sx * 3 + 1] = gmax;
                    dst[sy * stride + sx * 3 + 2] = bmax;
                }
            }
        }
    }
}

}  // namespace cpp
