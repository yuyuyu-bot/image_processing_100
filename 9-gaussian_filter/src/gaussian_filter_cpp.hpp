#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <memory>


namespace cpp {

void gaussian_filter_naive(const std::uint8_t* const src, std::uint8_t* const dst,
                           const std::size_t width, const std::size_t height,
                           const std::size_t ksize, const float sigma) {
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;

    const auto kernel = std::make_unique<float[]>(ksize * ksize);
    auto kernel_sum = 0.f;
    for (int ky = -khalf; ky <= khalf; ky++) {
        for (int kx = -khalf; kx <= khalf; kx++) {
            kernel[(ky + khalf) * ksize + (kx + khalf)] = std::exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
            kernel_sum += std::exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
        }
    }

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
            int kidx = 0;
            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    const auto sx = std::clamp(static_cast<int>(x) + kx, 0, static_cast<int>(width) - 1);
                    const auto sy = std::clamp(static_cast<int>(y) + ky, 0, static_cast<int>(height) - 1);
                    rsum += kernel[kidx] * src[stride * sy + sx * 3 + 0];
                    gsum += kernel[kidx] * src[stride * sy + sx * 3 + 1];
                    bsum += kernel[kidx] * src[stride * sy + sx * 3 + 2];
                    kidx++;
                }
            }
            dst[stride * y + x * 3 + 0] = rsum / kernel_sum;
            dst[stride * y + x * 3 + 1] = gsum / kernel_sum;
            dst[stride * y + x * 3 + 2] = bsum / kernel_sum;
        }
    }
}


void gaussian_filter_separate(const std::uint8_t* const src, std::uint8_t* const dst,
                              const std::size_t width, const std::size_t height,
                              const std::size_t ksize, const float sigma) {
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;

    const auto kernel = std::make_unique<float[]>(ksize);
    for (int k = -khalf; k <= khalf; k++) {
        kernel[(k + khalf)] = std::exp(-(k * k) / (2 * sigma * sigma));
    }
    auto kernel_sum = 0.f;
    for (std::size_t k1 = 0; k1 < ksize; k1++) {
        for (std::size_t k2 = 0; k2 < ksize; k2++) {
            kernel_sum += kernel[k1] * kernel[k2];
        }
    }

    const auto vertical_sum = std::make_unique<float[]>(width * height * 3);

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
            int kidx = 0;
            for (int k = -khalf; k <= khalf; k++) {
                const auto sy = std::clamp(static_cast<int>(y) + k, 0, static_cast<int>(height) - 1);
                rsum += kernel[kidx] * src[stride * sy + x * 3 + 0];
                gsum += kernel[kidx] * src[stride * sy + x * 3 + 1];
                bsum += kernel[kidx] * src[stride * sy + x * 3 + 2];
                kidx++;
            }
            vertical_sum[stride * y + x * 3 + 0] = rsum;
            vertical_sum[stride * y + x * 3 + 1] = gsum;
            vertical_sum[stride * y + x * 3 + 2] = bsum;
        }
    }

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
            int kidx = 0;
            for (int k = -khalf; k <= khalf; k++) {
                const auto sx = std::clamp(static_cast<int>(x) + k, 0, static_cast<int>(width) - 1);
                rsum += kernel[kidx] * vertical_sum[stride * y + sx * 3 + 0];
                gsum += kernel[kidx] * vertical_sum[stride * y + sx * 3 + 1];
                bsum += kernel[kidx] * vertical_sum[stride * y + sx * 3 + 2];
                kidx++;
            }
            dst[stride * y + x * 3 + 0] = rsum / kernel_sum;
            dst[stride * y + x * 3 + 1] = gsum / kernel_sum;
            dst[stride * y + x * 3 + 2] = bsum / kernel_sum;
        }
    }
}

}  // namespace cpp
