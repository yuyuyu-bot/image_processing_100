#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>


namespace cpp {

void mean_filter_naive(const std::uint8_t* const src, std::uint8_t* const dst,
                       const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto k_half = static_cast<int>(ksize >> 1);

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            int Rsum = 0;
            int Gsum = 0;
            int Bsum = 0;
            for (int dy = -k_half; dy <= k_half; dy++) {
                for (int dx = -k_half; dx <= k_half; dx++) {
                    const auto sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
                    const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                    Rsum += src[sy * width * 3 + sx * 3 + 0];
                    Gsum += src[sy * width * 3 + sx * 3 + 1];
                    Bsum += src[sy * width * 3 + sx * 3 + 2];
                }
            }
            dst[y * width * 3 + x * 3 + 0] = static_cast<std::uint8_t>(Rsum / static_cast<float>(ksize * ksize));
            dst[y * width * 3 + x * 3 + 1] = static_cast<std::uint8_t>(Gsum / static_cast<float>(ksize * ksize));
            dst[y * width * 3 + x * 3 + 2] = static_cast<std::uint8_t>(Bsum / static_cast<float>(ksize * ksize));
        }
    }
}


void mean_filter_integral(const std::uint8_t* const src, std::uint8_t* const dst,
                          const std::size_t width, const std::size_t height,
                          const std::size_t ksize) {
    const auto stride = width * 3;
    const auto k_half = static_cast<int>(ksize >> 1);

    using IntegralType = std::uint32_t;
    assert(std::numeric_limits<std::uint8_t>::max() * (width + ksize) * (height * ksize)
            < std::numeric_limits<IntegralType>::max());

    const auto R_integral_image = std::make_unique<IntegralType[]>((width + ksize) * (height * ksize));
    const auto G_integral_image = std::make_unique<IntegralType[]>((width + ksize) * (height * ksize));
    const auto B_integral_image = std::make_unique<IntegralType[]>((width + ksize) * (height * ksize));
    const auto integral_stride = width + ksize;

    const auto get_src = [&src, &stride](const int y, const int x, const int ch) {
            return src + (y * stride + x * 3 + ch);
        };
    const auto get_R_integral
        = [&R_integral_image, &k_half, &integral_stride](const int y, const int x) {
            return &R_integral_image[(y + k_half + 1) * integral_stride + x + k_half + 1];
        };
    const auto get_G_integral
        = [&G_integral_image, &k_half, &integral_stride](const int y, const int x) {
            return &G_integral_image[(y + k_half + 1) * integral_stride + x + k_half + 1];
        };
    const auto get_B_integral
        = [&B_integral_image, &k_half, &integral_stride](const int y, const int x) {
            return &B_integral_image[(y + k_half + 1) * integral_stride + x + k_half + 1];
        };

    for (int y = -k_half; y < static_cast<int>(height) + k_half; y++) {
        const auto src_y = std::clamp(y, 0, static_cast<int>(height) - 1);
        // left cols
        std::fill_n(get_R_integral(y, -k_half), k_half, *get_src(src_y, 0, 0));
        std::fill_n(get_G_integral(y, -k_half), k_half, *get_src(src_y, 0, 1));
        std::fill_n(get_B_integral(y, -k_half), k_half, *get_src(src_y, 0, 2));
        // middle cols
        for (int x = 0; x < static_cast<int>(width); x++) {
            *get_R_integral(y, x) = *get_src(src_y, x, 0);
            *get_G_integral(y, x) = *get_src(src_y, x, 1);
            *get_B_integral(y, x) = *get_src(src_y, x, 2);
        }
        // right cols
        std::fill_n(get_R_integral(y, width), k_half, *get_src(src_y, width - 1, 0));
        std::fill_n(get_G_integral(y, width), k_half, *get_src(src_y, width - 1, 1));
        std::fill_n(get_B_integral(y, width), k_half, *get_src(src_y, width - 1, 2));
    }

    for (int y = -k_half; y < static_cast<int>(height) + k_half; y++) {
        for (int x = -k_half; x < static_cast<int>(width) + k_half; x++) {
            *get_R_integral(y, x) += *get_R_integral(y, x - 1)
                                   + *get_R_integral(y - 1, x)
                                   - *get_R_integral(y - 1, x - 1);
            *get_G_integral(y, x) += *get_G_integral(y, x - 1)
                                   + *get_G_integral(y - 1, x)
                                   - *get_G_integral(y - 1, x - 1);
            *get_B_integral(y, x) += *get_B_integral(y, x - 1)
                                   + *get_B_integral(y - 1, x)
                                   - *get_B_integral(y - 1, x - 1);
        }
    }

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            const auto left = static_cast<int>(x) - k_half;
            const auto top = static_cast<int>(y) - k_half;
            const auto right = static_cast<int>(x) + k_half;
            const auto bottom = static_cast<int>(y) + k_half;

            const auto R_ave = (
                *get_R_integral(bottom, right)
                - *get_R_integral(bottom, left - 1)
                - *get_R_integral(top - 1, right)
                + *get_R_integral(top - 1, left - 1)
                ) / static_cast<float>(ksize * ksize);
            const auto G_ave = (
                *get_G_integral(bottom, right)
                - *get_G_integral(bottom, left - 1)
                - *get_G_integral(top - 1, right)
                + *get_G_integral(top - 1, left - 1)
                ) / static_cast<float>(ksize * ksize);
            const auto B_ave = (
                *get_B_integral(bottom, right)
                - *get_B_integral(bottom, left - 1)
                - *get_B_integral(top - 1, right)
                + *get_B_integral(top - 1, left - 1)
                ) / static_cast<float>(ksize * ksize);

            dst[y * stride + x * 3 + 0] = R_ave;
            dst[y * stride + x * 3 + 1] = G_ave;
            dst[y * stride + x * 3 + 2] = B_ave;
        }
    }
}


void mean_filter_sliding(const std::uint8_t* const src, std::uint8_t* const dst,
                         const std::size_t width, const std::size_t height,
                         const std::size_t ksize) {
    const auto stride = width * 3;
    const auto k_half = static_cast<int>(ksize >> 1);

    for (std::size_t y = 0; y < height; y++) {
        int Rsum = 0, Gsum = 0, Bsum = 0;
        for (int dy = -k_half; dy <= k_half; dy++) {
            for (int dx = -k_half; dx <= k_half; dx++) {
                const auto sx = std::max(dx, 0);
                const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                Rsum += src[sy * stride + sx * 3 + 0];
                Gsum += src[sy * stride + sx * 3 + 1];
                Bsum += src[sy * stride + sx * 3 + 2];
            }
        }

        for (std::size_t x = 0; x < width; x++) {
            dst[y * stride + x * 3 + 0] = static_cast<std::uint8_t>(Rsum / static_cast<float>(ksize * ksize));
            dst[y * stride + x * 3 + 1] = static_cast<std::uint8_t>(Gsum / static_cast<float>(ksize * ksize));
            dst[y * stride + x * 3 + 2] = static_cast<std::uint8_t>(Bsum / static_cast<float>(ksize * ksize));

            const auto sub_x = std::max(static_cast<int>(x) - k_half, 0);
            const auto add_x = std::min(static_cast<int>(x) + k_half + 1, static_cast<int>(width) - 1);

            for (int dy = -k_half; dy <= k_half; dy++) {
                const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                Rsum -= src[sy * stride + sub_x * 3 + 0];
                Gsum -= src[sy * stride + sub_x * 3 + 1];
                Bsum -= src[sy * stride + sub_x * 3 + 2];
                Rsum += src[sy * stride + add_x * 3 + 0];
                Gsum += src[sy * stride + add_x * 3 + 1];
                Bsum += src[sy * stride + add_x * 3 + 2];
            }
        }
    }
}


void mean_filter_separate(const std::uint8_t* const src, std::uint8_t* const dst,
                          const std::size_t width, const std::size_t height,
                          const std::size_t ksize) {
    const auto stride = width * 3;
    const auto k_half = static_cast<int>(ksize >> 1);

    using VerticalBufferType = std::uint16_t;
    assert(std::numeric_limits<std::uint8_t>::max() * ksize
            < std::numeric_limits<VerticalBufferType>::max());

    const auto vertical_buffer = std::make_unique<VerticalBufferType[]>(width * height * 3);

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            VerticalBufferType Rsum = 0, Bsum = 0, Gsum = 0;
            for (int dy = -k_half; dy <= k_half; dy++) {
                const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                Rsum += src[sy * stride + x * 3 + 0];
                Gsum += src[sy * stride + x * 3 + 1];
                Bsum += src[sy * stride + x * 3 + 2];
            }
            vertical_buffer[y * stride + x * 3 + 0] = Rsum;
            vertical_buffer[y * stride + x * 3 + 1] = Gsum;
            vertical_buffer[y * stride + x * 3 + 2] = Bsum;
        }
    }

    for (std::size_t y = 0; y < height; y++) {
        int Rsum = 0, Bsum = 0, Gsum = 0;
        for (int dx = -k_half; dx <= k_half; dx++) {
            const auto sx = std::max(dx, 0);
            Rsum += vertical_buffer[y * stride + sx * 3 + 0];
            Gsum += vertical_buffer[y * stride + sx * 3 + 1];
            Bsum += vertical_buffer[y * stride + sx * 3 + 2];
        }

        for (std::size_t x = 0; x < width; x++) {
            dst[y * stride + x * 3 + 0]
                = static_cast<std::uint16_t>(Rsum / static_cast<float>(ksize * ksize));
            dst[y * stride + x * 3 + 1]
                = static_cast<std::uint16_t>(Gsum / static_cast<float>(ksize * ksize));
            dst[y * stride + x * 3 + 2]
                = static_cast<std::uint16_t>(Bsum / static_cast<float>(ksize * ksize));

            const auto sub_x = std::max(static_cast<int>(x) - k_half, 0);
            const auto add_x = std::min(static_cast<int>(x) + k_half + 1, static_cast<int>(width) - 1);

            Rsum -= vertical_buffer[y * stride + sub_x * 3 + 0];
            Gsum -= vertical_buffer[y * stride + sub_x * 3 + 1];
            Bsum -= vertical_buffer[y * stride + sub_x * 3 + 2];
            Rsum += vertical_buffer[y * stride + add_x * 3 + 0];
            Gsum += vertical_buffer[y * stride + add_x * 3 + 1];
            Bsum += vertical_buffer[y * stride + add_x * 3 + 2];
        }
    }
}

}  // namespace cpp
