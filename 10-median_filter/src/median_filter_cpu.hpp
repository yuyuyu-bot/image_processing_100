#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>


namespace cpp {

template <typename ValueType>
void sort(ValueType* const array, const std::size_t len) {
    for (std::size_t i = 0; i < len - 1; i++) {
        for (std::size_t j = len - 1; j > i; j--) {
            if (array[j - 1] > array[j]) {
                const auto tmp = array[j - 1];
                array[j - 1] = array[j];
                array[j] = tmp;
            }
        }
    }
}

void median_filter(const std::uint8_t* const src, std::uint8_t* const dst,
                   const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto k_half = static_cast<int>(ksize >> 1);
    const auto kernel_size = ksize * ksize;
    const auto stride = width * 3;
    const auto r_buffer = std::make_unique<std::uint8_t[]>(kernel_size);
    const auto g_buffer = std::make_unique<std::uint8_t[]>(kernel_size);
    const auto b_buffer = std::make_unique<std::uint8_t[]>(kernel_size);

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            int kidx = 0;
            for (int dy = -k_half; dy <= k_half; dy++) {
                for (int dx = -k_half; dx <= k_half; dx++) {
                    const auto sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
                    const auto sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                    r_buffer[kidx] = src[sy * stride + sx * 3 + 0];
                    g_buffer[kidx] = src[sy * stride + sx * 3 + 1];
                    b_buffer[kidx] = src[sy * stride + sx * 3 + 2];
                    kidx++;
                }
            }

            std::sort(r_buffer.get(), r_buffer.get() + kernel_size);
            std::sort(g_buffer.get(), g_buffer.get() + kernel_size);
            std::sort(b_buffer.get(), b_buffer.get() + kernel_size);
            dst[y * stride + x * 3 + 0] = r_buffer[kernel_size / 2];
            dst[y * stride + x * 3 + 1] = g_buffer[kernel_size / 2];
            dst[y * stride + x * 3 + 2] = b_buffer[kernel_size / 2];
        }
    }
}

}  // namespace cpp
