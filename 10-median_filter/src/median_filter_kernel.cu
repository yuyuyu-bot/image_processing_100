#include <cassert>
#include <cuda_runtime.h>

#include "cuda_safe_call.hpp"
#include "median_filter_cuda.hpp"


namespace {

constexpr std::size_t KERNEL_SIZE_MAX = 25;

template <typename ValueType>
__device__ ValueType clamp(const ValueType v, const ValueType min_v, const ValueType max_v) {
    return max(min_v, min(max_v, v));
}

template <typename ValueType>
__device__ void sort(ValueType* const array, const std::size_t len) {
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

__global__ void median_filter_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                     const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto x = idx % width;
    const auto y = idx / width;
    const auto stride = width * 3;
    const auto kernel_size = ksize * ksize;

    std::uint8_t r_buffer[KERNEL_SIZE_MAX * KERNEL_SIZE_MAX];
    std::uint8_t g_buffer[KERNEL_SIZE_MAX * KERNEL_SIZE_MAX];
    std::uint8_t b_buffer[KERNEL_SIZE_MAX * KERNEL_SIZE_MAX];

    const auto k_half = static_cast<int>(ksize) >> 1;
    int kidx = 0;
    for (int dy = -k_half; dy <= k_half; dy++) {
        for (int dx = -k_half; dx <= k_half; dx++) {
            const auto sx = clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
            const auto sy = clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
            r_buffer[kidx] = src[sy * stride + sx * 3 + 0];
            g_buffer[kidx] = src[sy * stride + sx * 3 + 1];
            b_buffer[kidx] = src[sy * stride + sx * 3 + 2];
            kidx++;
        }
    }

    sort(r_buffer, kernel_size);
    sort(g_buffer, kernel_size);
    sort(b_buffer, kernel_size);

    dst[idx * 3 + 0] = r_buffer[kernel_size / 2];
    dst[idx * 3 + 1] = g_buffer[kernel_size / 2];
    dst[idx * 3 + 2] = b_buffer[kernel_size / 2];
}

}  // anonymous namespace


namespace cuda {

void median_filter(const std::uint8_t* const src, std::uint8_t* const dst,
                   const std::size_t width, const std::size_t height, const std::size_t ksize) {
    assert(ksize <= KERNEL_SIZE_MAX);
    const auto grid_dim = dim3{static_cast<std::uint32_t>(height)};
    const auto block_dim = dim3{static_cast<std::uint32_t>(width)};

    median_filter_kernel<<<grid_dim, block_dim>>>(src, dst, width, height, ksize);
    CUDASafeCall();
}

}  // namespace cuda
