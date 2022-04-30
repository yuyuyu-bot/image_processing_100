#include <cuda_runtime.h>

#include "rgb_to_bgr_cuda.hpp"


namespace {

__global__ void rgb_to_bgr_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                  const std::size_t width, const std::size_t height,
                                  const int pixel_per_thread) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    const auto end = min((idx + 1) * pixel_per_thread, static_cast<int>(width * height));
    for (int i = idx * pixel_per_thread; i < end; i++) {
        dst[i * 3 + 0] = src[i * 3 + 2];
        dst[i * 3 + 1] = src[i * 3 + 1];
        dst[i * 3 + 2] = src[i * 3 + 0];
    }
}

}  // anonymous namespace


namespace cuda {

void rgb_to_bgr(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto grid_dim = dim3{16};
    constexpr auto block_dim = dim3{256};
    const auto pixel_per_thread = static_cast<int>(
        ceilf(static_cast<float>(width * height) / (grid_dim.x * block_dim.x))
    );

    rgb_to_bgr_kernel<<<grid_dim, block_dim>>>(src, dst, width, height, pixel_per_thread);
}

}  // namespace cuda
