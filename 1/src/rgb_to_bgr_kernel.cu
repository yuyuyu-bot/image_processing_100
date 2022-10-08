#include <cuda_runtime.h>

#include "common.hpp"
#include "cuda_safe_call.hpp"
#include "rgb_to_bgr_cuda.hpp"


namespace {

__global__ void rgb_to_bgr_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                  const std::size_t width, const std::size_t height) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    dst[idx * 3 + 0] = src[idx * 3 + 2];
    dst[idx * 3 + 1] = src[idx * 3 + 1];
    dst[idx * 3 + 2] = src[idx * 3 + 0];
}

}  // anonymous namespace


namespace cuda {

void rgb_to_bgr(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto grid_dim = dim3{image_height};
    constexpr auto block_dim = dim3{image_width};
    const auto pixel_per_thread = static_cast<int>(
        ceilf(static_cast<float>(width * height) / (grid_dim.x * block_dim.x))
    );

    rgb_to_bgr_kernel<<<grid_dim, block_dim>>>(src, dst, width, height);
    CUDASafeCall();
}

}  // namespace cuda
