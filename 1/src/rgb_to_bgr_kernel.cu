#include <cuda_runtime.h>

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
    rgb_to_bgr_kernel<<<height, width>>>(src, dst, width, height);
}

}  // namespace cuda
