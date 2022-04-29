#include <cuda_runtime.h>
#include "cuda.hpp"


namespace
{

__global__ void rgb_to_gray_kernel(const std::uint8_t* src, std::uint8_t* dst,
                                   const std::size_t width, const std::size_t height) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    constexpr auto r_coeff = 0.2126f;
    constexpr auto g_coeff = 0.7152f;
    constexpr auto b_coeff = 0.0722f;

    dst[idx] = src[idx * 3] * r_coeff + src[idx * 3 + 1] * g_coeff + src[idx * 3 + 2] * b_coeff;
}

}  // anonymous namespace


namespace cuda {

void rgb_to_gray(const std::uint8_t* src, std::uint8_t* dst,
                 const std::size_t width, const std::size_t height) {
    rgb_to_gray_kernel<<<height, width>>>(src, dst, width, height);
}

}
