#include <cuda_runtime.h>

#include "common.hpp"
#include "cuda_safe_call.hpp"
#include "rgb_to_gray_cuda.hpp"


namespace {

__global__ void rgb_to_gray_kernel(const std::uint8_t* src, std::uint8_t* dst,
                                   const std::size_t width, const std::size_t height) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    constexpr auto r_coeff = 4899;
    constexpr auto g_coeff = 9617;
    constexpr auto b_coeff = 1868;
    constexpr auto normalize_shift_bits = 14;

    dst[idx] = (src[idx * 3 + 0] * r_coeff +
                src[idx * 3 + 1] * g_coeff +
                src[idx * 3 + 2] * b_coeff) >> normalize_shift_bits;
}

}  // anonymous namespace


namespace cuda {

void rgb_to_gray(const std::uint8_t* src, std::uint8_t* dst,
                 const std::size_t width, const std::size_t height) {
    constexpr auto grid_dim = dim3{image_height};
    constexpr auto block_dim = dim3{image_width};

    rgb_to_gray_kernel<<<grid_dim, block_dim>>>(src, dst, width, height);
    CUDASafeCall();
}

}  // namespace cuda
