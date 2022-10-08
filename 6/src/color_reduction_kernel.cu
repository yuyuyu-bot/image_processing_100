#include <array>
#include <cuda_runtime.h>
#include <numeric>

#include "common.hpp"
#include "cuda_safe_call.hpp"
#include "color_reduction_cuda.hpp"


namespace {

__global__ void color_reduction_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                       const std::size_t width, const std::size_t height) {
    constexpr auto shr = 6u;
    constexpr auto offset = 1 << (shr - 1);
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    dst[idx * 3 + 0] = ((src[idx * 3 + 0] >> shr) << shr) + offset;
    dst[idx * 3 + 1] = ((src[idx * 3 + 1] >> shr) << shr) + offset;
    dst[idx * 3 + 2] = ((src[idx * 3 + 2] >> shr) << shr) + offset;
}

}  // anonymous namespace


namespace cuda {

void color_reduction(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height) {
    constexpr auto block_dim = dim3{image_height};
    constexpr auto thread_dim = dim3{image_width};

    color_reduction_kernel<<<block_dim, thread_dim>>>(src, dst, width, height);
    CUDASafeCall();
}

}  // namespace cuda
