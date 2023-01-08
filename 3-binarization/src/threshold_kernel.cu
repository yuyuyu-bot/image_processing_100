#include <cuda_runtime.h>
#include <numeric>

#include "cuda_safe_call.hpp"
#include "threshold_cuda.hpp"


namespace {

constexpr auto min_value = std::numeric_limits<std::uint8_t>::min();
constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();

__global__ void threshold_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                 const std::size_t width, const std::size_t height,
                                 const std::uint8_t thresh) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    dst[idx] = src[idx] <= thresh ? min_value : max_value;
}

}  // anonymous namespace


namespace cuda {

void threshold(const std::uint8_t* const src, std::uint8_t* const dst,
               const std::size_t width, const std::size_t height, const std::uint8_t thresh) {
    const auto grid_dim = dim3{static_cast<std::uint32_t>(height)};
    const auto block_dim = dim3{static_cast<std::uint32_t>(width)};

    threshold_kernel<<<grid_dim, block_dim>>>(src, dst, width, height, thresh);
    CUDASafeCall();
}

}  // namespace cuda
