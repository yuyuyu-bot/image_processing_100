#include <cuda_runtime.h>
#include <numeric>

#include "cuda_safe_call.hpp"
#include "threshold_cuda.hpp"


namespace {

constexpr auto min_value = std::numeric_limits<std::uint8_t>::min();
constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();

__global__ void threshold_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                 const std::size_t width, const std::size_t height,
                                 const std::uint8_t thresh, const int pixel_per_thread) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    const auto end = min((idx + 1) * pixel_per_thread , static_cast<int>(width * height));
    for (int i = idx * pixel_per_thread; i < end; i++) {
        dst[i] = src[i] < thresh ? min_value : max_value;
    }
}

}  // anonymous namespace


namespace cuda {

void threshold(const std::uint8_t* const src, std::uint8_t* const dst,
               const std::size_t width, const std::size_t height, const std::uint8_t thresh) {
    constexpr auto grid_dim = dim3{16};
    constexpr auto block_dim = dim3{256};
    const auto pixel_per_thread = static_cast<int>(
        ceilf(static_cast<float>(width * height) / (grid_dim.x * block_dim.x))
    );

    threshold_kernel<<<grid_dim, block_dim>>>(src, dst, width, height, thresh, pixel_per_thread);
    CUDASafeCall();
}

}  // namespace cuda
