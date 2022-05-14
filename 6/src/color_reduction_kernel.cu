#include <array>
#include <cuda_runtime.h>
#include <numeric>

#include "cuda_safe_call.hpp"
#include "color_reduction_cuda.hpp"


namespace {

__global__ void color_reduction_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                       const std::size_t width, const std::size_t height,
                                       const int pixel_per_thread) {
    constexpr auto shr = 6u;
    constexpr auto offset = 1 << (shr - 1);
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    const auto end = min((idx + 1) * pixel_per_thread , static_cast<int>(width * height));
    for (int i = idx * pixel_per_thread; i < end; i++) {
        dst[i * 3 + 0] = ((src[i * 3 + 0] >> shr) << shr) + offset;
        dst[i * 3 + 1] = ((src[i * 3 + 1] >> shr) << shr) + offset;
        dst[i * 3 + 2] = ((src[i * 3 + 2] >> shr) << shr) + offset;
    }
}

}  // anonymous namespace


namespace cuda {

void color_reduction(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height) {
    constexpr auto block_dim = dim3{32};
    constexpr auto thread_dim = dim3{32};
    constexpr auto num_threads = block_dim.x * thread_dim.x;
    const auto pixel_per_block =
        static_cast<int>(ceilf(static_cast<float>(width * height) / num_threads));

    color_reduction_kernel<<<block_dim, thread_dim>>>(src, dst, width, height, pixel_per_block);
    CUDASafeCall();
}

}  // namespace cuda
