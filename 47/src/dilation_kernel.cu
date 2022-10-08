#include <cassert>
#include <cuda_runtime.h>

#include "common.hpp"
#include "cuda_safe_call.hpp"
#include "dilation_cuda.hpp"


namespace {

__global__ void dilation_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                      const std::size_t width, const std::size_t height,
                                      const std::size_t ksize) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto k_half = static_cast<int>(ksize) >> 1;

    const auto clamp = [](auto v, auto minv, auto maxv) { return max(minv, min(maxv, v)); };

    const auto cx = idx % static_cast<int>(width);
    const auto cy = idx / static_cast<int>(width);

    std::uint8_t max_value = 0;
    for (int dy = -k_half; dy <= k_half; dy++) {
        for (int dx = -k_half; dx <= k_half; dx++) {
            const auto sx = clamp(static_cast<int>(cx) + dx, 0, static_cast<int>(width) - 1);
            const auto sy = clamp(static_cast<int>(cy) + dy, 0, static_cast<int>(height) - 1);
            max_value = max(max_value, src[sy * width + sx]);
        }
    }

    dst[idx] = max_value;
}

}  // anonymous namespace


namespace cuda {

void dilation(const std::uint8_t* const src, std::uint8_t* const dst,
                    const std::size_t width, const std::size_t height, const std::size_t ksize) {
    constexpr auto grid_dim = dim3{image_height};
    constexpr auto block_dim = dim3{image_width};

    dilation_kernel<<<grid_dim, block_dim>>>(src, dst, width, height, ksize);
    CUDASafeCall();
}

}  // namespace cuda
