#include <array>
#include <cuda_runtime.h>
#include <numeric>

#include "common.hpp"
#include "cuda_safe_call.hpp"
#include "rgb_to_hsv_cuda.hpp"


namespace {

__global__ void rgb_to_hsv_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                  const std::size_t width, const std::size_t height) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    const auto R = src[idx * 3 + 0];
    const auto G = src[idx * 3 + 1];
    const auto B = src[idx * 3 + 2];

    const auto vmax = max(R, max(G, B));
    const auto vmin = min(R, min(G, B));

    int H;
    if (vmin == vmax) {
        H = 0;
    } else if (vmin == B) {
        H = 60 * (G - R) / (vmax - vmin) + 60;
    } else if (vmin == R) {
        H = 60 * (B - G) / (vmax - vmin) + 180;
    } else {
        H = 60 * (R - B) / (vmax - vmin) + 300;
    }

    const auto S = vmax - vmin;
    const auto V = vmax;

    dst[idx * 3 + 0] = static_cast<std::uint8_t>(H / 360.f * 255.f);
    dst[idx * 3 + 1] = S;
    dst[idx * 3 + 2] = V;
}

}  // anonymous namespace


namespace cuda {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto block_dim = dim3{image_height};
    constexpr auto thread_dim = dim3{image_width};

    rgb_to_hsv_kernel<<<block_dim, thread_dim>>>(src, dst, width, height);
    CUDASafeCall();
}

}  // namespace cuda
