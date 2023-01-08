#include <array>
#include <cuda_runtime.h>
#include <numeric>

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

    float H;
    if (vmin == vmax) {
        H = 0.f;
    } else if (vmax == R) {
        H = 60.f * (G - B) / (vmax - vmin);
        H = H < 0 ? H + 360.f : H;
    } else if (vmax == G) {
        H = 60.f * (B - R) / (vmax - vmin) + 120.f;
    } else { // vmax == B
        H = 60.f * (R - G) / (vmax - vmin) + 240.f;
    }

    const auto S = vmax != 0 ? 255.f * (vmax - vmin) / vmax : 0;
    const auto V = vmax;

    dst[idx * 3 + 0] = static_cast<std::uint8_t>(H / 2.f + 0.5f);
    dst[idx * 3 + 1] = static_cast<std::uint8_t>(S + 0.5f);
    dst[idx * 3 + 2] = V;
}

}  // anonymous namespace


namespace cuda {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    const auto block_dim = dim3{static_cast<std::uint32_t>(height)};
    const auto thread_dim = dim3{static_cast<std::uint32_t>(width)};

    rgb_to_hsv_kernel<<<block_dim, thread_dim>>>(src, dst, width, height);
    CUDASafeCall();
}

}  // namespace cuda
