#include <array>
#include <cuda_runtime.h>
#include <numeric>

#include "cuda_safe_call.hpp"
#include "rgb_to_hsv_cuda.hpp"


namespace {

__global__ void rgb_to_hsv_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                  const std::size_t width, const std::size_t height,
                                  const int pixel_per_thread) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    const auto end = min((idx + 1) * pixel_per_thread , static_cast<int>(width * height));
    for (int i = idx; i < end; i++) {
        const auto R = src[i * 3 + 0];
        const auto G = src[i * 3 + 1];
        const auto B = src[i * 3 + 2];

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

        dst[i * 3 + 0] = static_cast<std::uint8_t>(H / 360.f * 255.f);
        dst[i * 3 + 1] = S;
        dst[i * 3 + 2] = V;
    }
}

}  // anonymous namespace


namespace cuda {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    constexpr auto block_dim = dim3{32};
    constexpr auto thread_dim = dim3{32};
    constexpr auto num_threads = block_dim.x * thread_dim.x;
    const auto pixel_per_block =
        static_cast<int>(ceilf(static_cast<float>(width * height) / num_threads));

    rgb_to_hsv_kernel<<<block_dim, thread_dim>>>(src, dst, width, height, pixel_per_block);
    CUDASafeCall();
}

}  // namespace cuda
