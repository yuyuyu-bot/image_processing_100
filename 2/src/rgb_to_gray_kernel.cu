#include <cuda_runtime.h>

#include "cuda_safe_call.hpp"
#include "rgb_to_gray_cuda.hpp"


namespace {

__global__ void rgb_to_gray_kernel(const std::uint8_t* src, std::uint8_t* dst,
                                   const std::size_t width, const std::size_t height,
                                   const int pixel_per_thread) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    constexpr auto r_coeff = 4899;
    constexpr auto g_coeff = 9617;
    constexpr auto b_coeff = 1868;
    constexpr auto normalize_shift_bits = 14;

    const auto end = min((idx + 1) * pixel_per_thread, static_cast<int>(width * height));
    for (int i = idx * pixel_per_thread; i < end; i++) {
        dst[i] = (src[i * 3] * r_coeff + src[i * 3 + 1] * g_coeff + src[i * 3 + 2] * b_coeff)
                    >> normalize_shift_bits;
    }
}

}  // anonymous namespace


namespace cuda {

void rgb_to_gray(const std::uint8_t* src, std::uint8_t* dst,
                 const std::size_t width, const std::size_t height) {
    constexpr auto grid_dim = dim3{16};
    constexpr auto block_dim = dim3{256};
    const auto pixel_per_thread = static_cast<int>(
        ceilf(static_cast<float>(width * height) / (grid_dim.x * block_dim.x))
    );

    rgb_to_gray_kernel<<<grid_dim, block_dim>>>(src, dst, width, height, pixel_per_thread);
    CUDASafeCall();
}

}  // namespace cuda
