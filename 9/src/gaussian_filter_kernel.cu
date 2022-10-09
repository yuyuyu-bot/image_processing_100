#include <cuda_runtime.h>
#include <cassert>

#include "cuda_safe_call.hpp"
#include "gaussian_filter_cuda.hpp"


namespace {

__global__ void gaussian_filter_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                       const std::size_t width, const std::size_t height,
                                       const std::size_t ksize, const float sigma) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;

    extern __shared__ float shared_kernel[];
    __shared__ float shared_kernel_sum;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_kernel_sum = 0.f;
        auto kidx = 0;
        for (int ky = -khalf; ky <= khalf; ky++) {
            for (int kx = -khalf; kx <= khalf; kx++) {
                const auto value = exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
                shared_kernel[kidx++] = value;
                shared_kernel_sum += value;
            }
        }
    }
    __syncthreads();

    auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
    auto kidx = 0;
    for (int ky = -khalf; ky <= khalf; ky++) {
        for (int kx = -khalf; kx <= khalf; kx++) {
            const auto sx = max(0, min(x + kx, static_cast<int>(width) - 1));
            const auto sy = max(0, min(y + ky, static_cast<int>(height) - 1));
            rsum += shared_kernel[kidx] * src[stride * sy + sx * 3 + 0];
            gsum += shared_kernel[kidx] * src[stride * sy + sx * 3 + 1];
            bsum += shared_kernel[kidx] * src[stride * sy + sx * 3 + 2];
            kidx++;
        }
    }
    dst[stride * y + x * 3 + 0] = rsum / shared_kernel_sum;
    dst[stride * y + x * 3 + 1] = gsum / shared_kernel_sum;
    dst[stride * y + x * 3 + 2] = bsum / shared_kernel_sum;
}

}  // anonymous namespace


namespace cuda {

void gaussian_filter(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height, const std::size_t ksize, const float sigma) {
    constexpr auto block_dim_x = 16u;
    constexpr auto block_dim_y = 16u;
    assert(width % block_dim_x == 0 && height % block_dim_y == 0);

    const auto grid_dim = dim3{static_cast<uint32_t>(width) / block_dim_x, static_cast<uint32_t>(height) / block_dim_y};
    const auto block_dim = dim3{block_dim_x, block_dim_y};
    const auto smem_size = ksize * ksize * sizeof(float);

    gaussian_filter_kernel<<<grid_dim, block_dim, smem_size>>>(src, dst, width, height, ksize, sigma);
    CUDASafeCall();
}

}  // namespace cuda
