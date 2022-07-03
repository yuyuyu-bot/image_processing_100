#include <cuda_runtime.h>

#include "cuda_safe_call.hpp"
#include "gaussian_filter_cuda.hpp"


namespace {

__global__ void gaussian_filter_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                       const std::size_t width, const std::size_t height,
                                       const std::size_t ksize, const float sigma,
                                       const int pixel_per_thread) {
    const auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;
    int x = (tidx * pixel_per_thread) % width;
    int y = (tidx * pixel_per_thread) / width;

    extern __shared__ float kernel[];
    if (threadIdx.x == 0) {
        auto kidx = 0;
        for (int ky = -khalf; ky <= khalf; ky++) {
            for (int kx = -khalf; kx <= khalf; kx++) {
                kernel[kidx++] = exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
            }
        }
    }
    __syncthreads();

    auto kernel_sum = 0.f;
    {
        auto kidx = 0;
        for (int ky = -khalf; ky <= khalf; ky++) {
            for (int kx = -khalf; kx <= khalf; kx++) {
                kernel_sum += kernel[kidx++];
            }
        }
    }

    const auto next = [width, height](int& x, int& y) {
        x++;
        if (x == width) {
            x = 0;
            y++;
            return y < height;
        }
        return true;
    };

    for (int i = 0; i < pixel_per_thread; i++) {
        auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
        auto kidx = 0;
        for (int ky = -khalf; ky <= khalf; ky++) {
            for (int kx = -khalf; kx <= khalf; kx++) {
                const auto sx = max(0, min(x + kx, static_cast<int>(width) - 1));
                const auto sy = max(0, min(y + ky, static_cast<int>(height) - 1));
                rsum += kernel[kidx] * src[stride * sy + sx * 3 + 0];
                gsum += kernel[kidx] * src[stride * sy + sx * 3 + 1];
                bsum += kernel[kidx] * src[stride * sy + sx * 3 + 2];
                kidx++;
            }
        }
        dst[stride * y + x * 3 + 0] = rsum / kernel_sum;
        dst[stride * y + x * 3 + 1] = gsum / kernel_sum;
        dst[stride * y + x * 3 + 2] = bsum / kernel_sum;
        if (!next(x, y)) {
            break;
        }
    }
}

}  // anonymous namespace


namespace cuda {

void gaussian_filter(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height, const std::size_t ksize,
                     const float sigma) {
    constexpr auto block_dim = dim3{64};
    constexpr auto thread_dim = dim3{64};
    const auto pixel_per_thread =
        (width * height + block_dim.x * thread_dim.x - 1) / (block_dim.x * thread_dim.x);
    const auto smem_size = ksize * ksize * sizeof(float);

    gaussian_filter_kernel<<<block_dim, thread_dim, smem_size>>>(
        src, dst, width, height, ksize, sigma, pixel_per_thread);
    CUDASafeCall();
}

}  // namespace cuda
