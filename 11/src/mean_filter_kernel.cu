#include <cuda_runtime.h>

#include "cuda_safe_call.hpp"
#include "mean_filter_cuda.hpp"


namespace {

using AccumulatorType = std::uint32_t;

__device__ int clamp(int a, int b, int c) {
    return max(b, min(a, c));
}

__global__ void mean_filter_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                   const std::size_t width, const std::size_t height,
                                   const std::size_t ksize, const int pixel_per_thread) {
    const int pidx = blockIdx.x * pixel_per_thread;
    const auto kidx = threadIdx.x;

    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;
    const auto ksquare = ksize * ksize;

    extern __shared__ AccumulatorType s_kernel_buffer[];
    AccumulatorType* s_kernel_buffer_r = s_kernel_buffer;
    AccumulatorType* s_kernel_buffer_g = s_kernel_buffer + ksize;
    AccumulatorType* s_kernel_buffer_b = s_kernel_buffer + ksize * 2;

    for (int i = 0; i < pixel_per_thread; i++) {
        if (pidx + i >= width * height) {
            break;
        }

        AccumulatorType local_rsum = 0;
        AccumulatorType local_gsum = 0;
        AccumulatorType local_bsum = 0;

        // kernel center
        const auto cx = (pidx + i) % static_cast<int>(width);
        const auto cy = (pidx + i) / static_cast<int>(width);

        const auto sy = clamp(cy + (kidx - khalf), 0, height - 1);  // y idx in kernel
        for (int d = -khalf; d <= khalf; d++) {
            const auto sx = clamp(cx + d, 0, width - 1);  // x idx in kernel
            local_rsum += src[sy * stride + sx * 3 + 0];
            local_gsum += src[sy * stride + sx * 3 + 1];
            local_bsum += src[sy * stride + sx * 3 + 2];
        }
        s_kernel_buffer_r[kidx] = local_rsum;
        s_kernel_buffer_g[kidx] = local_gsum;
        s_kernel_buffer_b[kidx] = local_bsum;
        __syncthreads();

        for (std::uint32_t delta = 1; delta < ksize; delta <<= 1) {
            if (kidx + delta < ksize) {
                s_kernel_buffer_r[kidx] += s_kernel_buffer_r[kidx + delta];
                s_kernel_buffer_g[kidx] += s_kernel_buffer_g[kidx + delta];
                s_kernel_buffer_b[kidx] += s_kernel_buffer_b[kidx + delta];
            }
            __syncthreads();
        }

        const auto rsum = s_kernel_buffer_r[kidx];
        const auto gsum = s_kernel_buffer_g[kidx];
        const auto bsum = s_kernel_buffer_b[kidx];
        if (kidx == 0) {
            dst[(pidx + i) * 3 + 0] = static_cast<std::uint8_t>(rsum / (ksquare));
            dst[(pidx + i) * 3 + 1] = static_cast<std::uint8_t>(gsum / (ksquare));
            dst[(pidx + i) * 3 + 2] = static_cast<std::uint8_t>(bsum / (ksquare));
        }
        __syncthreads();
    }
}

}  // anonymous namespace


namespace cuda {

void mean_filter(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height, const std::size_t ksize) {
    constexpr auto block_dim = dim3{32};
    const auto thread_dim = dim3{static_cast<std::uint32_t>(ksize)};
    constexpr auto num_threads = block_dim.x;
    const auto pixel_per_block
        = static_cast<int>(ceilf(static_cast<float>(width * height) / num_threads));

    mean_filter_kernel<<<block_dim, thread_dim, ksize * sizeof(AccumulatorType) * 3>>>(
        src, dst, width, height, ksize, pixel_per_block);
    CUDASafeCall();
}

}  // namespace cuda
