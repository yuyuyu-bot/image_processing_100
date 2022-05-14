#include <cassert>
#include <cuda_runtime.h>

#include "cuda_safe_call.hpp"
#include "mean_filter_cuda.hpp"


namespace {

using AccumulatorType = std::uint32_t;
constexpr auto SUM_BUFFER_SIZE = 8;

__device__ int clamp(int a, int b, int c) {
    return max(b, min(a, c));
}

__global__ void mean_filter_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                   const std::size_t width, const std::size_t height,
                                   const std::size_t ksize, const int pixel_per_thread) {
    const int pidx = blockIdx.x * pixel_per_thread;
    const auto kidx = threadIdx.x;
    const auto lane_idx = kidx % 32;
    const auto warp_idx = kidx / 32;
    const auto num_warps = (blockDim.x + 31) / 32;

    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) >> 1;
    const auto ksquare = ksize * ksize;

    __shared__ AccumulatorType s_rsum_buffer[SUM_BUFFER_SIZE];
    __shared__ AccumulatorType s_gsum_buffer[SUM_BUFFER_SIZE];
    __shared__ AccumulatorType s_bsum_buffer[SUM_BUFFER_SIZE];

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

        // reduction in warp
        const auto warp_size = min(static_cast<unsigned>(ksize), 32u);
        for (std::uint32_t delta = 1; delta < warp_size; delta <<= 1) {
            const auto next_rsum = __shfl_down_sync(0xffffffff, local_rsum, delta);
            const auto next_gsum = __shfl_down_sync(0xffffffff, local_gsum, delta);
            const auto next_bsum = __shfl_down_sync(0xffffffff, local_bsum, delta);
            if (lane_idx + delta < warp_size) {
                local_rsum += next_rsum;
                local_gsum += next_gsum;
                local_bsum += next_bsum;
            }
        }

        if (blockDim.x < 32) {
            if (kidx == 0) {
                dst[(pidx + i) * 3 + 0] = static_cast<std::uint8_t>(local_rsum / ksquare);
                dst[(pidx + i) * 3 + 1] = static_cast<std::uint8_t>(local_gsum / ksquare);
                dst[(pidx + i) * 3 + 2] = static_cast<std::uint8_t>(local_bsum / ksquare);
            }
        } else {
            if (lane_idx == 0) {
                s_rsum_buffer[warp_idx] = local_rsum;
                s_gsum_buffer[warp_idx] = local_gsum;
                s_bsum_buffer[warp_idx] = local_bsum;
                __syncthreads();

                for (std::uint32_t delta = 1; delta < num_warps; delta <<= 1) {
                    if (warp_idx + delta < num_warps) {
                        s_rsum_buffer[warp_idx] += s_rsum_buffer[warp_idx + delta];
                        s_gsum_buffer[warp_idx] += s_gsum_buffer[warp_idx + delta];
                        s_bsum_buffer[warp_idx] += s_bsum_buffer[warp_idx + delta];
                    }
                    __syncthreads();
                }
            }
            if (kidx == 0) {
                dst[(pidx + i) * 3 + 0] = static_cast<std::uint8_t>(s_rsum_buffer[0] / ksquare);
                dst[(pidx + i) * 3 + 1] = static_cast<std::uint8_t>(s_gsum_buffer[0] / ksquare);
                dst[(pidx + i) * 3 + 2] = static_cast<std::uint8_t>(s_bsum_buffer[0] / ksquare);
            }
        }

        __syncthreads();
    }
}

}  // anonymous namespace


namespace cuda {

void mean_filter(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height, const std::size_t ksize) {
    assert(ksize < 32 * SUM_BUFFER_SIZE);

    constexpr auto block_dim = dim3{32};
    const auto thread_dim = dim3{static_cast<std::uint32_t>(ksize)};
    constexpr auto num_threads = block_dim.x;
    const auto pixel_per_block
        = static_cast<int>(ceilf(static_cast<float>(width * height) / num_threads));

    mean_filter_kernel<<<block_dim, thread_dim>>>(src, dst, width, height, ksize, pixel_per_block);
    CUDASafeCall();
}

}  // namespace cuda
