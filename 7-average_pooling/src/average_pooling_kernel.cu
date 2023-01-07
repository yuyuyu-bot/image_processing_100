#include <cuda_runtime.h>

#include "cuda_safe_call.hpp"
#include "average_pooling_cuda.hpp"


namespace {

__global__ void average_pooling_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                       const std::size_t width, const std::size_t height,
                                       const std::size_t ksize) {
    const auto kernel_start_x = blockIdx.x * ksize;
    const auto kernel_start_y = blockIdx.y * ksize;
    const auto tid = threadIdx.x;
    const auto stride = width * 3;
    const auto y = min(kernel_start_y + tid, height - 1);

    int rsum = 0, gsum = 0, bsum = 0;
    for (std::size_t dx = 0; dx < ksize; dx++) {
        const auto x = min(kernel_start_x + dx, width - 1);
        rsum += src[y * stride + x * 3 + 0];
        gsum += src[y * stride + x * 3 + 1];
        bsum += src[y * stride + x * 3 + 2];
    }

    extern __shared__ int s_mem[];
    int* s_rsum = s_mem;
    int* s_gsum = s_mem + ksize;
    int* s_bsum = s_mem + ksize * 2;

    s_rsum[tid] = rsum;
    s_gsum[tid] = gsum;
    s_bsum[tid] = bsum;
    __syncthreads();

    for (std::uint32_t delta = 1; delta < ksize; delta <<= 1) {
        int rtmp = 0, gtmp = 0, btmp = 0;
        if (tid + delta < ksize) {
            rtmp = s_rsum[tid + delta];
            gtmp = s_gsum[tid + delta];
            btmp = s_bsum[tid + delta];
        }
        __syncthreads();
        s_rsum[tid] += rtmp;
        s_gsum[tid] += gtmp;
        s_bsum[tid] += btmp;
        __syncthreads();
    }

    rsum = s_rsum[0] / (ksize * ksize);
    gsum = s_gsum[0] / (ksize * ksize);
    bsum = s_bsum[0] / (ksize * ksize);

    if (y < height) {
        for (std::size_t dx = 0; dx < ksize; dx++) {
            const auto x = kernel_start_x + dx;
            if (x < width) {
                dst[y * stride + x * 3 + 0] = rsum;
                dst[y * stride + x * 3 + 1] = gsum;
                dst[y * stride + x * 3 + 2] = bsum;
            }
        }
    }
}

}  // anonymous namespace


namespace cuda {

void average_pooling(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto block_dim = dim3{
        static_cast<std::uint32_t>((width + ksize - 1) / ksize),
        static_cast<std::uint32_t>((height + ksize - 1) / ksize)
    };
    const auto thread_dim = dim3{static_cast<std::uint32_t>(ksize)};
    const auto smem_size = ksize * 3 * sizeof(int);

    average_pooling_kernel<<<block_dim, thread_dim, smem_size>>>(src, dst, width, height, ksize);
    CUDASafeCall();
}

}  // namespace cuda
