#include <cuda_runtime.h>

#include "cuda_safe_call.hpp"
#include "max_pooling_cuda.hpp"


namespace {

__global__ void max_pooling_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                   const std::size_t width, const std::size_t height,
                                   const std::size_t ksize) {
    const auto kernel_start_x = blockIdx.x * ksize;
    const auto kernel_start_y = blockIdx.y * ksize;
    const auto tid = threadIdx.x;
    const auto stride = width * 3;
    const auto y = kernel_start_y + tid;

    std::uint8_t rmax = 0, gmax = 0, bmax = 0;
    if (y < height) {
        for (std::size_t dx = 0; dx < ksize; dx++) {
            const auto x = kernel_start_x + dx;
            if (x < width) {
                rmax = max(rmax, src[y * stride + x * 3 + 0]);
                gmax = max(gmax, src[y * stride + x * 3 + 1]);
                bmax = max(bmax, src[y * stride + x * 3 + 2]);
            }
        }
    }

    extern __shared__ std::uint8_t s_mem[];
    std::uint8_t* s_rmax = s_mem;
    std::uint8_t* s_gmax = s_mem + ksize;
    std::uint8_t* s_bmax = s_mem + ksize * 2;

    s_rmax[tid] = rmax;
    s_gmax[tid] = gmax;
    s_bmax[tid] = bmax;
    __syncthreads();

    for (std::uint32_t delta = 1; delta < ksize; delta <<= 1) {
        std::uint8_t rtmp = 0, gtmp = 0, btmp = 0;
        if (tid + delta < ksize) {
            rtmp = s_rmax[tid + delta];
            gtmp = s_gmax[tid + delta];
            btmp = s_bmax[tid + delta];
        }
        __syncthreads();
        s_rmax[tid] = max(s_rmax[tid], rtmp);
        s_gmax[tid] = max(s_gmax[tid], gtmp);
        s_bmax[tid] = max(s_bmax[tid], btmp);
        __syncthreads();
    }

    rmax = s_rmax[0];
    gmax = s_gmax[0];
    bmax = s_bmax[0];

    if (y < height) {
        for (std::size_t dx = 0; dx < ksize; dx++) {
            const auto x = kernel_start_x + dx;
            if (x < width) {
                dst[y * stride + x * 3 + 0] = rmax;
                dst[y * stride + x * 3 + 1] = gmax;
                dst[y * stride + x * 3 + 2] = bmax;
            }
        }
    }
}

}  // anonymous namespace


namespace cuda {

void max_pooling(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height, const std::size_t ksize) {
    const auto block_dim = dim3{
        static_cast<std::uint32_t>((width + ksize - 1) / ksize),
        static_cast<std::uint32_t>((height + ksize - 1) / ksize)
    };
    const auto thread_dim = dim3{static_cast<std::uint32_t>(ksize)};
    const auto smem_size = ksize * 3 * sizeof(std::uint8_t);

    max_pooling_kernel<<<block_dim, thread_dim, smem_size>>>(src, dst, width, height, ksize);
    CUDASafeCall();
}

}  // namespace cuda
