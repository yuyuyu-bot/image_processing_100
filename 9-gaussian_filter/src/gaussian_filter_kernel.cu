#include <cuda_runtime.h>
#include <cassert>

#include "cuda_safe_call.hpp"
#include "gaussian_filter_cuda.hpp"


namespace {

__device__ float compute_kernel(float* kernel, const std::size_t ksize, const float sigma) {
    for (int ky = threadIdx.y; ky < ksize; ky += blockDim.y) {
        for (int kx = threadIdx.x; kx < ksize; kx += blockDim.x) {
            const int kx_norm = kx - (ksize >> 1);
            const int ky_norm = ky - (ksize >> 1);
            const auto value = exp(-(kx_norm * kx_norm + ky_norm * ky_norm) / (2 * sigma * sigma));
            kernel[ksize * ky + kx] = value;
        }
    }
    __syncthreads();

    auto kernel_sum = 0.f;
    for (int i = 0; i < ksize * ksize; i++) {
        kernel_sum += kernel[i];
    }

    return kernel_sum;
}

__global__ void gaussian_filter_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                       const std::size_t width, const std::size_t height,
                                       const std::size_t ksize, const float sigma) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const auto stride = width * 3;
    const auto khalf = static_cast<int>(ksize) / 2;

    extern __shared__ float shared_kernel[];
    const auto kernel_sum = compute_kernel(shared_kernel, ksize, sigma);

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
    dst[stride * y + x * 3 + 0] = rsum / kernel_sum;
    dst[stride * y + x * 3 + 1] = gsum / kernel_sum;
    dst[stride * y + x * 3 + 2] = bsum / kernel_sum;
}

__global__ void gaussian_filter_shared_kernel(const std::uint8_t* const src, std::uint8_t* const dst,
                                              const std::size_t width, const std::size_t height,
                                              const std::size_t ksize, const float sigma) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int img_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int img_y = blockDim.y * blockIdx.y + threadIdx.y;

    const auto img_stride = width * 3;
    const auto khalf = static_cast<int>(ksize) / 2;

    extern __shared__ float shared_buffer[];
    auto shared_kernel = shared_buffer;
    auto shared_image_bufffer = reinterpret_cast<std::uint8_t*>(&shared_kernel[ksize * ksize]);

    const auto kernel_sum = compute_kernel(shared_kernel, ksize, sigma);

    const int image_buffer_width = ksize - 1 + blockDim.x;
    const int image_buffer_height = ksize - 1 + blockDim.y;
    const int image_buffer_stride = image_buffer_width * 3;
    {
        const auto clamp = [](auto v, auto min_v, auto max_v) { return max(min_v, min(v, max_v)); };
        for (int buffer_y = ty, src_y = img_y - khalf; buffer_y < image_buffer_height; buffer_y += blockDim.y, src_y += blockDim.y) {
            for (int buffer_x = tx, src_x = img_x - khalf; buffer_x < image_buffer_width; buffer_x += blockDim.x, src_x += blockDim.x) {
                const int buffer_idx
                    = image_buffer_stride * buffer_y + buffer_x * 3;
                const int src_idx
                    = img_stride * clamp(src_y, 0, static_cast<int>(height) - 1) + clamp(src_x, 0, static_cast<int>(width) - 1) * 3;
                shared_image_bufffer[buffer_idx + 0] = src[src_idx + 0];
                shared_image_bufffer[buffer_idx + 1] = src[src_idx + 1];
                shared_image_bufffer[buffer_idx + 2] = src[src_idx + 2];
            }
        }
    }
    __syncthreads();

    auto rsum = 0.f, bsum = 0.f, gsum = 0.f;
    auto kidx = 0;
    for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
            rsum += shared_kernel[kidx] * shared_image_bufffer[image_buffer_stride * (ty + ky) + (tx + kx) * 3 + 0];
            gsum += shared_kernel[kidx] * shared_image_bufffer[image_buffer_stride * (ty + ky) + (tx + kx) * 3 + 1];
            bsum += shared_kernel[kidx] * shared_image_bufffer[image_buffer_stride * (ty + ky) + (tx + kx) * 3 + 2];
            kidx++;
        }
    }
    dst[img_stride * img_y + img_x * 3 + 0] = rsum / kernel_sum;
    dst[img_stride * img_y + img_x * 3 + 1] = gsum / kernel_sum;
    dst[img_stride * img_y + img_x * 3 + 2] = bsum / kernel_sum;
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

void gaussian_filter_shared(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height, const std::size_t ksize, const float sigma) {
    constexpr auto block_dim_x = 16u;
    constexpr auto block_dim_y = 16u;
    assert(width % block_dim_x == 0 && height % block_dim_y == 0);

    const auto grid_dim = dim3{static_cast<uint32_t>(width) / block_dim_x, static_cast<uint32_t>(height) / block_dim_y};
    const auto block_dim = dim3{block_dim_x, block_dim_y};
    const auto smem_size =
        ksize * ksize * sizeof(float) + // for kernel
        (ksize - 1 + block_dim_x) * (ksize - 1 + block_dim_y) * 3 * sizeof(std::uint8_t); // for image buffer

    gaussian_filter_shared_kernel<<<grid_dim, block_dim, smem_size>>>(src, dst, width, height, ksize, sigma);
    CUDASafeCall();
}

}  // namespace cuda
