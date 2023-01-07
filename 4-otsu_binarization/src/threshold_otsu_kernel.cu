#include <array>
#include <cuda_runtime.h>
#include <numeric>

#include "cuda_safe_call.hpp"
#include "threshold_otsu_cuda.hpp"


namespace {

constexpr auto min_value = std::numeric_limits<std::uint8_t>::min();
constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();


__global__ void construct_histrogram_kernel(
    const std::uint8_t* const img, const std::size_t width, const std::size_t height,
    int* const histogram, const int pixel_per_thread) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto tid = threadIdx.x;

    int histogram_local[256] = { 0 };

    for (int i = 0; i < pixel_per_thread; i++) {
        if (idx * pixel_per_thread + i >= width * height) {
            break;
        }
        const auto value = img[idx * pixel_per_thread + i];
        histogram_local[value]++;
    }

    const auto lane_idx = tid % 32;

    for (int arr_idx = 0; arr_idx < 256; arr_idx++) {
        for (std::uint32_t delta = 1; delta < 32; delta <<= 1) {
            const auto tmp = __shfl_down_sync(0xffffffff, histogram_local[arr_idx], delta);
            histogram_local[arr_idx] += tmp;
        }
    }

    if (lane_idx == 0) {
        for (int i = 0; i < 256; i++) {
            atomicAdd(&histogram[i], histogram_local[i]);
        }
    }
}


__global__ void threshold_otsu_kernel(
    const std::uint8_t* const src, std::uint8_t* const dst,
    const std::size_t width, const std::size_t height, const int* const histogram,
    const int pixel_per_thread) {

    const auto tid = threadIdx.x;

    int histogram_local[256];
    memcpy(histogram_local, histogram, 256 * sizeof(int));

    std::uint64_t value_sum = tid * histogram_local[tid];
    std::uint64_t square_value_sum = tid * value_sum;

    for (std::uint32_t delta = 1; delta < 32; delta <<= 1) {
        value_sum += __shfl_down_sync(0xffffffff, value_sum, delta);
        square_value_sum += __shfl_down_sync(0xffffffff, square_value_sum, delta);
    }

    const auto warp_id = tid / 32;
    const auto lane_id = tid % 32;
    __shared__ std::uint64_t s_value_sum_buffer[8];
    __shared__ std::uint64_t s_square_value_sum_buffer[8];

    if (lane_id == 0) {
        s_value_sum_buffer[warp_id] = value_sum;
        s_square_value_sum_buffer[warp_id] = square_value_sum;
    }
    __syncthreads();

    if (tid < 8) {
        value_sum = s_value_sum_buffer[tid];
        square_value_sum = s_square_value_sum_buffer[tid];
    }
    __syncthreads();

    for (std::uint32_t delta = 4; delta >= 1; delta >>= 1) {
        value_sum += __shfl_down_sync(0xffffffff, value_sum, delta);
        square_value_sum += __shfl_down_sync(0xffffffff, square_value_sum, delta);
    }

    if (tid == 0) {
        s_value_sum_buffer[0] = value_sum;
        s_square_value_sum_buffer[0] = square_value_sum;
    }
    __syncthreads();

    value_sum = s_value_sum_buffer[0];
    square_value_sum = s_square_value_sum_buffer[0];

    int n1 = 0, n2 = static_cast<int>(width * height);
    std::uint64_t sum1 = 0, sum2 = value_sum;
    std::uint64_t square_sum1 = 0, square_sum2 = square_value_sum;

    for (int i = 0; i < tid; i++) {
        n1 += histogram_local[i];
        n2 -= histogram_local[i];

        sum1 += i * histogram_local[i];
        sum2 -= i * histogram_local[i];
        square_sum1 += i * i * histogram_local[i];
        square_sum2 -= i * i * histogram_local[i];
    }

    float S;
    if (n1 > 0 && n2 > 0) {
        const auto mean_all = static_cast<float>(value_sum) / (n1 + n2);
        const auto mean1 = static_cast<float>(sum1) / n1;
        const auto mean2 = static_cast<float>(sum2) / n2;
        const auto dev1 = static_cast<float>(square_sum1) / n1 - mean1 * mean1;
        const auto dev2 = static_cast<float>(square_sum2) / n2 - mean2 * mean2;
        S = (n1 * (mean1 - mean_all) * (mean1 - mean_all) +
             n2 * (mean2 - mean_all) * (mean2 - mean_all)) / (n1 * dev1 + n2 * dev2);
    } else {
        S = 0.f;
    }

    auto S_idx = tid;
    for (std::uint32_t delta = 1; delta < 32; delta <<= 1) {
        const auto next_S = __shfl_down_sync(0xffffffff, S, delta);
        const auto next_S_idx = __shfl_down_sync(0xffffffff, S_idx, delta);
        if (S < next_S) {
            S = next_S;
            S_idx = next_S_idx;
        }
    }

    __shared__ float s_S_buffer[8];
    __shared__ int s_S_idx_buffer[8];

    if (lane_id == 0) {
        s_S_buffer[warp_id] = S;
        s_S_idx_buffer[warp_id] = S_idx;
    }
    __syncthreads();

    auto max_S = 0.f;
    auto max_S_idx = 0;
    for (int i = 0; i < 8; i++) {
        if (max_S < s_S_buffer[i]) {
            max_S = s_S_buffer[i];
            max_S_idx = s_S_idx_buffer[i];
        }
    }
    const auto thresh = max_S_idx;

    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto end = min((idx + 1) * pixel_per_thread , static_cast<int>(width * height));
    for (int i = idx * pixel_per_thread; i < end; i++) {
        dst[i] = src[i] < thresh ? min_value : max_value;
    }
}

}  // anonymous namespace


namespace cuda {

void threshold_otsu(const std::uint8_t* const src, std::uint8_t* const dst,
                    const std::size_t width, const std::size_t height,
                    int* const histogram_buffer) {
    {
        const auto block_dim = dim3{static_cast<std::uint32_t>(height / 64)};
        const auto thread_dim = dim3{64};
        cudaMemset((void*)histogram_buffer, 0, 256 * sizeof(int));

        construct_histrogram_kernel<<<block_dim, thread_dim>>>(src, width, height, histogram_buffer, width);
    }

    {
        constexpr auto block_dim = dim3{32};
        constexpr auto thread_dim = dim3{256};  // must be 256
        constexpr auto num_threads = block_dim.x * thread_dim.x;
        const auto pixel_per_block =
            static_cast<int>(ceilf(static_cast<float>(width * height) / num_threads));

        threshold_otsu_kernel<<<block_dim, thread_dim>>>(src, dst, width, height, histogram_buffer, pixel_per_block);
        CUDASafeCall();
    }
}

}  // namespace cuda
