#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "cpp.hpp"

#include "cuda.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " image_path width height" << std::endl;
        return 0;
    }

    using IMG_T = std::uint8_t;

    const auto width = std::stoul(argv[2]);
    const auto height = std::stoul(argv[3]);
    const Image<IMG_T, 3> src(argv[1], width, height);

    Image<IMG_T, 1> dst_cpu(width, height);
    Image<IMG_T, 1> dst_cuda(width, height);

    {
        const auto duration = measure(100, cpp::rgb_to_gray, src.data(), dst_cpu.data(), width, height);

        std::cout << "durations:" << std::endl;
        std::cout << "\tcpp: " << duration << " [usec]" << std::endl;
    }

    {
        IMG_T* d_src;
        IMG_T* d_dst;
        cudaMalloc((void**)&d_src, width * height * 3 * sizeof(IMG_T));
        cudaMalloc((void**)&d_dst, width * height * sizeof(IMG_T));
        cudaMemcpy(d_src, src.data(), width * height * 3 * sizeof(IMG_T), cudaMemcpyHostToDevice);

        const auto duration = measure(100, cuda::rgb_to_gray, d_src, d_dst, width, height);

        cudaMemcpy(dst_cuda.data(), d_dst, width * height * sizeof(IMG_T), cudaMemcpyDeviceToHost);
        cudaFree((void*)d_src);
        cudaFree((void*)d_dst);

        std::cout << "\tcuda: " << duration << " [usec]" << std::endl;

        // validate output
        auto max_diff = 0;
        for (std::size_t i = 0; i < width * height; i++) {
            const auto expected = static_cast<int>(dst_cpu.data()[i]);
            const auto actual = static_cast<int>(dst_cuda.data()[i]);
            const auto diff = std::abs(expected - actual);
            max_diff = std::max(max_diff, diff);
            if (diff > 0) {
                std::cout << "expected: " << expected << ", actual: " << actual << std::endl;
            }
        }
        if (max_diff > 0) {
            std::cout << "max diff: " << max_diff << std::endl;
        }
    }

    dst_cpu.write("cpp.png");
    dst_cuda.write("cuda.png");

    return 0;
}
