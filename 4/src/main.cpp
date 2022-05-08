#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "common.hpp"
#include "threshold_otsu_cpp.hpp"
#include "threshold_otsu_cuda.hpp"
#include "threshold_otsu_neon.hpp"


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " image_path width height threshold" << std::endl;
        return 0;
    }

    constexpr auto iteration = 100;

    using IMG_T = std::uint8_t;

    const auto width = std::stoul(argv[2]);
    const auto height = std::stoul(argv[3]);
    const Image<IMG_T, 1> src_img(argv[1], width, height);
    const auto src = src_img.data();

    Image<IMG_T, 1> dst_cpp(width, height);
    Image<IMG_T, 1> dst_neon(width, height);
    Image<IMG_T, 1> dst_cuda(width, height);

    {
        const auto dst = dst_cpp.data();
        const auto duration =
            measure(iteration, cpp::threshold_otsu, src, dst, width, height);
        std::cout << "durations:" << std::endl;
        std::cout << "\tcpp : " << duration << " [usec]" << std::endl;
    }

    {
        const auto dst = dst_neon.data();
        const auto duration =
            measure(iteration, neon::threshold_otsu, src, dst, width, height);
        std::cout << "\tneon: " << duration << " [usec]" << std::endl;

        compare_images(dst_cpp, dst_neon);
    }

    {
        IMG_T* d_src;
        IMG_T* d_dst;
        int* d_histogram_buffer;
        cudaMalloc((void**)&d_src, width * height * sizeof(IMG_T));
        cudaMalloc((void**)&d_dst, width * height * sizeof(IMG_T));
        cudaMalloc((void**)&d_histogram_buffer, 256 * sizeof(int));
        cudaMemcpy(d_src, src, width * height * sizeof(IMG_T), cudaMemcpyHostToDevice);

        const auto duration =
            measure(iteration, cuda::threshold_otsu, d_src, d_dst, width, height, d_histogram_buffer);
        std::cout << "\tcuda: " << duration << " [usec]" << std::endl;

        cudaMemcpy(dst_cuda.data(), d_dst, width * height * sizeof(IMG_T), cudaMemcpyDeviceToHost);
        cudaFree((void*)d_src);
        cudaFree((void*)d_dst);
        cudaFree((void*)d_histogram_buffer);

        compare_images(dst_cpp, dst_cuda);
    }

    dst_cpp.write("cpp.png");
    dst_neon.write("neon.png");
    dst_cuda.write("cuda.png");

    return 0;
}
