#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "common.hpp"
#include "threshold_cpp.hpp"
#include "threshold_cuda.hpp"
#include "threshold_neon.hpp"


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " image_path width height threshold" << std::endl;
        return 0;
    }

    constexpr auto iteration = 100;

    using IMG_T = std::uint8_t;

    const auto width = std::stoul(argv[2]);
    const auto height = std::stoul(argv[3]);
    const auto thresh = std::stoul(argv[4]);
    const Image<IMG_T, 1> src_img(argv[1], width, height);
    const auto src = src_img.data();

    Image<IMG_T, 1> dst_cpp(width, height);
    Image<IMG_T, 1> dst_neon(width, height);
    Image<IMG_T, 1> dst_cuda(width, height);

    {
        const auto dst = dst_cpp.data();
        const auto duration = measure(iteration, cpp::threshold, src, dst, width, height, thresh);
        std::cout << "durations:" << std::endl;
        std::cout << "\tcpp : " << duration << " [usec]" << std::endl;
    }

    {
        const auto dst = dst_neon.data();
        const auto duration = measure(iteration, neon::threshold, src, dst, width, height, thresh);
        std::cout << "\tneon: " << duration << " [usec]" << std::endl;

        compare_images(dst_cpp, dst_neon);
    }

    {
        IMG_T* d_src;
        IMG_T* d_dst;
        cudaMalloc((void**)&d_src, width * height * sizeof(IMG_T));
        cudaMalloc((void**)&d_dst, width * height * sizeof(IMG_T));
        cudaMemcpy(d_src, src, width * height * sizeof(IMG_T), cudaMemcpyHostToDevice);

        const auto duration =
            measure(iteration, cuda::threshold, d_src, d_dst, width, height, thresh);
        std::cout << "\tcuda: " << duration << " [usec]" << std::endl;

        cudaMemcpy(dst_cuda.data(), d_dst, width * height * sizeof(IMG_T), cudaMemcpyDeviceToHost);
        cudaFree((void*)d_src);
        cudaFree((void*)d_dst);

        compare_images(dst_cpp, dst_cuda);
    }

    dst_cpp.write("cpp.png");
    dst_neon.write("neon.png");
    dst_cuda.write("cuda.png");

    return 0;
}
