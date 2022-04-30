#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "common.hpp"
#include "rgb_to_bgr_cpp.hpp"
#include "rgb_to_bgr_cuda.hpp"
#include "rgb_to_bgr_neon.hpp"


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " image_path width height" << std::endl;
        return 0;
    }

    constexpr auto iteration = 100;

    using IMG_T = std::uint8_t;

    const auto width = std::stoul(argv[2]);
    const auto height = std::stoul(argv[3]);
    const Image<IMG_T, 3> src_img(argv[1], width, height);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_cpp(width, height);
    Image<IMG_T, 3> dst_neon(width, height);
    Image<IMG_T, 3> dst_cuda(width, height);

    {
        const auto dst = dst_cpp.data();
        const auto duration = measure(iteration, cpp::rgb_to_bgr, src, dst, width, height);
        std::cout << "durations:" << std::endl;
        std::cout << "\tcpp : " << duration << " [usec]" << std::endl;
    }

    {
        const auto dst = dst_neon.data();
        const auto duration = measure(iteration, neon::rgb_to_bgr, src, dst, width, height);
        std::cout << "\tneon: " << duration << " [usec]" << std::endl;

        compare_images(dst_cpp, dst_neon);
    }

    {
        IMG_T* d_src;
        IMG_T* d_dst;
        cudaMalloc((void**)&d_src, width * height * 3 * sizeof(IMG_T));
        cudaMalloc((void**)&d_dst, width * height * 3 * sizeof(IMG_T));
        cudaMemcpy(d_src, src, width * height * 3 * sizeof(IMG_T), cudaMemcpyHostToDevice);

        const auto duration = measure(iteration, cuda::rgb_to_bgr, d_src, d_dst, width, height);
        std::cout << "\tcuda: " << duration << " [usec]" << std::endl;

        cudaMemcpy(dst_cuda.data(), d_dst, width * height * 3 * sizeof(IMG_T), cudaMemcpyDeviceToHost);
        cudaFree((void*)d_src);
        cudaFree((void*)d_dst);

        compare_images(dst_cpp, dst_cuda);
    }

    dst_cpp.write("cpp.png");
    dst_neon.write("neon.png");
    dst_cuda.write("cuda.png");

    return 0;
}