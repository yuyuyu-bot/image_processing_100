#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "gaussian_filter_cpp.hpp"
#include "gaussian_filter_cuda.hpp"


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " image_path width height dump_flag" << std::endl;
        return 0;
    }

    constexpr auto iteration = 10;

    using IMG_T = std::uint8_t;

    const auto width = std::stoul(argv[2]);
    const auto height = std::stoul(argv[3]);
    const auto dump_flag = std::stoi(argv[4]) != 0;
    const Image<IMG_T, 3> src_img(argv[1], width, height);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_cpp_naive(width, height);
    Image<IMG_T, 3> dst_cpp_separate(width, height);
    Image<IMG_T, 3> dst_neon(width, height);
    Image<IMG_T, 3> dst_cuda(width, height);

    constexpr auto ksize = 5;
    constexpr auto sigma = 10.f;
    static_assert(ksize % 2 == 1);

    {
        const auto dst = dst_cpp_naive.data();
        MEASURE(iteration, cpp::gaussian_filter_naive, src, dst, width, height, ksize, sigma);
    }

    {
        const auto dst = dst_cpp_separate.data();
        MEASURE(iteration, cpp::gaussian_filter_separate, src, dst, width, height, ksize, sigma);
        compare_images(dst_cpp_naive, dst_cpp_separate);
    }

    // {
    //     const auto dst = dst_neon.data();
    //     MEASURE(iteration, neon::max_pooling, src, dst, width, height, ksize);
    //     compare_images(dst_cpp_naive, dst_neon);
    // }

    {
        device_buffer<IMG_T> d_src(width * height * 3, src);
        device_buffer<IMG_T> d_dst(width * height * 3);

        MEASURE(iteration, cuda::gaussian_filter, d_src.get(), d_dst.get(), width, height, ksize,
                sigma);

        d_dst.download(dst_cuda.data());
        compare_images(dst_cpp_naive, dst_cuda);
    }

    if (dump_flag) {
        dst_cpp_naive.write("cpp_naive.png");
        dst_cpp_separate.write("cpp_separate.png");
        dst_neon.write("neon.png");
        dst_cuda.write("cuda.png");
    }

    return 0;
}
