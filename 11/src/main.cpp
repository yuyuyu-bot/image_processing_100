#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "mean_filter_cpu.hpp"
#include "mean_filter_cuda.hpp"
#include "mean_filter_neon.hpp"


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
    Image<IMG_T, 3> dst_cpp_integral(width, height);
    Image<IMG_T, 3> dst_cpp_sliding(width, height);
    Image<IMG_T, 3> dst_cpp_separate(width, height);
    Image<IMG_T, 3> dst_neon_separate(width, height);
    Image<IMG_T, 3> dst_cuda(width, height);

    constexpr auto ksize = 15;
    static_assert(ksize % 2 == 1);

    std::cout << "durations:" << std::endl;
    {
        const auto dst = dst_cpp_naive.data();
        const auto duration
            = measure(iteration, cpp::mean_filter_naive, src, dst, width, height, ksize);
        std::cout << "\tcpp naive    : " << duration << " [usec]" << std::endl;
    }

    {
        const auto dst = dst_cpp_integral.data();
        const auto duration
            = measure(iteration, cpp::mean_filter_integral, src, dst, width, height, ksize);
        std::cout << "\tcpp integral : " << duration << " [usec]" << std::endl;
        compare_images(dst_cpp_naive, dst_cpp_integral);
    }

    {
        const auto dst = dst_cpp_sliding.data();
        const auto duration
            = measure(iteration, cpp::mean_filter_sliding, src, dst, width, height, ksize);
        std::cout << "\tcpp sliding  : " << duration << " [usec]" << std::endl;
        compare_images(dst_cpp_naive, dst_cpp_sliding);
    }

    {
        const auto dst = dst_cpp_separate.data();
        const auto duration
            = measure(iteration, cpp::mean_filter_separate, src, dst, width, height, ksize);
        std::cout << "\tcpp separate : " << duration << " [usec]" << std::endl;
        compare_images(dst_cpp_naive, dst_cpp_separate);
    }

    {
        const auto dst = dst_neon_separate.data();
        const auto duration
            = measure(iteration, neon::mean_filter_separate, src, dst, width, height, ksize);
        std::cout << "\tneon separate: " << duration << " [usec]" << std::endl;
        compare_images(dst_cpp_naive, dst_neon_separate);
    }

    {
        device_buffer<IMG_T> d_src(width * height * 3, src);
        device_buffer<IMG_T> d_dst(width * height * 3);

        const auto duration =
            measure(iteration, cuda::mean_filter, d_src.get(), d_dst.get(), width, height, ksize);
        std::cout << "\tcuda         : " << duration << " [usec]" << std::endl;

        d_dst.download(dst_cuda.data());
        compare_images(dst_cpp_integral, dst_cuda);
    }

    if (dump_flag) {
        dst_cpp_naive.write("cpp_naive.png");
        dst_cpp_integral.write("cpp_integral.png");
        dst_cpp_sliding.write("cpp_sliding.png");
        dst_cpp_separate.write("cpp_separate.png");
        dst_neon_separate.write("neon_separate.png");
        dst_cuda.write("cuda.png");
    }

    return 0;
}
