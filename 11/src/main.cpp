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
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " num_itr dump_flag" << std::endl;
        return 0;
    }
    const auto num_itr = std::stoi(argv[1]);
    const auto dump_flag = std::stoi(argv[2]) != 0;

    const Image<IMG_T, 3> src_img(image_color_path, image_width, image_height);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_cpp_naive(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_integral(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_sliding(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_separate(image_width, image_height);
    Image<IMG_T, 3> dst_neon_separate(image_width, image_height);
    Image<IMG_T, 3> dst_cuda(image_width, image_height);

    constexpr auto ksize = 15;
    static_assert(ksize % 2 == 1);

    {
        const auto dst = dst_cpp_naive.data();
        MEASURE(num_itr, cpp::mean_filter_naive, src, dst, image_width, image_height, ksize);
    }

    {
        const auto dst = dst_cpp_integral.data();
        MEASURE(num_itr, cpp::mean_filter_integral, src, dst, image_width, image_height, ksize);
        compare_images(dst_cpp_naive, dst_cpp_integral);
    }

    {
        const auto dst = dst_cpp_sliding.data();
        MEASURE(num_itr, cpp::mean_filter_sliding, src, dst, image_width, image_height, ksize);
        compare_images(dst_cpp_naive, dst_cpp_sliding);
    }

    {
        const auto dst = dst_cpp_separate.data();
        MEASURE(num_itr, cpp::mean_filter_separate, src, dst, image_width, image_height, ksize);
        compare_images(dst_cpp_naive, dst_cpp_separate);
    }

    {
        const auto dst = dst_neon_separate.data();
        MEASURE(num_itr, neon::mean_filter_separate, src, dst, image_width, image_height, ksize);
        compare_images(dst_cpp_naive, dst_neon_separate);
    }

    {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height * 3);

        MEASURE(num_itr, cuda::mean_filter, d_src.get(), d_dst.get(), image_width, image_height, ksize);

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
