#include <cstddef>
#include <cstdint>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "mean_filter_cpu.hpp"
#include "mean_filter_cuda.hpp"
#include "mean_filter_neon.hpp"


int main(const int argc, const char** argv) {
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " num_itr [--simd] [--cuda] [--dump]" << std::endl;
        return 0;
    }
    const auto num_itr = std::stoi(argv[1]);
    const auto flags = parse_flags(argc, argv);

    const Image<IMG_T, 3> src_img(image_color_path);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_bench(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_naive(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_integral(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_sliding(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_separate(image_width, image_height);
    Image<IMG_T, 3> dst_neon_separate(image_width, image_height);
    Image<IMG_T, 3> dst_cuda(image_width, image_height);

    constexpr auto ksize = 15;
    static_assert(ksize % 2 == 1);

    // benchmark
    {
        const cv::Mat src_mat(image_height, image_width, CV_8UC3, const_cast<IMG_T*>(src));
        cv::Mat dst_mat(image_height, image_width, CV_8UC3, dst_bench.data());
        MEASURE(num_itr, cv::boxFilter, src_mat, dst_mat, -1, cv::Size{ ksize, ksize }, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp_naive.data();
        MEASURE(num_itr, cpp::mean_filter_naive, src, dst, image_width, image_height, ksize);
        compare_images(dst_bench, dst_cpp_naive);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp_integral.data();
        MEASURE(num_itr, cpp::mean_filter_integral, src, dst, image_width, image_height, ksize);
        compare_images(dst_bench, dst_cpp_integral);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp_sliding.data();
        MEASURE(num_itr, cpp::mean_filter_sliding, src, dst, image_width, image_height, ksize);
        compare_images(dst_bench, dst_cpp_sliding);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp_separate.data();
        MEASURE(num_itr, cpp::mean_filter_separate, src, dst, image_width, image_height, ksize);
        compare_images(dst_bench, dst_cpp_separate);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon_separate.data();
        MEASURE(num_itr, neon::mean_filter_separate, src, dst, image_width, image_height, ksize);
        compare_images(dst_bench, dst_neon_separate);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height * 3);

        MEASURE(num_itr, cuda::mean_filter, d_src.get(), d_dst.get(), image_width, image_height, ksize);

        d_dst.download(dst_cuda.data());
        compare_images(dst_bench, dst_cuda);
    }

    if (flags.dump_imgs) {
        dst_bench.write("bench.png");
        if (flags.run_cpp) { dst_cpp_naive.write("cpp_naive.png"); }
        if (flags.run_cpp) { dst_cpp_integral.write("cpp_integral.png"); }
        if (flags.run_cpp) { dst_cpp_sliding.write("cpp_sliding.png"); }
        if (flags.run_cpp) { dst_cpp_separate.write("cpp_separate.png"); }
        if (flags.run_simd) { dst_neon_separate.write("neon_separate.png"); }
        if (flags.run_cuda) { dst_cuda.write("cuda.png"); }
    }

    return 0;
}
