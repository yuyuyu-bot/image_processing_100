#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "rgb_to_gray_avx2.hpp"
#include "rgb_to_gray_cpp.hpp"
#include "rgb_to_gray_cuda.hpp"
#include "rgb_to_gray_neon.hpp"

int main(const int argc, const char** argv) {
    const auto [num_itr, flags] = parse_args(argc, argv);

    const Image<IMG_T, 3> src_img(image_color_path);
    const auto src = src_img.data();

    Image<IMG_T, 1> dst_bench(image_width, image_height);
    Image<IMG_T, 1> dst_cpp(image_width, image_height);
    Image<IMG_T, 1> dst_neon(image_width, image_height);
    Image<IMG_T, 1> dst_cuda(image_width, image_height);

    // benchmark
    {
        const cv::Mat src_mat(image_height, image_width, CV_8UC3, const_cast<IMG_T*>(src));
        cv::Mat dst_mat(image_height, image_width, CV_8UC1, dst_bench.data());
        MEASURE(num_itr, cv::cvtColor, src_mat, dst_mat, cv::COLOR_RGB2GRAY, 1);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp.data();
        MEASURE(num_itr, cpp::rgb_to_gray, src, dst, image_width, image_height);
        compare_images(dst_bench, dst_cpp);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::rgb_to_gray, src, dst, image_width, image_height);
        compare_images(dst_bench, dst_neon);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, avx2::rgb_to_gray, src, dst, image_width, image_height);
        compare_images(dst_bench, dst_neon);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height);

        MEASURE(num_itr, cuda::rgb_to_gray, d_src.get(), d_dst.get(), image_width, image_height);

        d_dst.download(dst_cuda.data());
        compare_images(dst_bench, dst_cuda);
    }

    if (flags.dump_imgs) {
        dst_bench.write("bench.png");
        if (flags.run_cpp) { dst_cpp.write("cpp.png"); }
        if (flags.run_simd) { dst_neon.write("neon.png"); }
        if (flags.run_cuda) { dst_cuda.write("cuda.png"); }
    }

    return 0;
}
