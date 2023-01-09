#include <cstddef>
#include <cstdint>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "gaussian_filter_cpp.hpp"
#include "gaussian_filter_cuda.hpp"
#include "gaussian_filter_neon.hpp"


int main(const int argc, const char** argv) {
    const auto [num_itr, flags] = parse_args(argc, argv);

    const Image<IMG_T, 3> src_img(image_color_path);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_bench(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_naive(image_width, image_height);
    Image<IMG_T, 3> dst_cpp_separate(image_width, image_height);
    Image<IMG_T, 3> dst_neon(image_width, image_height);
    Image<IMG_T, 3> dst_cuda(image_width, image_height);
    Image<IMG_T, 3> dst_cuda_shared(image_width, image_height);

    constexpr auto ksize = 5;
    constexpr auto sigma = 10.f;
    static_assert(ksize % 2 == 1);

    // benchmark
    {
        const cv::Mat src_mat(image_height, image_width, CV_8UC3, const_cast<IMG_T*>(src));
        cv::Mat dst_mat(image_height, image_width, CV_8UC3, dst_bench.data());
        MEASURE(num_itr, cv::GaussianBlur, src_mat, dst_mat, cv::Size{ksize, ksize}, sigma, sigma, cv::BORDER_REPLICATE);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp_naive.data();
        MEASURE(num_itr, cpp::gaussian_filter_naive, src, dst, image_width, image_height, ksize, sigma);
        compare_images(dst_bench, dst_cpp_naive);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp_separate.data();
        MEASURE(num_itr, cpp::gaussian_filter_separate, src, dst, image_width, image_height, ksize, sigma);
        compare_images(dst_bench, dst_cpp_separate);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::gaussian_filter_separate, src, dst, image_width, image_height, ksize, sigma);
        compare_images(dst_bench, dst_neon);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height * 3);

        MEASURE(num_itr, cuda::gaussian_filter, d_src.get(), d_dst.get(), image_width, image_height, ksize, sigma);

        d_dst.download(dst_cuda.data());
        compare_images(dst_bench, dst_cuda);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height * 3);

        MEASURE(num_itr, cuda::gaussian_filter_shared, d_src.get(), d_dst.get(), image_width, image_height, ksize, sigma);

        d_dst.download(dst_cuda_shared.data());
        compare_images(dst_bench, dst_cuda_shared);
    }

    if (flags.dump_imgs) {
        dst_bench.write("bench.png");
        if (flags.run_cpp) { dst_cpp_naive.write("cpp_naive.png"); }
        if (flags.run_cpp) { dst_cpp_separate.write("cpp_separate.png"); }
        if (flags.run_simd) { dst_neon.write("neon.png"); }
        if (flags.run_cuda) { dst_cuda.write("cuda.png"); }
        if (flags.run_cuda) { dst_cuda_shared.write("cuda_shared.png"); }
    }

    return 0;
}
