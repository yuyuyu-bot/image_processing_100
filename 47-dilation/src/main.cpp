#include <cstddef>
#include <cstdint>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "dilation_cpp.hpp"
#include "dilation_cuda.hpp"
#include "dilation_neon.hpp"


int main(const int argc, const char** argv) {
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " num_itr [--simd] [--cuda] [--dump]" << std::endl;
        return 0;
    }
    const auto num_itr = std::stoi(argv[1]);
    const auto flags = parse_flags(argc, argv);

    const Image<IMG_T, 1> src_img(image_gray_path);
    const auto src = src_img.data();

    Image<IMG_T, 1> dst_bench(image_width, image_height);
    Image<IMG_T, 1> dst_cpp(image_width, image_height);
    Image<IMG_T, 1> dst_neon(image_width, image_height);
    Image<IMG_T, 1> dst_cuda(image_width, image_height);

    constexpr auto ksize = 15;
    static_assert(ksize % 2 == 1);

    // benchmark
    {
        const cv::Mat src_mat(image_height, image_width, CV_8UC1, const_cast<IMG_T*>(src));
        cv::Mat dst_mat(image_height, image_width, CV_8UC1, dst_bench.data());
        MEASURE(num_itr, cv::dilate, src_mat, dst_mat, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size{ksize, ksize}),
                cv::Point{-1, -1}, 1, cv::BORDER_REPLICATE, 0);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp.data();
        MEASURE(num_itr, cpp::dilation, src, dst, image_width, image_height, ksize);
        compare_images(dst_bench, dst_cpp);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::dilation, src, dst, image_width, image_height, ksize);
        compare_images(dst_bench, dst_neon);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height, src);
        device_buffer<IMG_T> d_dst(image_width * image_height);

        MEASURE(num_itr, cuda::dilation, d_src.get(), d_dst.get(), image_width, image_height, ksize);

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
