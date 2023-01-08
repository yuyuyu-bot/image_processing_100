#include <cstddef>
#include <cstdint>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "threshold_otsu_cpp.hpp"
#include "threshold_otsu_cuda.hpp"
#include "threshold_otsu_neon.hpp"


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

    // benchmark
    {
        const cv::Mat src_mat(image_height, image_width, CV_8UC1, const_cast<IMG_T*>(src));
        cv::Mat dst_mat(image_height, image_width, CV_8UC1, dst_bench.data());
        MEASURE(num_itr, cv::threshold, src_mat, dst_mat, 0, 255, cv::THRESH_OTSU);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp.data();
        MEASURE(num_itr, cpp::threshold_otsu, src, dst, image_width, image_height);
        compare_images(dst_bench, dst_cpp);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::threshold_otsu, src, dst, image_width, image_height);
        compare_images(dst_bench, dst_neon);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height, src);
        device_buffer<IMG_T> d_dst(image_width * image_height);
        device_buffer<int> d_histogram_buffer(256);

        MEASURE(num_itr, cuda::threshold_otsu, d_src.get(), d_dst.get(), image_width, image_height, d_histogram_buffer.get());

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
