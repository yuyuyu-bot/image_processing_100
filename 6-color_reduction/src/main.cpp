#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "color_reduction_cpp.hpp"
#include "color_reduction_cuda.hpp"
#include "color_reduction_neon.hpp"

void cv_color_reduction(const cv::Mat* const src_mat, cv::Mat* const dst_mat) {
    (*src_mat).copyTo(*dst_mat);
    (*dst_mat).forEach<cv::Vec3b>([](cv::Vec3b& color, const int*) {
        color[0] = (color[0] / 64) * 64 + 32;
        color[1] = (color[1] / 64) * 64 + 32;
        color[2] = (color[2] / 64) * 64 + 32;
    });
}

int main(const int argc, const char** argv) {
    const auto [num_itr, flags] = parse_args(argc, argv);

    const Image<IMG_T, 3> src_img(image_color_path);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_bench(image_width, image_height);
    Image<IMG_T, 3> dst_cpp(image_width, image_height);
    Image<IMG_T, 3> dst_neon(image_width, image_height);
    Image<IMG_T, 3> dst_cuda(image_width, image_height);

    // benchmark
    {
        const cv::Mat src_mat(image_height, image_width, CV_8UC3, const_cast<IMG_T*>(src));
        cv::Mat dst_mat(image_height, image_width, CV_8UC3, dst_bench.data());
        MEASURE(num_itr, cv_color_reduction, &src_mat, &dst_mat);
    }

    if (flags.run_cpp) {
        const auto dst = dst_cpp.data();
        MEASURE(num_itr, cpp::color_reduction, src, dst, image_width, image_height);
        compare_images(dst_bench, dst_cpp);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::color_reduction, src, dst, image_width, image_height);
        compare_images(dst_bench, dst_neon);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height * 3);

        MEASURE(num_itr, cuda::color_reduction, d_src.get(), d_dst.get(), image_width, image_height);

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
