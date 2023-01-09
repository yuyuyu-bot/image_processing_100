#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "average_pooling_cpp.hpp"
#include "average_pooling_cuda.hpp"
#include "average_pooling_neon.hpp"


int main(const int argc, const char** argv) {
    const auto [num_itr, flags] = parse_args(argc, argv);

    const Image<IMG_T, 3> src_img(image_color_path);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_cpp(image_width, image_height);
    Image<IMG_T, 3> dst_neon(image_width, image_height);
    Image<IMG_T, 3> dst_cuda(image_width, image_height);

    constexpr auto ksize = 101;
    static_assert(ksize % 2 == 1);

    if (flags.run_cpp) {
        const auto dst = dst_cpp.data();
        MEASURE(num_itr, cpp::average_pooling, src, dst, image_width, image_height, ksize);
    }

    if (flags.run_simd) {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::average_pooling, src, dst, image_width, image_height, ksize);
        compare_images(dst_cpp, dst_neon);
    }

    if (flags.run_cuda) {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height * 3);

        MEASURE(num_itr, cuda::average_pooling, d_src.get(), d_dst.get(), image_width, image_height, ksize);

        d_dst.download(dst_cuda.data());
        compare_images(dst_cpp, dst_cuda);
    }

    if (flags.dump_imgs) {
        if (flags.run_cpp) { dst_cpp.write("cpp.png"); }
        if (flags.run_simd) { dst_neon.write("neon.png"); }
        if (flags.run_cuda) { dst_cuda.write("cuda.png"); }
    }

    return 0;
}
