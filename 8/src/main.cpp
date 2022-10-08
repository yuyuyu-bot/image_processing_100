#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "max_pooling_cpp.hpp"
#include "max_pooling_cuda.hpp"
#include "max_pooling_neon.hpp"


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " num_itr dump_flag" << std::endl;
        return 0;
    }
    const auto num_itr = std::stoi(argv[1]);
    const auto dump_flag = std::stoi(argv[2]) != 0;

    const Image<IMG_T, 3> src_img(image_color_path, image_width, image_height);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_cpp(image_width, image_height);
    Image<IMG_T, 3> dst_neon(image_width, image_height);
    Image<IMG_T, 3> dst_cuda(image_width, image_height);

    constexpr auto ksize = 101;
    static_assert(ksize % 2 == 1);

    {
        const auto dst = dst_cpp.data();
        MEASURE(num_itr, cpp::max_pooling, src, dst, image_width, image_height, ksize);
    }

    {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::max_pooling, src, dst, image_width, image_height, ksize);
        compare_images(dst_cpp, dst_neon);
    }

    {
        device_buffer<IMG_T> d_src(image_width * image_height * 3, src);
        device_buffer<IMG_T> d_dst(image_width * image_height * 3);

        MEASURE(num_itr, cuda::max_pooling, d_src.get(), d_dst.get(), image_width, image_height, ksize);

        d_dst.download(dst_cuda.data());
        compare_images(dst_cpp, dst_cuda);
    }

    if (dump_flag) {
        dst_cpp.write("cpp.png");
        dst_neon.write("neon.png");
        dst_cuda.write("cuda.png");
    }

    return 0;
}
