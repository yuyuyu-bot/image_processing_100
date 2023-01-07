#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "threshold_cpp.hpp"
#include "threshold_cuda.hpp"
#include "threshold_neon.hpp"


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " num_itr dump_flag" << std::endl;
        return 0;
    }
    const auto num_itr = std::stoi(argv[1]);
    const auto dump_flag = std::stoi(argv[2]) != 0;

    const Image<IMG_T, 1> src_img(image_gray_path);
    const auto src = src_img.data();

    Image<IMG_T, 1> dst_cpp(image_width, image_height);
    Image<IMG_T, 1> dst_neon(image_width, image_height);
    Image<IMG_T, 1> dst_cuda(image_width, image_height);

    constexpr auto thresh = 150;

    {
        const auto dst = dst_cpp.data();
        MEASURE(num_itr, cpp::threshold, src, dst, image_width, image_height, thresh);
    }

    {
        const auto dst = dst_neon.data();
        MEASURE(num_itr, neon::threshold, src, dst, image_width, image_height, thresh);
        compare_images(dst_cpp, dst_neon);
    }

    {
        device_buffer<IMG_T> d_src(image_width * image_height, src);
        device_buffer<IMG_T> d_dst(image_width * image_height);

        MEASURE(num_itr, cuda::threshold, d_src.get(), d_dst.get(), image_width, image_height, thresh);

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
