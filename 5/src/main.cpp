#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "device_buffer.hpp"
#include "rgb_to_hsv_cpp.hpp"
#include "rgb_to_hsv_cuda.hpp"
#include "rgb_to_hsv_neon.hpp"


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " image_path width height dump_flag" << std::endl;
        return 0;
    }

    constexpr auto iteration = 10;

    using IMG_T = std::uint8_t;

    const auto width = std::stoul(argv[2]);
    const auto height = std::stoul(argv[3]);
    const auto dump_flag = std::stoi(argv[4]) != 0;
    const Image<IMG_T, 3> src_img(argv[1], width, height);
    const auto src = src_img.data();

    Image<IMG_T, 3> dst_cpp(width, height);
    Image<IMG_T, 3> dst_neon(width, height);
    Image<IMG_T, 3> dst_cuda(width, height);

    {
        const auto dst = dst_cpp.data();
        MEASURE(iteration, cpp::rgb_to_hsv, src, dst, width, height);
    }

    {
        const auto dst = dst_neon.data();
        MEASURE(iteration, neon::rgb_to_hsv, src, dst, width, height);
        compare_images(dst_cpp, dst_neon);
    }

    {
        device_buffer<IMG_T> d_src(width * height * 3, src);
        device_buffer<IMG_T> d_dst(width * height * 3);

        MEASURE(iteration, cuda::rgb_to_hsv, d_src.get(), d_dst.get(), width, height);

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
