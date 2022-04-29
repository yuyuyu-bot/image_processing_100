#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "common.hpp"
#include "cpp.hpp"


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " image_path width height" << std::endl;
        return 0;
    }

    using IMG_T = std::uint8_t;

    const auto width = std::stoul(argv[2]);
    const auto height = std::stoul(argv[3]);
    const Image<IMG_T, 3> src(argv[1], width, height);
    Image<IMG_T, 3> dst(width, height);

    const auto cpp_duration = measure(100, cpp::rgb_to_bgr, src.data(), dst.data(), width, height);

    std::cout << "durations:" << std::endl;
    std::cout << "\tcpp: " << cpp_duration << " [usec]" << std::endl;
    dst.write("cpp.png");


    return 0;
}
