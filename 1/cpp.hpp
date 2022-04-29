#include <cstddef>
#include <cstdint>


namespace cpp {

void rgb_to_bgr(const std::uint8_t* src, std::uint8_t* dst,
                const std::size_t width, const std::size_t height) {
    for (std::size_t i = 0; i < width * height; i++) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        src += 3;
        dst += 3;
    }
}

}  // namespace cpp
