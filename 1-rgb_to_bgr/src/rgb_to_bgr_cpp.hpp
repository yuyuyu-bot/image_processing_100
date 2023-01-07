#include <cstddef>
#include <cstdint>


namespace cpp {

void rgb_to_bgr(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    auto src_ptr = src;
    auto dst_ptr = dst;

    for (std::size_t i = 0; i < width * height; i++) {
        dst_ptr[0] = src_ptr[2];
        dst_ptr[1] = src_ptr[1];
        dst_ptr[2] = src_ptr[0];
        src_ptr += 3;
        dst_ptr += 3;
    }
}

}  // namespace cpp
