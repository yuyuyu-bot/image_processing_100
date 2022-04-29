#include <cstddef>
#include <cstdint>


namespace cpp {

void rgb_to_gray(const std::uint8_t* src, std::uint8_t* dst,
                 const std::size_t width, const std::size_t height) {
    constexpr auto r_coeff = 0.2126f;
    constexpr auto g_coeff = 0.7152f;
    constexpr auto b_coeff = 0.0722f;

    for (std::size_t i = 0; i < width * height; i++) {
        *dst = r_coeff * src[0] + g_coeff * src[1] + b_coeff * src[2];
        src += 3;
        dst++;
    }
}

}  // namespace cpp
