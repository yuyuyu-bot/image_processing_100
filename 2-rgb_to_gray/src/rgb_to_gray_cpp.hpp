#include <cstddef>
#include <cstdint>


namespace cpp {

void rgb_to_gray(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height) {
    constexpr auto r_coeff = 4899;
    constexpr auto g_coeff = 9617;
    constexpr auto b_coeff = 1868;
    constexpr auto normalize_shift_bits = 14;

    auto src_ptr = src;
    auto dst_ptr = dst;

    for (std::size_t i = 0; i < width * height; i++) {
        *dst_ptr = (r_coeff * src_ptr[0] + g_coeff * src_ptr[1] + b_coeff * src_ptr[2]) >> normalize_shift_bits;
        src_ptr += 3;
        dst_ptr++;
    }
}

}  // namespace cpp
