#include <cstddef>
#include <cstdint>
#include <emmintrin.h>

#include "avx2_utils.hpp"

#include <iostream>

namespace avx2 {

void rgb_to_gray(const std::uint8_t* const src, std::uint8_t* const /*dst*/,
                 const std::size_t width, const std::size_t height) {
    constexpr auto num_pixel_per_itr = sizeof(__m128i) / sizeof(std::uint8_t);

    std::size_t i = num_pixel_per_itr * width * 3;
    for (; i + num_pixel_per_itr < width * height; i += num_pixel_per_itr) {
        const auto [v_r, v_g, v_b] = mm128_load3_u8(&src[i * 3]);

        std::cout << (int)src[i * 3 + 0 * 3 + 0] << " " << _mm_extract_epi8(v_r, 0) << std::endl;
        std::cout << (int)src[i * 3 + 0 * 3 + 1] << " " << _mm_extract_epi8(v_g, 0) << std::endl;
        std::cout << (int)src[i * 3 + 0 * 3 + 2] << " " << _mm_extract_epi8(v_b, 0) << std::endl;
        std::cout << std::endl;
        std::cout << (int)src[i * 3 + 1 * 3 + 0] << " " << _mm_extract_epi8(v_r, 1) << std::endl;
        std::cout << (int)src[i * 3 + 1 * 3 + 1] << " " << _mm_extract_epi8(v_g, 1) << std::endl;
        std::cout << (int)src[i * 3 + 1 * 3 + 2] << " " << _mm_extract_epi8(v_b, 1) << std::endl;
        std::cout << std::endl;
        std::cout << (int)src[i * 3 + 2 * 3 + 0] << " " << _mm_extract_epi8(v_r, 2) << std::endl;
        std::cout << (int)src[i * 3 + 2 * 3 + 1] << " " << _mm_extract_epi8(v_g, 2) << std::endl;
        std::cout << (int)src[i * 3 + 2 * 3 + 2] << " " << _mm_extract_epi8(v_b, 2) << std::endl;
        std::cout << std::endl;
        std::cout << (int)src[i * 3 + 3 * 3 + 0] << " " << _mm_extract_epi8(v_r, 3) << std::endl;
        std::cout << (int)src[i * 3 + 3 * 3 + 1] << " " << _mm_extract_epi8(v_g, 3) << std::endl;
        std::cout << (int)src[i * 3 + 3 * 3 + 2] << " " << _mm_extract_epi8(v_b, 3) << std::endl;
        std::cout << std::endl;
        std::cout << (int)src[i * 3 + 4 * 3 + 0] << " " << _mm_extract_epi8(v_r, 4) << std::endl;
        std::cout << (int)src[i * 3 + 4 * 3 + 1] << " " << _mm_extract_epi8(v_g, 4) << std::endl;
        std::cout << (int)src[i * 3 + 4 * 3 + 2] << " " << _mm_extract_epi8(v_b, 4) << std::endl;
        std::cout << std::endl;


        std::exit(0);
    }
}

}  // namespace cpp
