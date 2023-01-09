#ifndef AVX2_UTILS_HPP
#define AVX2_UTILS_HPP

#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>
#include <tuple>
#include <type_traits>

namespace {

template <typename ElemType>
inline auto mm256_dup_n_32(const ElemType value) {
    static_assert(std::is_same_v<ElemType, std::int32_t> || std::is_same_v<ElemType, std::uint32_t>);
    return _mm256_setr_epi32(value, value, value, value, value, value, value, value);
}

inline auto mm128_load3_u8(const std::uint8_t* const ptr) {
    constexpr std::uint8_t mask_a[] = {
        0, 3, 6,  9, 12, 15,
        1, 4, 7, 10, 13,
        2, 5, 8, 11, 14,
    };
    constexpr std::uint8_t mask_b[] = {
           2, 5,  8, 11, 14,
        0, 3, 6,  9, 12, 15,
        1, 4, 7, 10, 13
    };
    constexpr std::uint8_t mask_c[] = {
           1, 4, 7, 10, 13,
           2, 5, 8, 11, 14,
        0, 3, 6, 9, 12, 15
    };

    // load vectors
    const __m128i v_ptr_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr +  0));
    const __m128i v_ptr_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + 16));
    const __m128i v_ptr_c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + 32));

    const __m128i reorder_a = _mm_shuffle_epi8(v_ptr_a, *reinterpret_cast<const __m128i*>(mask_a));
    const __m128i reorder_b = _mm_shuffle_epi8(v_ptr_b, *reinterpret_cast<const __m128i*>(mask_b));
    const __m128i reorder_c = _mm_shuffle_epi8(v_ptr_c, *reinterpret_cast<const __m128i*>(mask_c));

    const __m128i dst_a_1 = _mm_srli_si128(_mm_slli_si128(reorder_a, 10), 10);
    const __m128i dst_a_2 = _mm_srli_si128(_mm_slli_si128(reorder_b, 11), 5);
    const __m128i dst_a_3 =                _mm_slli_si128(reorder_c, 11);
    const __m128i dst_a   = _mm_or_si128(dst_a_1, _mm_or_si128(dst_a_2, dst_a_3));

    const __m128i dst_b_1 =                _mm_srli_si128(_mm_slli_si128(reorder_a,  5), 11);
    const __m128i dst_b_2 = _mm_slli_si128(_mm_srli_si128(_mm_slli_si128(reorder_b,  5), 10), 5);
    const __m128i dst_b_3 = _mm_slli_si128(_mm_srli_si128(reorder_c,  5), 11);
    const __m128i dst_b   = _mm_or_si128(dst_b_1, _mm_or_si128(dst_b_2, dst_b_3));

    const __m128i dst_c_1 =                _mm_srli_si128(reorder_a, 11);
    const __m128i dst_c_2 = _mm_slli_si128(_mm_srli_si128(reorder_b, 11),  5);
    const __m128i dst_c_3 = _mm_slli_si128(_mm_srli_si128(reorder_c, 10), 10);
    const __m128i dst_c   = _mm_or_si128(dst_c_1, _mm_or_si128(dst_c_2, dst_c_3));

    return std::make_tuple(dst_a, dst_b, dst_c);
}

} // anonymous namespace

#endif // AVX2_UTILS_HPP
