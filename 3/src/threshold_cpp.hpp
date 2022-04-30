#include <cstddef>
#include <cstdint>
#include <numeric>


namespace cpp {

void threshold(const std::uint8_t* const src, std::uint8_t* const dst,
               const std::size_t width, const std::size_t height, const std::uint8_t thresh) {
    constexpr auto min_value = std::numeric_limits<std::uint8_t>::min();
    constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();

    for (std::size_t i = 0; i < width * height; i++) {
        dst[i] = src[i] < thresh ? min_value : max_value;
    }
}

}  // namespace cpp
