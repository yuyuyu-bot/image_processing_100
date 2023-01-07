#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <algorithm>


namespace cpp {

void color_reduction(const std::uint8_t* const src, std::uint8_t* const dst,
                     const std::size_t width, const std::size_t height) {
    constexpr auto shr = 6u;
    constexpr auto offset = 1u << (shr - 1);

    std::transform(src, src + width * height * 3, dst,
        [&shr, &offset](const auto& val) { return ((val >> shr) << shr) + offset; });
}

}  // namespace cpp
