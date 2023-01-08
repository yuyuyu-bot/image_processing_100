#include <cstddef>
#include <cstdint>
#include <algorithm>


namespace cpp {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {

    for (std::size_t i = 0, end = width * height * 3; i < end; i += 3) {
#ifdef USE_FIXED_POINT
        constexpr auto N120 = 120u << 8;     // u8.8
        constexpr auto N240 = 240u << 8;     // u8.8
        constexpr auto N360 = 360u << 8;     // u9.8
        constexpr auto N0p5 = (1u << 8) / 2; // u0.8

        const std::int32_t R = src[i + 0] << 8; // s8.8
        const std::int32_t G = src[i + 1] << 8; // s8.8
        const std::int32_t B = src[i + 2] << 8; // s8.8

        const std::int32_t vmax = std::max({R, G, B}); // s8.8
        const std::int32_t vmin = std::min({R, G, B}); // s8.8
        const std::int32_t diff = (vmax - vmin) >> 8;  // u8

        std::int32_t H;
        if (vmin == vmax) {
            H = 0;
        } else if (vmax == R) {
            H = 60 * (G - B); // s14.8
            H /= diff;        // s6.8
            H = H < 0 ? H + N360 : H; // u9.8
        } else if (vmax == G) {
            H = 60 * (B - R); // s14.8
            H /= diff;        // s6.8
            H += N120;        // s9.8
        } else { // vmax == B
            H = 60 * (R - G); // s14.8
            H /= diff;        // s6.8
            H += N240;        // s9.8
        }

        auto S = vmax - vmin; // s8.8
        S = vmax != 0 ? 255 * S / (vmax >> 8) : 0; // s8.8

        dst[i + 0] = static_cast<std::uint8_t>(((H >> 1) + N0p5) >> 8);
        dst[i + 1] = static_cast<std::uint8_t>((S + N0p5) >> 8);
        dst[i + 2] = static_cast<std::uint8_t>(vmax >> 8);
#else
        const auto R = src[i + 0];
        const auto G = src[i + 1];
        const auto B = src[i + 2];

        const auto vmax = std::max({R, G, B});
        const auto vmin = std::min({R, G, B});

        float H;
        if (vmin == vmax) {
            H = 0.f;
        } else if (vmax == R) {
            H = 60.f * (G - B) / (vmax - vmin);
            H = H < 0 ? H + 360.f : H;
        } else if (vmax == G) {
            H = 60.f * (B - R) / (vmax - vmin) + 120.f;
        } else { // vmax == B
            H = 60.f * (R - G) / (vmax - vmin) + 240.f;
        }

        const auto S = vmax != 0 ? 255.f * (vmax - vmin) / vmax : 0;
        const auto V = vmax;

        dst[i + 0] = static_cast<std::uint8_t>(H / 2.f + 0.5f);
        dst[i + 1] = static_cast<std::uint8_t>(S + 0.5f);
        dst[i + 2] = V;
#endif // USE_FIXED_POINT
    }
}

}  // namespace cpp
