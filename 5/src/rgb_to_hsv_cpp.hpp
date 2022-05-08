#include <cstddef>
#include <cstdint>
#include <algorithm>


namespace cpp {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height) {
    const auto stride = width * 3;

    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            const auto R = src[y * stride + x * 3 + 0];
            const auto G = src[y * stride + x * 3 + 1];
            const auto B = src[y * stride + x * 3 + 2];

            const auto vmax = std::max({R, G, B});
            const auto vmin = std::min({R, G, B});

            int H;
            if (vmin == vmax) {
                H = 0;
            } else if (vmin == B) {
                H = 60 * (G - R) / (vmax - vmin) + 60;
            } else if (vmin == R) {
                H = 60 * (B - G) / (vmax - vmin) + 180;
            } else {
                H = 60 * (R - B) / (vmax - vmin) + 300;
            }

            const auto S = vmax - vmin;
            const auto V = vmax;

            dst[y * stride + x * 3 + 0] = static_cast<std::uint8_t>(H / 360.f * 255.f);
            dst[y * stride + x * 3 + 1] = S;
            dst[y * stride + x * 3 + 2] = V;
        }
    }
}

}  // namespace cpp
