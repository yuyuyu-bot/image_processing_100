#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>


namespace cpp {

void threshold_otsu(const std::uint8_t* const src, std::uint8_t* const dst,
                    const std::size_t width, const std::size_t height) {
    constexpr auto min_value = std::numeric_limits<std::uint8_t>::min();
    constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();

    auto histogram = std::array<int, max_value - min_value + 1>();
    auto value_sum = 0ull;
    auto square_value_sum = 0ull;
    for (std::size_t i = 0; i < width * height; i++) {
        histogram[src[i]]++;
        value_sum += src[i];
        square_value_sum += src[i] * src[i];
    }
    const auto mean_all = static_cast<float>(value_sum) / (width * height);

    auto maxS = 0.f;
    auto thresh = std::uint8_t{0};
    auto n1 = 0, n2 = static_cast<int>(width * height);
    auto sum1 = 0ull, sum2 = value_sum;
    auto square_sum1 = 0ull, square_sum2 = square_value_sum;
    for (std::size_t i = 1; i < histogram.size(); i++) {
        n1 += histogram[i - 1];
        n2 -= histogram[i - 1];

        if (n1 == 0) {
            continue;
        }
        if (n2 <= 0) {
            break;
        }

        sum1 += (i - 1) * histogram[i - 1];
        sum2 -= (i - 1) * histogram[i - 1];
        square_sum1 += (i - 1) * (i - 1) * histogram[i - 1];
        square_sum2 -= (i - 1) * (i - 1) * histogram[i - 1];

        const auto mean1 = static_cast<float>(sum1) / n1;
        const auto mean2 = static_cast<float>(sum2) / n2;
        const auto dev1 = static_cast<float>(square_sum1) / n1 - mean1 * mean1;
        const auto dev2 = static_cast<float>(square_sum2) / n2 - mean2 * mean2;
        const auto S = (n1 * (mean1 - mean_all) * (mean1 - mean_all) +
                        n2 * (mean2 - mean_all) * (mean2 - mean_all)) / (n1 * dev1 + n2 * dev2);

        if (maxS < S) {
            maxS = S;
            thresh = i;
        }
    }

    for (std::size_t i = 0; i < width * height; i++) {
        dst[i] = src[i] < thresh ? min_value : max_value;
    }
}

}  // namespace cpp
