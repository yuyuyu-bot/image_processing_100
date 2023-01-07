#include <cstddef>
#include <cstdint>


namespace cuda {

void rgb_to_hsv(const std::uint8_t* const src, std::uint8_t* const dst,
                const std::size_t width, const std::size_t height);

}  // namespace cuda
