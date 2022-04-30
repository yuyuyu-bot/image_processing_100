#include <cstddef>
#include <cstdint>


namespace cuda {

void rgb_to_bgr(const std::uint8_t* const, std::uint8_t* const,
                const std::size_t, const std::size_t);

}  // namespace cuda
