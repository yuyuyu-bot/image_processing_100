#include <cstddef>
#include <cstdint>


namespace cuda {

void mean_filter(const std::uint8_t* const src, std::uint8_t* const dst,
                 const std::size_t width, const std::size_t height, const std::size_t ksize);

}  // namespace cuda
