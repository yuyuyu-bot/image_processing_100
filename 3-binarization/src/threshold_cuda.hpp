#include <cstddef>
#include <cstdint>


namespace cuda {

void threshold(const std::uint8_t* const, std::uint8_t* const,
               const std::size_t, const std::size_t, const std::uint8_t);

}  // namespace cuda
