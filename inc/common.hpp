#ifndef COMMON_HPP
#define COMMON_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <png++/png.hpp>
#include <string>
#include <type_traits>

#define FUNC_NAME__(func) #func
#define MEASURE(itr, func, ...) measure(itr, func, FUNC_NAME__(func), __VA_ARGS__);

namespace {

using IMG_T = std::uint8_t;

constexpr auto image_color_path = "../images/peach.png";
constexpr auto image_gray_path = "../images/peach_gray.png";
constexpr auto image_width = 512;
constexpr auto image_height = 512;

template <typename IMG_T, int CH = 3>
class Image {
public:

    Image(const Image&) = delete;
    Image(const Image&&) = delete;
    Image& operator=(const Image&) = delete;
    Image& operator=(const Image&&) = delete;

    Image(const std::size_t width, const std::size_t height) :
        data_(std::make_unique<IMG_T[]>(width * height * CH)),
        width_(width), height_(height), stride_(width * CH) {}

    Image(const std::string& filename, const std::size_t width, const std::size_t height) :
        data_(std::make_unique<IMG_T[]>(width * height * CH)),
        width_(width), height_(height), stride_(width * CH) {
        static_assert(std::is_same_v<IMG_T, std::uint8_t>);
        static_assert(CH == 1 || CH == 3);

        if constexpr (CH == 1) {
            png::image<png::gray_pixel> image(filename);
            assert(width == image.get_width());
            assert(height == image.get_height());

            for (std::size_t y = 0; y < height_; y++) {
                const auto src_row = image[y];
                const auto dst_row = this->get_row(y);
                for (std::size_t x = 0; x < width_; x++) {
                    dst_row[x] = src_row[x];
                }
            }
        } else {
            png::image<png::rgb_pixel> image(filename);
            assert(width == image.get_width());
            assert(height == image.get_height());

            for (std::size_t y = 0; y < height_; y++) {
                const auto src_row = image[y];
                const auto dst_row = this->get_row(y);
                for (std::size_t x = 0; x < width_; x++) {
                    dst_row[x * CH + 0] = src_row.at(x).red;
                    dst_row[x * CH + 1] = src_row.at(x).green;
                    dst_row[x * CH + 2] = src_row.at(x).blue;
                }
            }
        }
    }

    void write(const std::string& filename) {
        static_assert(std::is_same_v<IMG_T, std::uint8_t>);
        static_assert(CH == 1 || CH == 3);

        if constexpr (CH == 1) {
            png::image<png::gray_pixel> image(width_, height_);
            for (std::size_t y = 0; y < height_; y++) {
                const auto src_row = this->get_row(y);
                auto& dst_row = image[y];
                for (std::size_t x = 0; x < width_; x++) {
                    dst_row[x] = src_row[x];
                }
            }
            image.write(filename);
        } else {
            png::image<png::rgb_pixel> image(width_, height_);
            for (std::size_t y = 0; y < height_; y++) {
                const auto src_row = this->get_row(y);
                auto& dst_row = image[y];
                for (std::size_t x = 0; x < width_; x++) {
                    dst_row[x] =
                        png::rgb_pixel(src_row[x * CH], src_row[x * CH + 1], src_row[x * CH + 2]);
                }
            }
            image.write(filename);
        }
    }

    const IMG_T* data() const {
        return data_.get();
    }

    IMG_T* data() {
        return data_.get();
    }

    const IMG_T* get_row(const std::size_t index) const {
        assert(index < height_);
        return data_.get() + stride_ * index;
    }

    IMG_T* get_row(const std::size_t index) {
        assert(index < height_);
        return data_.get() + stride_ * index;
    }

    auto width() const {
        return width_;
    }

    auto height() const {
        return height_;
    }

private:

    std::unique_ptr<IMG_T[]> data_;
    std::size_t width_, height_, stride_;
};


template <typename IMG_T, int CH>
void compare_images(const Image<IMG_T, CH>& img1, const Image<IMG_T, CH>& img2,
                    bool details = false){
    const auto size = img1.width() * img2.height() * CH;
    auto max_diff = 0;
    for (std::size_t i = 0; i < size; i++) {
        const auto expected = static_cast<int>(img1.data()[i]);
        const auto actual = static_cast<int>(img2.data()[i]);
        const auto diff = std::abs(expected - actual);
        max_diff = std::max(max_diff, diff);
        if (details && diff > 0) {
            std::printf("expected: %d, actual: %d at (%d, %d)\n",
                        static_cast<int>(expected), static_cast<int>(actual),
                        static_cast<int>(i % img1.width()), static_cast<int>(i / img1.width()));
        }
    }
    if (max_diff > 0) {
        std::cout << "max diff: " << max_diff << std::endl;
    }
}


template <class Fn, class... Args>
auto measure(const std::uint32_t N, Fn& fn, const char* const fn_str, const Args&... args) {
    auto accumulator = 0ll;

    for (std::uint32_t i = 0; i <= N; i++) {
        const auto begin = std::chrono::high_resolution_clock::now();
        fn(args...);
        const auto end = std::chrono::high_resolution_clock::now();

        if (i > 0) {
            accumulator += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        }
    }

    const auto duration = accumulator / N;
    std::printf("\t%-30s: %10.3f [usec]\n", fn_str, duration * 1e-3);

    return accumulator / N;
}

}  // anonymous namespace

#endif  // COMMON_HPP
