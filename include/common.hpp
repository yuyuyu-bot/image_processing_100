#ifndef COMMON_HPP
#define COMMON_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/highgui.hpp>
#include <string>

#define FUNC_NAME__(func) #func
#define MEASURE(itr, func, ...) measure(itr, func, FUNC_NAME__(func), __VA_ARGS__);

namespace {

using IMG_T = std::uint8_t;

constexpr auto image_color_path = "images/peach.png";
constexpr auto image_gray_path = "images/peach_gray.png";
constexpr auto image_width = 512;
constexpr auto image_height = 512;

template <typename ElemType, int CH = 3>
class Image {
public:
    Image(const Image&) = delete;
    Image(const Image&&) = delete;
    Image& operator=(const Image&) = delete;
    Image& operator=(const Image&&) = delete;

    Image(const std::size_t width, const std::size_t height)
        : data_(std::make_unique<ElemType[]>(width * height * CH)), width_(width), height_(height), stride_(width * CH) {}

    Image(const std::string& filename) {
        static_assert(std::is_same_v<ElemType, std::uint8_t>);

        const cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
        assert(image.channels() == CH);

        width_ = image.cols;
        height_ = image.rows;
        stride_ = width_ * CH;
        data_ = std::make_unique<ElemType[]>(stride_ * height_);
        std::copy_n(image.begin<ElemType>(), stride_ * height_, data_.get());
    }

    void write(const std::string& filename) {
        static_assert(std::is_same_v<ElemType, std::uint8_t>);

        cv::Mat image(height_, width_, CV_8UC(CH));
        std::copy_n(data_.get(), stride_ * height_, image.begin<ElemType>());
        cv::imwrite(filename, image);
    }

    const ElemType* data() const {
        return data_.get();
    }

    ElemType* data() {
        return data_.get();
    }

    const ElemType* get_row(const std::size_t index) const {
        assert(index < height_);
        return data_.get() + stride_ * index;
    }

    ElemType* get_row(const std::size_t index) {
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
    std::unique_ptr<ElemType[]> data_;
    std::size_t width_;
    std::size_t height_;
    std::size_t stride_;
};

struct RunFlags {
    bool run_cpp = true;
    bool run_simd = false;
    bool run_cuda = false;
    bool dump_imgs = false;
};

inline auto parse_flags(const int argc, const char** argv) {
    RunFlags flags;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--cpp") {
            flags.run_cpp = true;
        }
        else if (std::string(argv[i]) == "--simd") {
            flags.run_simd = true;
        }
        else if (std::string(argv[i]) == "--cuda") {
            flags.run_cuda = true;
        }
        else if (std::string(argv[i]) == "--dump") {
            flags.dump_imgs = true;
        }
    }

    return flags;
}

template <typename ElemType, int CH>
inline void compare_images(const Image<ElemType, CH>& img1, const Image<ElemType, CH>& img2,
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
inline auto measure(const std::uint32_t N, Fn& fn, const char* const fn_str, const Args&... args) {
    auto accumulator = 0ll;

    for (std::uint32_t i = 0; i <= N; i++) {
        const auto begin = std::chrono::high_resolution_clock::now();
        fn(args...);
        const auto end = std::chrono::high_resolution_clock::now();

        if (N == 1 || i > 0) {
            accumulator += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        }
    }

    const auto duration = accumulator / N;
    std::printf("\t%-30s: %10.3f [usec]\n", fn_str, duration * 1e-3);

    return accumulator / N;
}

}  // anonymous namespace

#endif  // COMMON_HPP
