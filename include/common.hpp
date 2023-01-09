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
        static_assert(CH == 1 || CH == 3);

        const cv::Mat image = cv::imread(filename, (CH == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
        assert(image.depth() == CV_8U);
        assert(image.channels() == CH);

        width_  = image.cols;
        height_ = image.rows;
        stride_ = width_ * CH;
        data_   = std::make_unique<ElemType[]>(stride_ * height_);

        for (std::size_t y = 0; y < height_; y++) {
            auto row_ptr = this->get_row(y);
            for (std::size_t x = 0; x < width_; x++) {
                auto col_ptr = row_ptr + x * CH;
                if constexpr (CH == 1) {
                    *col_ptr = *image.ptr<ElemType>(y, x);
                }
                else if constexpr (CH == 3) {
                    *(col_ptr + 0) = image.at<cv::Vec3b>(y, x)[2];
                    *(col_ptr + 1) = image.at<cv::Vec3b>(y, x)[1];
                    *(col_ptr + 2) = image.at<cv::Vec3b>(y, x)[0];
                }
                else {
                    std::cerr << "Not implemented." << std::endl;
                }
            }
        }
    }

    void write(const std::string& filename) const {
        static_assert(std::is_same_v<ElemType, std::uint8_t>);

        cv::Mat image(height_, width_, CV_8UC(CH));
        for (std::size_t y = 0; y < height_; y++) {
            auto row_ptr = this->get_row(y);
            for (std::size_t x = 0; x < width_; x++) {
                auto col_ptr = row_ptr + x * CH;
                if constexpr (CH == 1) {
                    *image.ptr<ElemType>(y, x) = *col_ptr;
                }
                else if constexpr (CH == 3) {
                    image.at<cv::Vec3b>(y, x)[0] = *(col_ptr + 2);
                    image.at<cv::Vec3b>(y, x)[1] = *(col_ptr + 1);
                    image.at<cv::Vec3b>(y, x)[2] = *(col_ptr + 0);
                }
                else {
                    std::cerr << "Not implemented." << std::endl;
                }
            }
        }
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

inline auto parse_args(const int argc, const char** argv) {
    int num_itr = 100;
    RunFlags flags;

    for (int i = 1; i < argc;) {
        if (std::string(argv[i]) == "--itr") {
            if (i + 1 >= argc) {
                std::cerr << "Missing argument for itr." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            num_itr = std::stoi(argv[i + 1]);
            i += 2;
        }
        else if (std::string(argv[i]) == "--cpp") {
            flags.run_cpp = true;
            i++;
        }
        else if (std::string(argv[i]) == "--simd") {
            flags.run_simd = true;
            i++;
        }
        else if (std::string(argv[i]) == "--cuda") {
            flags.run_cuda = true;
            i++;
        }
        else if (std::string(argv[i]) == "--dump") {
            flags.dump_imgs = true;
            i++;
        }
        else {
            i++;
        }
    }

    return std::make_pair(num_itr, flags);
}

template <typename ElemType, int CH>
inline void compare_images(const Image<ElemType, CH>& img1, const Image<ElemType, CH>& img2){
    const auto size = img1.width() * img2.height() * CH;
    auto max_diff = 0;
    for (std::size_t i = 0; i < size; i++) {
        const auto expected = static_cast<int>(img1.data()[i]);
        const auto actual = static_cast<int>(img2.data()[i]);
        const auto diff = std::abs(expected - actual);
        max_diff = std::max(max_diff, diff);
    }
    if (max_diff > 0) {
        std::cout << "\tImages did not match. max diff: " << max_diff << std::endl;
    }
}

inline void compare_hsv_images(const Image<std::uint8_t, 3>& img1, const Image<std::uint8_t, 3>& img2, const std::uint8_t h_max = 180){
    const auto size = img1.width() * img2.height();
    auto max_diff = 0;
    for (std::size_t i = 0; i < size; i++) {
        // H
        {
            auto expected = static_cast<int>(img1.data()[i * 3]);
            if (expected == h_max) { expected = 0; }
            auto actual = static_cast<int>(img2.data()[i * 3]);
            if (actual == h_max) { actual = 0; }
            const auto diff = std::abs(expected - actual);
            max_diff = std::max(max_diff, diff);
        }
        // S
        {
            const auto expected = static_cast<int>(img1.data()[i * 3 + 1]);
            const auto actual = static_cast<int>(img2.data()[i * 3 + 1]);
            const auto diff = std::abs(expected - actual);
            max_diff = std::max(max_diff, diff);
        }
        // V
        {
            const auto expected = static_cast<int>(img1.data()[i * 3 + 2]);
            const auto actual = static_cast<int>(img2.data()[i * 3 + 2]);
            const auto diff = std::abs(expected - actual);
            max_diff = std::max(max_diff, diff);
        }
    }
    if (max_diff > 0) {
        std::cout << "\tImages did not match. max diff: " << max_diff << std::endl;
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

        if (N == 1) {
            break;
        }
    }

    const auto duration = accumulator / N;
    std::printf("\t%-30s: %10.3f [usec]\n", fn_str, duration * 1e-3);

    return accumulator / N;
}

}  // anonymous namespace

#endif  // COMMON_HPP
