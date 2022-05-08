#include <cstddef>
#include <cuda_runtime_api.h>

#include "cuda_safe_call.hpp"


template <typename T>
class device_buffer {
public:
    device_buffer(const device_buffer&) = delete;
    device_buffer(const device_buffer&&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&&) = delete;

    device_buffer() : data_(nullptr), size_(0) {}

    device_buffer(const std::size_t size, const T* data = nullptr) : size_(size) {
        cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T));
        CUDASafeCall();
        if (data != nullptr) {
            this->upload(data, size);
        }
    }

    ~device_buffer() {
        cudaFree(reinterpret_cast<void*>(data_));
        CUDASafeCall();
    }

    T* get() {
        return data_;
    }

    const T* get() const {
        return data_;
    }

    void upload(const T* data, std::size_t size = 0) {
        if (size == 0) {
            size = size_;
        }
        cudaMemcpy(data_, data, size * sizeof(T), cudaMemcpyHostToDevice);
        CUDASafeCall();
    }

    void download(T* const data, std::size_t size = 0) {
        if (size == 0) {
            size = size_;
        }
        cudaMemcpy(data, data_, size * sizeof(T), cudaMemcpyDeviceToHost);
        CUDASafeCall();
    }

private:
    T* data_;
    std::size_t size_;
};
