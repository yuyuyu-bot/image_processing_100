#ifndef CUDA_SAFE_CALL_HPP
#define CUDA_SAFE_CALL_HPP

#include <cuda_runtime_api.h>
#include <iostream>

#define CUDASafeCall() cuda_safe_call(cudaGetLastError(), __FILE__, __LINE__);


inline void cuda_safe_call(const cudaError& error, const char* const file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s : %d %s\n", file, line, cudaGetErrorString(error));
    }
}

#endif  // CUDA_SAFE_CALL_HPP
