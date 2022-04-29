CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++17

NVCC=nvcc
NVCCFLAGS = -arch=sm_61 -O3 -x=cu -std=c++17 --compiler-bindir=$(CXX) -fmad=false

INCLUDE = -I../inc -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lpng

CPP_SRC = main.cpp
CPP_OBJ = build/main.o
TARGET = build/main.a

CUDA_SRC = cuda_kernel.cu
CUDA_OBJ = build/cuda_kernel.o
TARGET = build/main.a


all: $(TARGET)

$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(CUDA_OBJ): $(CUDA_SRC)
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ -c $^

$(CPP_OBJ): $(CPP_SRC)
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ -c $^
clean:
	rm build -rf
