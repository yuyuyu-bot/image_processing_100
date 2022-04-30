CXX        = g++
CXXFLAGS  := -Wall -Wextra -O3 -std=c++17 -mssse3
CXXFLAGS  += -Wno-deprecated-declarations -Wno-unknown-pragmas

NVCC      := nvcc
NVCCFLAGS := -arch=sm_61 -O3 -x=cu -std=c++17 --compiler-bindir=$(CXX) -fmad=false

INCLUDES  := -I../inc -I/usr/local/cuda/include -I../3rdparty/neon2sse
LDFLAGS   := -L/usr/local/cuda/lib64 -lcudart -lpng

SRC_DIR   := src
SRCS       = $(shell find $(SRC_DIR) -name "*.cu" -o -name "*.cpp")
BUILD_DIR := build

OBJS       = $(subst $(SRC_DIR),$(BUILD_DIR),$(SRCS))
OBJS      := $(subst .cpp,.o,$(OBJS))
OBJS      := $(subst .cu,.o,$(OBJS))

TARGET    = $(BUILD_DIR)/main.a

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	[ -d $(BUILD_DIR) ] || mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -c -o $@

$(SRC_DIR)/%.cpp: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< --cuda -o $@

clean:
	rm $(BUILD_DIR) -rf
