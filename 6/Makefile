CXX        = g++
CXXFLAGS  := -Wall -Wextra -O3 -std=c++17 -mssse3 -fno-tree-vectorize
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

DEPS       = $(subst $(SRC_DIR),$(BUILD_DIR),$(SRCS))
DEPS      := $(subst .cpp,.d,$(DEPS))

TARGET    = $(BUILD_DIR)/main.a

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -c -o $@

$(SRC_DIR)/%.cpp: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< --cuda -o $@

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp
	[ -d $(BUILD_DIR) ] || mkdir -p $(BUILD_DIR)
	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $< | sed -e 's~\($*\)\.o: *~$(BUILD_DIR)/\1.o $@ : ~g' > $@

-include $(DEPS)

run:
	$(TARGET) ../rgb_image_5000x5000.png 5000 5000

clean:
	rm $(BUILD_DIR) -rf
