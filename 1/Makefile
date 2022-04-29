CXX = g++
INCLUDE = -I../inc
FLAGS = -Wall -Wextra -O3 -std=c++17
LDFLAGS = -lpng

DSTDIR = out
SRC = main.cpp
OBJ = build/main.o
EXE = build/main.a


all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(FLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(OBJ): $(SRC)
	mkdir -p build
	$(CXX) $(FLAGS) $(INCLUDE) -o $@ -c $^

clean:
	rm build -rf
