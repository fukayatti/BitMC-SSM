CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -march=native -ffast-math -pthread -fopenmp -I./src

TARGET = infer
SRCS = src/infer.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

test: test_tmac test_hadamard
	./test_hadamard
	./test_tmac
	pytest tests/

test_hadamard: tests/test_hadamard.cpp src/hadamard.h
	$(CXX) $(CXXFLAGS) tests/test_hadamard.cpp -o test_hadamard

test_tmac: tests/test_tmac.cpp src/tmac_gemm.h
	$(CXX) $(CXXFLAGS) tests/test_tmac.cpp -o test_tmac

clean:
	rm -f $(TARGET) test_hadamard test_tmac

run: $(TARGET)
	./$(TARGET) model_medium-30M.bin 60

.PHONY: all clean run test
