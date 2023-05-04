DEBUG ?= 0
PARALLEL ?= 0
CXXFLAGS=-Werror -Wall -Wextra -std=c++2a
ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG -Og -g
else ifeq ($(PARALLEL), 1)
    CXXFLAGS += -DPARALLEL -fopenmp
else
	CXXFLAGS += -Ofast
endif
CXX=g++

exe=main mnist mnist_generator conv-mnist test-conv test-tensor

all: ${exe}

main: Main.cpp Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Optimizer.o
	${CXX} ${CXXFLAGS} $^ -o $@

mnist: Mnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Metric.o Flatten.o Optimizer.o Dropout.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

mnist_generator: MnistGenerator.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Metric.o Reshape.o Flatten.o Optimizer.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

conv-mnist: ConvMnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Conv2D.o Metric.o Flatten.o Optimizer.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

test-conv: testConv.cpp Utils.o Tensor.hpp
	${CXX} ${CXXFLAGS} $^ -o $@

test-tensor: testTensor.cpp Tensor.hpp
	${CXX} ${CXXFLAGS} $^ -o $@

clean:
	rm *.o ${exe} vgcore.* gmon.out