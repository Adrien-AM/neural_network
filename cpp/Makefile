DEBUG ?= 0
PARALLEL ?= 0
CXXFLAGS=-Werror -Wall -Wextra -std=c++2a
ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG -Og
else ifeq ($(PARALLEL), 1)
    CXXFLAGS += -DPARALLEL -fopenmp
else
	CXXFLAGS += -Ofast -pg
endif
CXX=g++

src=Layer Dense NeuralNetwork Neuron
exe=main mnist

all: ${exe}

main: Main.cpp Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o
	${CXX} ${CXXFLAGS} $^ -o $@

mnist: Mnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Metric.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

mnist_generator: MnistGenerator.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Metric.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

clean:
	rm *.o ${exe} vgcore.* gmon.out