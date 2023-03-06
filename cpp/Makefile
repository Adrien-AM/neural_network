DEBUG ?= 0
CXXFLAGS=-Werror -Wall -Wextra -std=c++2a -pg
ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG
endif
CXX=g++

exe=main mnist conv-mnist

all: ${exe}

main: Main.cpp Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o
	${CXX} ${CXXFLAGS} $^ -o $@

mnist: Mnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

conv-mnist: ConvMnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Conv2D.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

clean:
	rm *.o ${exe} vgcore.*