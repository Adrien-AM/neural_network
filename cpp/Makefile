DEBUG ?= 0
CXXFLAGS=-Werror -Wall -Wextra -std=c++2a -Ofast
ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG
endif
CXX=g++

src=Layer Dense NeuralNetwork Neuron
exe=main mnist

all: ${exe}

main: Main.cpp Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o
	${CXX} ${CXXFLAGS} $^ -o $@

mnist: Mnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

clean:
	rm *.o ${exe} vgcore.*