CC=g++
CFLAGS=-Werror -Wall -Wextra -std=c++2a -g

src=Layer Dense NeuralNetwork Neuron
exe=main

all: ${exe}

main: Main.cpp Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o
	${CC} ${CFLAGS} $^ -o $@

clean:
	rm *.o ${exe} vgcore.*