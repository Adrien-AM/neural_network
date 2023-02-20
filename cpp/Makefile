CC=g++
CFLAGS=-Werror -Wall -Wextra -std=c++2a

src=Layer Dense NeuralNetwork
exe=main

all: ${exe}

main: Main.cpp Dense.o NeuralNetwork.o Loss.o
	${CC} ${CFLAGS} $^ -o $@

clean:
	rm *.o ${exe} vgcore.*