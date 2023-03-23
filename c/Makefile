CC=gcc
CFLAGS=-Werror -Wall -Wextra -std=c2x -Ofast
LDFLAGS=-lm

src=utils data_utils neural_network layer
exe=main house mnist mnist_generator

all: ${exe}

main: main.c neural_network.o utils.o data_utils.o layer.o
	${CC} ${CFLAGS} $^ -o $@ ${LDFLAGS}

test: test_read_csv.c data_utils.o
	${CC} ${CFLAGS} $^ -o $@ ${LDFLAGS}
	@./test

house: house_data.c data_utils.o neural_network.o utils.o layer.o
	${CC} ${CFLAGS} $^ -o $@ ${LDFLAGS}

mnist: mnist.c data_utils.o utils.o neural_network.o layer.o mnist_utils.o
	${CC} ${CFLAGS} $^ -o $@ ${LDFLAGS} -lSDL2

mnist_generator: mnist_generator.c mnist_utils.o utils.o data_utils.o neural_network.o layer.o
	${CC} ${CFLAGS} $^ -o $@ ${LDFLAGS} -lSDL2

clean:
	rm *.o ${exe} vgcore.*
