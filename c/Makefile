CC=gcc
CFLAGS=-Werror -Wall -Wextra -std=c2x -Ofast
LDFLAGS=-lm

src=utils data_utils neural_network layer
exe=main house mnist

all: ${exe}

main: main.c neural_network.o utils.o data_utils.o layer.o
	${CC} ${CFLAGS} $^ -o $@ -lm

test: test_read_csv.c data_utils.o
	${CC} ${CFLAGS} $^ -o $@ -lm
	@./test

house: house_data.c data_utils.o neural_network.o utils.o layer.o
	${CC} ${CFLAGS} $^ -o $@ -lm

mnist: mnist.c data_utils.o utils.o neural_network.o layer.o
	${CC} ${CFLAGS} $^ -o $@ -lm

clean:
	rm *.o ${exe} vgcore.*