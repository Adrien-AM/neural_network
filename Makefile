CC=gcc
CFLAGS=-Werror -Wall -Wextra -std=c2x -Ofast
LDFLAGS=-lm

src=utils data_utils neural_network
exe=main house mnist

main: main.c neural_network.o utils.o data_utils.o layer.o
	${CC} ${CFLAGS} $^ -o $@ -lm

all: ${exe}

test: test_read_csv.c data_utils.o
	${CC} ${CFLAGS} $^ -o $@ -lm
	@./test

house: house_data.c data_utils.o neural_network.o utils.o layer.o
	${CC} ${CFLAGS} $^ -o $@ -lm

mnist: mnist.c data_utils.o utils.o neural_network.o layer.o
	${CC} ${CFLAGS} $^ -o $@ -lm

$(eval $(foreach f,src,$(echo $(f).o: $(f).c ${CC} ${CFLAGS} -c $@ -o $@ ${LDFLAGS})))

clean:
	rm *.o ${exe} test vgcore.*