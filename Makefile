CC=gcc
CFLAGS=-Werror -Wall -Wextra -std=c2x
LDFLAGS=-lm

src=utils data_utils neural_network

main: main.c neural_network.o utils.o data_utils.o
	${CC} ${CFLAGS} $^ -o $@ -lm

test: test_read_csv.c data_utils.o
	${CC} ${CFLAGS} $^ -o $@ -lm
	@./test

house: house_data.c data_utils.o
	${CC} ${CFLAGS} $^ -o $@ -lm

$(eval $(foreach f,src,$(echo $(f).o: $(f).c ${CC} ${CFLAGS} -c $@ -o $@ ${LDFLAGS})))

clean:
	rm *.o main test vgcore.*