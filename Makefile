CC=gcc
CFLAGS=-Werror -Wall -Wextra -std=c2x
LDFLAGS=-lm

src=utils neural_network

main: main.c neural_network.o utils.o
	${CC} ${CFLAGS} $^ -o $@ -lm

test: test_read_csv.c utils.o
	${CC} ${CFLAGS} $^ -o $@ -lm
	@./test

$(eval $(foreach f,src,$(echo $(f).o: $(f).c ${CC} ${CFLAGS} -c $@ -o $@ ${LDFLAGS})))

clean:
	rm *.o main test