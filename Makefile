CC=gcc
CFLAGS=-Werror -Wall -Wextra -std=c2x

nn: utils.o nn.c
	${CC} ${CFLAGS} $^ -o $@ -lm

utils.o: utils.c
	${CC} ${CFLAGS} -c utils.c -lm

clean:
	rm *.o nn