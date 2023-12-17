CXX=g++
CXXFLAGS=-Werror -Wall -Wextra -std=c++2a -g
LDFLAGS=-lm

exe=main

main: Main.cpp
	${CXX} ${CXXFLAGS} $^ -o $@ ${LDFLAGS}

clean:
	rm *.o ${exe} vgcore.* gmon.out