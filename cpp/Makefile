DEBUG ?= 0
PARALLEL ?= 0
CXXFLAGS=-Wall -Wextra -std=c++2a
ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG -g -Og -Werror
else ifeq ($(PARALLEL), 1)
    CXXFLAGS += -DPARALLEL -fopenmp -Ofast
else ifeq ($(PROF), 1)
	CXXFLAGS += -pg
else
	CXXFLAGS += -pg
endif
CXX=g++

exe=main mnist mnist-generator conv-mnist test
test=test-comp-graph test-tensor test-operations test-operations-tensor test-loss test-layers

all: ${exe}
test: ${test}

main: Main.cpp Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Optimizer.o Dataset.o
	${CXX} ${CXXFLAGS} $^ -o $@

mnist: Mnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Metric.o Flatten.o Optimizer.o Dataset.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

mnist-generator: MnistGenerator.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Metric.o Reshape.o Flatten.o Optimizer.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

conv-mnist: ConvMnist.cpp MnistUtils.o Dense.o Layer.o NeuralNetwork.o Loss.o Activation.o Utils.o Input.o Conv2D.o Metric.o Flatten.o Optimizer.o
	${CXX} ${CXXFLAGS} $^ -o $@ -lSDL2

test-comp-graph: TestCompGraph.cpp 
	${CXX} ${CXXFLAGS} $^ -o $@

test-operations: TestOperations.cpp
	${CXX} ${CXXFLAGS} $^ -o $@


test-tensor: TestTensor.cpp Tensor.hpp
	${CXX} ${CXXFLAGS} $^ -o $@

test-operations-tensor: TestOperationsTensor.cpp Tensor.hpp
	${CXX} ${CXXFLAGS} $^ -o $@

test-loss: TestLoss.cpp Tensor.hpp Loss.o
	${CXX} ${CXXFLAGS} $^ -o $@

test-layers: TestLayers.cpp Tensor.hpp Layer.o Activation.o Dense.o Flatten.o
	${CXX} ${CXXFLAGS} $^ -o $@
	
clean:
	rm *.o ${exe} ${test} vgcore.* gmon.out 2>/dev/null