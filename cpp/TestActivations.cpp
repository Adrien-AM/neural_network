#include <iostream>
#include <math.h>

#include "Activation.hpp"
#include "Test.hpp"

void
testLinear()
{
    start;

    Linear act;
    Tensor<double> x = vector<double>({ -1, 2.3 });
    Tensor<double> result = act.forward(x);
    assert(result == x);
    result.sum().get_graph().backward();

    assert(close(result.gradient({ 0 }), 1));
    assert(close(result.gradient({ 1 }), 1));

    assert(close(x.gradient({ 0 }), 1));
    assert(close(x.gradient({ 1 }), 1));

    result(0) = 2;
    assert(result != x);

    end;
}

void
testReLU()
{
    start;

    ReLU act;
    Tensor<double> x = vector<double>({ -1, 2.3 });
    Tensor<double> result = act.forward(x);
    
    Tensor<double> expected = vector<double>({ 0, 2.3 });
    assert(result == expected);

    result.sum().get_graph().backward();

    // Gradient for inputs after ReLU: {0, 1} since the derivative of ReLU is 0 for x <= 0 and 1 for x > 0
    assert(close(result.gradient({ 0 }), 1));
    assert(close(result.gradient({ 1 }), 1));

    assert(close(x.gradient({ 0 }), 0));
    assert(close(x.gradient({ 1 }), 1));

    end;
}


void
testSoftmax()
{
    start;

    Softmax act;
    Tensor<double> x = vector<double>({ -1, 2.3 });
    Tensor<double> result = act.forward(x);

    double sumExp = exp(-1) + exp(2.3);
    Tensor<double> expected = vector<double>({ exp(-1) / sumExp, exp(2.3) / sumExp });
    assert(result == expected);

    auto graph = result.sum().get_graph();
    graph.backward();

    assert(close(result.gradient({ 0 }), 1));
    assert(close(result.gradient({ 1 }), 1));

    // Output of softmax is always 1, so changing values of x do not change the result :)
    // Hence gradients are always 0
    assert(close(x.gradient({ 0 }), 0));
    assert(close(x.gradient({ 1 }), 0));


    end;
}


int
main()
{
    testLinear();
    testReLU();
    testSoftmax();
    return 0;
}
