#include <iostream>

#include "Layer.hpp"
#include "Activation.hpp"
#include "Dense.hpp"
#include "Flatten.hpp"
#include "Test.hpp"


void testDense()
{
    start;

    Linear act;
    Dense layer(1, act, true);
    layer.init({2});
    assert(layer.output_values.shape() == vector<size_t>({1}));
    assert(layer.weights.shape() == vector<size_t>({2, 1}));
    assert(layer.biases.shape() == vector<size_t>({1}));

    Tensor<double> x = vector<double>({2, 3});
    x.add_dimension();
    layer.weights({0, 0}) = -2;
    layer.weights({1, 0}) = 1.5;
    layer.biases({0}) = 0.3;
    layer.forward(x);
    assert(close(layer.output_values[0], 0.8));

    layer.output_values.get_graph().backward();
    assert(layer.weights.gradient({0, 0}) == 2);
    assert(layer.weights.gradient({1, 0}) == 3);
    assert(layer.biases.gradient({0}) == 1);
    assert(x.gradient({0, 0}) == -2);
    assert(x.gradient({0, 1}) == 1.5);

    end;
}

void testFlatten()
{
    start;

    Flatten layer;
    Tensor<double> input = vector<size_t>({2, 2});
    input[0] = {1, 2};
    input[1] = {3, 4};
    input.add_dimension();
    layer.init(input.shape());
    assert(layer.output_values.shape() == vector<size_t>({4})); // Flattened to 4 elements

    layer.forward(input);
    assert(layer.output_values[0] == 1);
    assert(layer.output_values[1] == 2);
    assert(layer.output_values[2] == 3);
    assert(layer.output_values[3] == 4);

    // Setting gradients for backpropagation
    layer.output_values.sum().get_graph().backward(); // gradients of 1 for each element
    assert(input.gradient({0, 0, 0}) == 1);
    assert(input.gradient({0, 0, 1}) == 1);
    assert(input.gradient({0, 1, 0}) == 1);
    assert(input.gradient({0, 1, 1}) == 1);

    end;
}


int main()
{
    testDense();
    testFlatten();
    return 0;
}
