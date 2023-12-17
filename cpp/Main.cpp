#include <iostream>
#include <random>

#include "Activation.hpp"
#include "Dense.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "NeuralNetwork.hpp"
#include "Optimizer.hpp"

using namespace std;

double
f(double x)
{
    return 0.4 * x - 12;
}

int
main()
{
    Linear act;
    vector<Layer*> layers = { new Dense(1, act, true) };
    MeanSquaredError loss;
    SGD optimizer(1e-2);
    NeuralNetwork nn({ 1 }, layers, loss, optimizer);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dis(-10, 10);
    Tensor<double> inputs(vector<size_t>{ 100, 1, 1 });

    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs({ i, 0, 0 }) = dis(gen);
    }

    Tensor<double> outputs(vector<size_t>({ 100, 1 }));
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i] = { f(inputs({ i, 0, 0 })) };
    }
    nn.fit(inputs, outputs, 50, 500);
    double newrand = dis(gen);
    Tensor<double> input(vector<size_t>{ 1, 1 });
    input({ 0, 0 }) = newrand;
    printf("f(%f) -> %f (should be %f)\n", newrand, nn.predict(input)({ 0, 0 }), f(newrand));

    return 0;
}