
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
    return 0.4 * x - 13;
}

int
main()
{
    Linear act;
    vector<Layer*> layers = { new Dense(1, act, true) };
    Loss mse = MeanSquaredError();
    SGD optimizer(1e-2);
    NeuralNetwork nn({ 1 }, layers, mse, optimizer);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);
    Tensor<double> inputs(vector<size_t>{ 100, 1 });

    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs.at(i) = { dis(gen) };
    }

    Tensor<double> outputs(vector<size_t>({ 100, 1 }));
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs.at(i) = { f(inputs.at(i)[0]) };
    }
    nn.fit(inputs, outputs, inputs.size(), 200);
    double newrand = dis(gen);
    vector<double> input = { newrand };
    printf("f(%f) -> %f (should be %f)\n", newrand, nn.predict(input)[0], f(newrand));

    return 0;
}