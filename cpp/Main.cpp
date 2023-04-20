
#include <iostream>
#include <random>

#include "Activation.hpp"
#include "Dense.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "NeuralNetwork.hpp"

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
    vector<Layer*> layers = { new Dense(1, act, false), new Dense(1, act, true) };
    MeanSquaredError mse;
    NeuralNetwork nn({ 1 }, layers, mse);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);
    Tensor<double> inputs(vector<size_t>{ 500, 1 });

    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs.at(i) = { dis(gen) };
    }

    Tensor<double> outputs(vector<size_t>({ 500, 1 }));
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs.at(i) = { f(inputs.at(i)[0]) };
    }
    nn.fit(inputs, outputs, 1e-4, 0, inputs.size(), 100);
    double newrand = dis(gen);
    vector<double> input = { newrand };
    printf("f(%f) -> %f (should be %f)\n", newrand, nn.predict(input)[0], f(newrand));

    return 0;
}