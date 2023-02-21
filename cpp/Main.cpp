#include <vector>
#include <iostream>
#include <random>

#include "Layer.hpp"
#include "Dense.hpp"
#include "Loss.hpp"
#include "NeuralNetwork.hpp"
#include "Activation.hpp"

double f(double x)
{
    return 0.4 * x - 1.3;
}

int
main()
{
    Linear act;
    std::vector<Layer*> layers = {new Dense(act, 3), new Dense(act, 1)};
    Loss l = mse();
    NeuralNetwork nn(3, layers, l);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);
    std::vector<std::vector<double>> inputs(300);

    for (unsigned int i = 0; i < inputs.size(); ++i) {
        inputs[i] = { dis(gen) };
    }

    std::vector<std::vector<double>> outputs(inputs.size());
    for (unsigned int i = 0; i < inputs.size(); i++) {
        outputs[i] = {f(inputs[i][0])};
    }
    nn.fit(inputs, outputs, 1e-3, 5);
    double newrand = dis(gen);
    printf("f(%f) -> %f (should be %f)\n", newrand, nn.predict({ newrand })[0], f(newrand));

    return 0;
}
