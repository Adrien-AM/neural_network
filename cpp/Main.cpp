#include <vector>

#include "Layer.hpp"
#include "Loss.hpp"
#include "NeuralNetwork.hpp"
#include "Neuron.hpp"

int
main()
{
    Loss l = mae();
    std::vector<Neuron*> neurons = {};
    std::vector<Layer*> layers = {};
    NeuralNetwork* nn = new NeuralNetwork(layers, l);

    delete nn;

    return 0;
}
