#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__

#include "Layer.hpp"
#include "Loss.hpp"
#include <vector>

class NeuralNetwork
{
  private:
    std::vector<Layer*> layers;
    Loss loss;

  public:
    NeuralNetwork(std::vector<Layer*> layers, Loss l)
      : layers(layers)
      , loss(l){};
    ~NeuralNetwork();
};

#endif // __NEURALNETWORK_HPP__