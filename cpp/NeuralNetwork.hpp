#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__

#include "Layer.hpp"
#include "Loss.hpp"
#include "Input.hpp"
#include <vector>

class NeuralNetwork
{
  private:
    unsigned int input_size;
    std::vector<Layer*> layers;
    Loss loss;
    double alpha;

    std::vector<double> feed_forward(std::vector<double> inputs);
    void backpropagation(std::vector<double>, std::vector<double>);
    void reset_values();
    void reset_errors();

  public:
    NeuralNetwork(unsigned int input_size, std::vector<Layer*> layers, Loss l);
    void fit(std::vector<std::vector<double>> inputs,
             std::vector<std::vector<double>> outputs,
             double learning_rate,
             unsigned int epochs);
    std::vector<double> predict(std::vector<double> inputs);
    ~NeuralNetwork();
};

#endif // __NEURALNETWORK_HPP__