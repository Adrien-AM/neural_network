#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__

#include "Input.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include <vector>

class NeuralNetwork
{
  private:
    unsigned int input_size;
    std::vector<Layer*> layers;
    Loss loss;
    double alpha;
    double gamma;

    std::vector<double> feed_forward(const std::vector<double>& inputs);
    void backpropagation(const std::vector<double>&, const std::vector<double>&);
    void reset_values();
    void reset_errors();

  public:
    NeuralNetwork(unsigned int input_size, std::vector<Layer*> layers, Loss loss);
    void fit(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& outputs,
             double learning_rate,
             double momentum,
             unsigned int epochs);
    std::vector<double> predict(const std::vector<double>& inputs);
    double evaluate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& outputs,
                    Loss loss);
    void summarize();
    ~NeuralNetwork();
};

#endif // __NEURALNETWORK_HPP__