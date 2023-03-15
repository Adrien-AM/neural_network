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

    void reset_values();
    void reset_errors();
    void backpropagation(const std::vector<double>&, const std::vector<double>&);

  public:
    /*
     * Create a Neural Network
     * @param input_size : size of each data sample fed as input (only 1D)
     * @param layers
     * @param loss : loss function used as evaluation and in GD
     */
    NeuralNetwork(unsigned int input_size, std::vector<Layer*> layers, Loss loss);
    /*
    * Train a Neural Network on data
    * @param inputs : data
    * @param outputs : groundtruth
    * @param learning_rate
    * @param momentum : high momentum = high impact of previous updates
    * @param epochs
    */
    void fit(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& outputs,
             double learning_rate,
             double momentum,
             unsigned int epochs);
    
    /*
    * Forward pass of Neural Net
    * @param inputs : data
    */
    std::vector<double> predict(const std::vector<double>& inputs);

    /*
    * Evaluate Neural Net on new dataset
    * @param inputs : data
    * @param outputs : data
    * @param loss
    */
    double evaluate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& outputs,
                    Loss loss);

    /*
    * Show a brief description of Neural Network's architecture
    */
    void summarize();
    
    ~NeuralNetwork();
};

#endif // __NEURALNETWORK_HPP__