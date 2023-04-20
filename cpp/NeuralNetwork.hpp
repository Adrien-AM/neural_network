#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__

#include "Input.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "Metric.hpp"


#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

class NeuralNetwork
{
  private:
    vector<size_t> input_shape;
    vector<Layer*> layers;
    Loss& loss;
    double alpha;
    double gamma;

    void reset_values();
    void reset_errors();
    Tensor<double> feed_forward(const Tensor<double>& inputs);
    void backpropagation(const Tensor<double>&, const Tensor<double>&);

  public:
    /*
     * Create a Neural Network
     * @param input_size : size of each data sample fed as input (only 1D)
     * @param layers
     * @param loss : loss function used as evaluation and in GD
     */
    NeuralNetwork(vector<size_t> input_shape, vector<Layer*> layers, Loss& loss);
    /*
    * Train a Neural Network on data
    * @param inputs : data
    * @param outputs : groundtruth
    * @param learning_rate
    * @param momentum : high momentum = high impact of previous updates
    * @param batch_size : number of samples to backpropagate per epoch
    * @param epochs
    */
    void fit(const Tensor<double>& inputs,
             const Tensor<double>& outputs,
             double learning_rate,
             double momentum,
             size_t batch_size,
             size_t epochs);

    /*
    * Forward pass of Neural Net
    * @param inputs : data
    */
    Tensor<double> predict(const Tensor<double>& inputs);

    /*
    * Evaluate Neural Net on new dataset
    * @param inputs : data
    * @param outputs : data
    * @param loss
    */
    double evaluate(const Tensor<double>& inputs,
                    const Tensor<double>& outputs,
                    Loss& loss, Metric *metric);

    /*
    * Show a brief description of Neural Network's architecture
    */
    void summarize();
    
    ~NeuralNetwork();
};

#endif // __NEURALNETWORK_HPP__