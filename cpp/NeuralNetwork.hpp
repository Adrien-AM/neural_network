#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__

#include "Input.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "Metric.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

class NeuralNetwork
{
  private:
    unsigned int input_size;
    vector<Layer*> layers;
    Loss loss;
    double alpha;
    double gamma;

    void reset_values();
    void reset_errors();
    vector<double> feed_forward(const vector<double>& inputs);
    void backpropagation(const vector<double>&, const vector<double>&);

  public:
    /*
     * Create a Neural Network
     * @param input_size : size of each data sample fed as input (only 1D)
     * @param layers
     * @param loss : loss function used as evaluation and in GD
     */
    NeuralNetwork(unsigned int input_size, vector<Layer*> layers, Loss loss);
    /*
    * Train a Neural Network on data
    * @param inputs : data
    * @param outputs : groundtruth
    * @param learning_rate
    * @param momentum : high momentum = high impact of previous updates
    * @param batch_size : number of samples to backpropagate per epoch
    * @param epochs
    */
    void fit(const vector<vector<double>>& inputs,
             const vector<vector<double>>& outputs,
             double learning_rate,
             double momentum,
             size_t batch_size,
             unsigned int epochs);

    /*
    * Forward pass of Neural Net
    * @param inputs : data
    */
    vector<double> predict(const vector<double>& inputs);

    /*
    * Evaluate Neural Net on new dataset
    * @param inputs : data
    * @param outputs : data
    * @param loss
    */
    double evaluate(const vector<vector<double>>& inputs,
                    const vector<vector<double>>& outputs,
                    Loss loss, Metric *metric);

    /*
    * Show a brief description of Neural Network's architecture
    */
    void summarize();
    
    ~NeuralNetwork();
};

#endif // __NEURALNETWORK_HPP__