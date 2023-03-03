#include "Dense.hpp"

Dense::Dense(unsigned int layer_size, const Activation& act, bool use_bias)
  : Layer(layer_size, act, use_bias)
{
}

void
Dense::forward(const std::vector<double>& inputs)
{
    for (unsigned int n = 0; n < this->size(); n++) {
        if (!this->biases.empty())
            this->values[n] = this->biases[n];
        for (unsigned int i = 0; i < inputs.size(); i++) {
            // Sum of weighted outputs from previous layer
            this->values[n] += inputs[i] * this->weights[n][i];
        }
    }

    // Then compute activation
    this->actv_values = this->activation.compute(this->values);
}

void
Dense::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    unsigned int size = this->size();
    std::vector<std::vector<double>> jacobian =
      this->activation.derivative(this->actv_values); // dav/dv
      
    for (unsigned int i = 0; i < size; i++) {
        double& de = this->delta_errors[i];
        for (unsigned int j = 0; j < size; j++) {
            de += this->errors[j] * jacobian[i][j]; // de/dav
        }
    }
    for (unsigned int i = 0; i < size; i++) {

        // update = alpha x input x error, for each weight
        double update = this->delta_errors[i] * learning_rate;
        for (unsigned int j = 0; j < input_layer->size(); j++) {
            input_layer->errors[j] += this->delta_errors[i] * this->weights[i][j];
            this->updates[i][j] = (momentum * this->updates[i][j]) +
                                  (1 - momentum) * update * input_layer->actv_values[j];
            this->weights[i][j] -= this->updates[i][j];
            // #ifdef DEBUG
            // printf("Layer size : %u, Error %f, dError %f, Update %f, Input %f\n",
            //        this->size(),
            //        this->errors[i],
            //        this->delta_errors[i],
            //        this->updates[i][j],
            //        input_layer->actv_values[j]);
            // #endif
        }
        if (!this->biases.empty()) {
            this->biases[i] -= this->delta_errors[i] * learning_rate;
        }
    }
    return;
}

void
Dense::init(unsigned int input_size)
{
    for (unsigned int i = 0; i < this->size(); i++) {
        this->weights[i] = std::vector<double>(input_size);
        this->updates[i] = std::vector<double>(input_size);
    }
    bool use_bias = !this->biases.empty();

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 1);
    for (unsigned int n = 0; n < this->weights.size(); n++) {
        for (unsigned int p = 0; p < this->weights[n].size(); p++) {
            this->weights[n][p] = normal(gen);
        }
        if (use_bias) {
            this->biases[n] = normal(gen);
        }
    }
}

void
Dense::summarize()
{
    printf("Dense | Size : %u. Input size : %zu.\n", this->size(), this->weights[0].size());
}