#include "Dense.hpp"

Dense::Dense(const Activation& act, unsigned int layer_size, bool use_bias)
  : Layer(act, layer_size, use_bias)
{
}

void
Dense::forward(std::vector<double> inputs)
{
    for (unsigned int n = 0; n < this->size(); n++) {
        if (!this->biases.empty())
            this->values[n] = this->biases[n];
        for (unsigned int i = 0; i < inputs.size(); i++) {
            // Sum of weighted outputs from previous layer
            this->values[n] += inputs[i] * this->weights[n][i];
        }

        // Then compute activation
        this->actv_values[n] = this->activation.compute(this->values[n]);
    }
}

void
Dense::backprop(Layer* input_layer, double learning_rate)
{
    for (unsigned int i = 0; i < this->size(); i++) {
        this->delta_errors[i] = this->errors[i] * this->activation.derivative(this->errors[i]);
        // print_vector(this->delta_errors);

        // update = alpha x input x error, for each weight
        double update = this->delta_errors[i] * learning_rate;
        for (unsigned int j = 0; j < input_layer->size(); j++) {
            input_layer->errors[j] += this->delta_errors[i] * this->weights[i][j];
            this->weights[i][j] -= update * input_layer->actv_values[j];
            // printf("Error %f, dError %f, Update %f, Input %f\n",
            //        this->errors[i],
            //        this->delta_errors[i],
            //        update,
            //        input_layer->actv_values[j]);
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
    for (std::vector<double>& row : this->weights) {
        row = std::vector<double>(input_size);
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
    printf("Dense | Size : %u. Input size : %u.\n", this->size(), this->weights[0].size());
}