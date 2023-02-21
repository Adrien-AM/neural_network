#include "Dense.hpp"

Dense::Dense(const Activation& act, unsigned int layer_size)
  : Layer(act, layer_size)
{
}

void
Dense::forward(std::vector<double> inputs)
{
    for (unsigned int n = 0; n < this->size(); n++) {
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
        this->delta_errors[i] += this->errors[i] * this->activation.derivative(this->errors[i]);
        // print_vector(this->delta_errors);

        // update = alpha x input x error, for each weight
        double update = this->delta_errors[i] * learning_rate;
        for (unsigned int j = 0; j < input_layer->size(); j++) {
            input_layer->errors[j] += this->delta_errors[i] * this->weights[i][j];
            this->weights[i][j] -= update * input_layer->actv_values[j];
            // printf("Error %f, Update %f, Input %f\n",
            //        this->delta_errors[i],
            //        update,
            //        input_layer->actv_values[j]);
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

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 1);
    for (auto& row : this->weights) {
        for (double& w : row) {
            w = normal(gen);
        }
    }
}