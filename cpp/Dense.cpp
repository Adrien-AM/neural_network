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
#ifdef DEBUG
    printf("Values :\n");
    print_vector(this->values);
    print_vector(this->actv_values);
#endif
}

void
Dense::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    std::vector<double> derrors = this->activation.derivative(this->values); // dav/dv
#ifdef DEBUG
    printf("Errors :\n");
    print_vector(this->errors);
    printf("Delta (should be 0) :\n");
    print_vector(this->delta_errors);
    printf("Actv derivatives :\n");
    print_vector(derrors);
#endif
    for (unsigned int i = 0; i < this->errors.size(); i++) {
        this->delta_errors[i] += this->errors[i] * derrors[i]; // de/dav

        // this disappears when using -Ofast :)
        if (std::isnan(this->delta_errors[i]) || std::isinf(this->delta_errors[i])) {
            printf("Delta errors diverged.\n");
            exit(0);
        }
    }
#ifdef DEBUG
    printf("After mult : \n");
    print_vector(this->delta_errors);
    printf("---\n");
#endif
    for (unsigned int i = 0; i < this->size(); i++) {

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