#include "Dense.hpp"

Dense::Dense(unsigned int layer_size, const Activation& act, bool use_bias)
  : Layer(layer_size, act, use_bias)
{
}

void
Dense::forward(const std::vector<double>& inputs)
{
    unsigned int size = this->size();
    unsigned int input_size = inputs.size();
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (unsigned int n = 0; n < size; n++) {
        if (!this->biases.empty())
            this->values[n] = this->biases[n];
        for (unsigned int i = 0; i < input_size; i++) {
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
    unsigned int input_size = input_layer->size();
    std::vector<std::vector<double>> jacobian =
      this->activation.derivative(this->actv_values); // dav/dv

#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (unsigned int j = 0; j < size; j++) {
        double& ej = this->errors[j];
        if (ej != 0) {
            for (unsigned int i = 0; i < size; i++) {
                this->delta_errors[i] += ej * jacobian[i][j]; // de/dav
            }
        }
    }

#ifdef PARALLEL
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < size; i++) {
        // update = alpha x input x error, for each weight
        double update = this->delta_errors[i] * learning_rate;
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned int j = 0; j < input_size; j++) {
#ifdef PARALLEL
            omp_set_lock(&lock);
#endif
            input_layer->errors[j] += this->delta_errors[i] * this->weights[i][j];
#ifdef PARALLEL
            omp_unset_lock(&lock);
#endif
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
#ifdef PARALLEL
    omp_destroy_lock(&lock);
#endif
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
    std::normal_distribution<double> normal(0, 0.3);
    unsigned int size = this->weights.size();
    unsigned int w_size = this->weights[0].size(); // Assume equally dimensioned
    for (unsigned int n = 0; n < size; n++) {
        for (unsigned int p = 0; p < w_size; p++) {
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