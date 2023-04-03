#include "Dense.hpp"

Dense::Dense(size_t layer_size, const Activation& act, bool use_bias)
  : activation(act)
{
    this->values = Tensor<double>(layer_size);
    this->errors = Tensor<double>(layer_size);
    this->delta_errors = Tensor<double>(layer_size);
    this->output_values = Tensor<double>(layer_size);

    if (use_bias)
        this->biases = Tensor<double>(layer_size);
    else
        this->biases = Tensor<double>();
};

void
Dense::print_layer() const
{
    std::cout << "--Layer--\n";
    this->weights.print();
    std::cout << "--------\n" << std::endl;
}

void
Dense::forward(const Tensor<double>& inputs)
{
    size_t size = this->size();
    size_t input_size = inputs.size();

    #ifdef PARALLEL
    #pragma omp parallel for
    #endif
    for (size_t n = 0; n < size; n++) {
        if (!this->biases.empty())
            this->values[n] = this->biases[n];
        Tensor<double> weight_n = this->weights.at(n);
        for (size_t i = 0; i < input_size; i++) {
            // Sum of weighted outputs from previous layer
            this->values[n] += inputs[i] * weight_n[i];
        }
    }

    // Then compute activation
    this->output_values = this->activation.compute(this->values);
}

size_t
Dense::size() const
{
    return this->values.size();
}

void
Dense::reset_values()
{
    for (size_t i = 0; i < this->values.size(); i++) {
        this->values[i] = 0;
        this->output_values[i] = 0;
    }
}

void
Dense::reset_errors()
{
    for (size_t i = 0; i < this->values.size(); i++) {
        this->errors[i] = 0;
        this->delta_errors[i] = 0;
    }
}

void
Dense::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    size_t size = this->size();
    vector<size_t> input_shape = input_layer->output_values.shape();

    Tensor<double> jacobian =
      this->activation.derivative(this->output_values); // dav/dv

    for (size_t j = 0; j < size; j++) {
        double& ej = this->errors[j];
        if (ej != 0) {
            #ifdef PARALLEL
            #pragma omp parallel for
            #endif
            for (size_t i = 0; i < size; i++) {
                this->delta_errors[i] += ej * jacobian.at(i)[j]; // de/dav
            }
        }
    }

    for (size_t i = 0; i < size; i++) {
        // update = alpha x input x error, for each weight
        double update = this->delta_errors[i] * learning_rate;
        #ifdef PARALLEL
        #pragma omp parallel for
        #endif
        Tensor<double> update_i = this->updates.at(i);
        Tensor<double> weight_i = this->weights.at(i);
        double* input_errors = input_layer->errors.data();
        double* input_outputs = input_layer->output_values.data();
        for (size_t j = 0; j < input_layer->errors.total_size(); j++) {
            input_errors[j] += this->delta_errors[i] * weight_i[j];
            update_i[j] = (momentum * update_i[j]) +
                                  (1 - momentum) * update * input_outputs[j];
            weight_i[j] -= update_i[j];
        }
        if (!this->biases.empty()) {
            this->biases[i] -= this->delta_errors[i] * learning_rate;
        }
    }
    return;
}

void
Dense::init(vector<size_t> input_shape)
{
    bool use_bias = !this->biases.empty();

    this->weights = vector<size_t>{ this->size(), input_shape[0] };
    this->updates = vector<size_t>{ this->size(), input_shape[0] };

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 0.3);
    size_t size = this->size();
    for (size_t n = 0; n < size; n++) {
        Tensor<double> weight_n = this->weights.at(n);
        for (size_t p = 0; p < input_shape[0]; p++) {
            weight_n[p] = normal(gen);
        }
        if (use_bias) {
            this->biases[n] = normal(gen);
        }
    }
}

void
Dense::summarize() const
{
    printf("Dense | Size : %zu. Input size : %zu.\n", this->size(), this->weights.shape()[0]);
}

Dense::~Dense() {}