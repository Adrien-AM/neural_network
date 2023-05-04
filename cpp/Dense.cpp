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
    cout << "--Layer--\n";
    this->weights.print();
    cout << "--------\n" << endl;
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
        double& value = this->values[n];
        if (!this->biases.empty())
            value = this->biases[n];
        Tensor<double> weight_n = this->weights.at(n);
        for (size_t i = 0; i < input_size; i++) {
            // Sum of weighted outputs from previous layer
            value += inputs[i] * weight_n[i];
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
    values.reset_data();
    output_values.reset_data();
}

void
Dense::reset_errors()
{
    errors.reset_data();
    delta_errors.reset_data();
}

Tensor<double>
Dense::backprop(Layer* input_layer)
{
    size_t size = this->size();
    vector<size_t> input_shape = input_layer->output_values.shape();

    Tensor<double> jacobian = this->activation.derivative(this->output_values); // dav/dv

    for (size_t j = 0; j < size; j++) {
        double& ej = this->errors[j];
        if (ej != 0) {
            for (size_t i = 0; i < size; i++) {
                this->delta_errors[i] += ej * jacobian.at(i)[j]; // de/dav
            }
        }
    }

    double* input_errors = input_layer->errors.data();
    size_t input_errors_size = input_layer->errors.total_size();
    double* input_outputs = input_layer->output_values.data();
    Tensor<double> gradients(weights.shape());
    for (size_t i = 0; i < size; i++) {
        double derror = this->delta_errors[i];
        Tensor<double> weight_i = this->weights.at(i);
        Tensor<double> gradients_i = gradients.at(i);
        for (size_t j = 0; j < input_errors_size; j++) {
            input_errors[j] += derror * weight_i[j];
            gradients_i[j] = derror * input_outputs[j];
        }
        if (!this->biases.empty()) {
            this->biases[i] -= derror * 1e-3; // temporary
        }
    }

    return gradients;
}

void
Dense::init(vector<size_t> input_shape)
{
    this->weights = vector<size_t>{ this->size(), input_shape[0] };

    random_device rd;
    mt19937 gen(rd()); // Mersenne Twister engine
    // normal_distribution<double> initializer(0, 0.3);
    double var = sqrt(6 / (double)(input_shape.size() + this->size()));
    uniform_real_distribution<double> initializer(-var, var);
    size_t size = this->size();
    for (size_t n = 0; n < size; n++) {
        Tensor<double> weight_n = this->weights.at(n);
        for (size_t p = 0; p < input_shape[0]; p++) {
            weight_n[p] = initializer(gen);
        }
    }
}

void
Dense::summarize() const
{
    printf("Dense | Size : %zu. Input size : %zu.\n", this->size(), this->weights.shape()[0]);
}

Dense::~Dense() {}