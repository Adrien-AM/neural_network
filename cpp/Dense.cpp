#include "Dense.hpp"

Dense::Dense(unsigned int layer_size, const Activation& act, bool use_bias)
  : activation(act)
{
    this->values = std::vector<double>(layer_size);
    this->errors = std::vector<double>(layer_size);
    this->delta_errors = std::vector<double>(layer_size);
    this->output_values = std::vector<double>(layer_size);

    this->updates = std::vector<std::vector<double>>(layer_size);
    this->weights = std::vector<std::vector<double>>(layer_size);
    if (use_bias)
        this->biases = std::vector<double>(layer_size);
    else
        this->biases = std::vector<double>(0);
};

void
Dense::print_layer() const
{
    std::cout << "--Layer--\n";
    for (auto& row : this->weights) {
        print_vector(row);
    }
    std::cout << "--------\n" << std::endl;
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
    this->output_values = this->activation.compute(this->values);
}

unsigned int
Dense::size() const
{
    return this->values.size();
}

void
Dense::reset_values()
{
    for (unsigned int i = 0; i < this->values.size(); i++) {
        this->values[i] = 0;
        this->output_values[i] = 0;
    }
}

void
Dense::reset_errors()
{
    for (unsigned int i = 0; i < this->values.size(); i++) {
        this->errors[i] = 0;
        this->delta_errors[i] = 0;
    }
}

void
Dense::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    unsigned int size = this->size();
    unsigned int input_size = input_layer->size();

    std::vector<std::vector<double>> jacobian =
      this->activation.derivative(this->output_values); // dav/dv

    for (unsigned int j = 0; j < size; j++) {
        double& ej = this->errors[j];
        if (ej != 0) {
            #ifdef PARALLEL
            #pragma omp parallel for
            #endif
            for (unsigned int i = 0; i < size; i++) {
                this->delta_errors[i] += ej * jacobian[i][j]; // de/dav
            }
        }
    }

    for (unsigned int i = 0; i < size; i++) {
        // update = alpha x input x error, for each weight
        double update = this->delta_errors[i] * learning_rate;
        #ifdef PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int j = 0; j < input_size; j++) {
            input_layer->errors[j] += this->delta_errors[i] * this->weights[i][j];
            this->updates[i][j] = (momentum * this->updates[i][j]) +
                                  (1 - momentum) * update * input_layer->output_values[j];
            this->weights[i][j] -= this->updates[i][j];
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
    bool use_bias = !this->biases.empty();

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 0.3);
    unsigned int size = this->weights.size();
    for (unsigned int n = 0; n < size; n++) {
        this->weights[n] = std::vector<double>(input_size);
        this->updates[n] = std::vector<double>(input_size);
        for (unsigned int p = 0; p < input_size; p++) {
            this->weights[n][p] = normal(gen);
        }
        if (use_bias) {
            this->biases[n] = normal(gen);
        }
    }
}

void
Dense::summarize() const
{
    printf("Dense | Size : %u. Input size : %zu.\n", this->size(), this->weights[0].size());
}

Dense::~Dense() {}