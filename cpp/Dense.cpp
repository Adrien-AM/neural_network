#include "Dense.hpp"

Dense::Dense(unsigned int layer_size, const Activation& act, bool use_bias)
  : Layer()
  , activation(act)
{
    this->values = std::vector<double>(layer_size);
    this->output_values = std::vector<double>(layer_size);
    this->errors = std::vector<double>(layer_size);
    this->delta_errors = std::vector<double>(layer_size);
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

unsigned int
Dense::size() const
{
    return this->values.size();
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
Dense::forward(const std::vector<double>& inputs)
{
    for (unsigned int n = 0; n < this->size(); n++) {
        if (!this->biases.empty())
            this->values[n] = this->biases[n];
        else
            this->values[n] = 0;

        for (unsigned int i = 0; i < inputs.size(); i++) {
            // Sum of weighted outputs from previous layer
            this->values[n] += inputs[i] * this->weights[n][i];
        }
    }

    // Then compute activation
    this->output_values = this->activation.compute(this->values);
#ifdef DEBUG
    // printf("Values :\n");
    // print_vector(this->values);
    // print_vector(this->output_values);
#endif
}

void
Dense::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    std::vector<std::vector<double>> jacobian =
      this->activation.derivative(this->output_values); // dav/dv

    for (unsigned int i = 0; i < this->errors.size(); i++) {
        double& de = this->delta_errors[i];
        de = 0;
        for (unsigned int j = 0; j < this->errors.size(); j++) {
            de += this->errors[j] * jacobian[i][j]; // de/dav
        }
    }

    for (unsigned int j = 0; j < input_layer->size(); j++) {
        double& input_error = input_layer->errors[j];
        input_error = 0;
        for (unsigned int i = 0; i < this->size(); i++) {
            double update = this->delta_errors[i] * learning_rate;
            // update = alpha x input x error, for each weight
            input_error += this->delta_errors[i] * this->weights[i][j];
            this->updates[i][j] = (momentum * this->updates[i][j]) +
                                  (1 - momentum) * update * input_layer->output_values[j];
            this->weights[i][j] -= this->updates[i][j];
        }
    }

    // Then update bias
    for (unsigned int i = 0; i < this->size(); i++) {
        if (!this->biases.empty()) {
            this->biases[i] -= this->delta_errors[i] * learning_rate;
        }
    }
    return;
}

void
Dense::summarize() const
{
    printf("Dense | Size : %u. Input size : %zu.\n", this->size(), this->weights[0].size());
}

Dense::~Dense() {}