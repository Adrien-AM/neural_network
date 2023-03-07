#include "Conv2D.hpp"

Conv2D::Conv2D(unsigned int features,
               unsigned int kernel_size,
               unsigned int input_width,
               const Activation& act,
               bool use_bias)
  : features_size(features)
  , kernel_size(kernel_size)
  , input_width(input_width)
  , activation(act)
{
    this->weights = std::vector<std::vector<double>>(features);
    for (unsigned int i = 0; i < features; i++) {
        this->weights[i] = std::vector<double>(kernel_size * kernel_size);
    }
    if (use_bias)
        this->biases = std::vector<double>(features);
    else
        this->biases = std::vector<double>();
}

void
Conv2D::init(unsigned int input_size)
{
    if (input_size % this->kernel_size != 0) {
        printf("Padding not supported yet.\n");
        exit(0);
    }
    this->output_values = std::vector<double>(this->features_size * input_size);
    this->values = std::vector<double>(this->features_size * input_size);

    for (unsigned int i = 0; i < this->size(); i++) {
        this->weights[i] = std::vector<double>(this->kernel_size * this->kernel_size);
        // this->updates[i] = std::vector<double>(this->kernel_size * this->kernel_size); //
        // momentum later
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
Conv2D::forward(const std::vector<double>& inputs)
{

    // wrong because output is multi dimensional and not 2D :) change this.values
    for (unsigned int n = 0; n < this->features_size; n++) {
        for (unsigned int inp = 0; inp < inputs.size(); inp++) {
            for (unsigned int kh = 0; kh < this->kernel_size; kh++) {
                for (unsigned int kw = 0; kw < this->kernel_size; kw++) {
                    this->values[n * this->features_size + inp] +=
                      inputs[inp + kw + (this->input_width * kh)] *
                      this->weights[n][kw + this->kernel_size * kh];
                }
            }
        }
    }
}

void
Conv2D::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    (void)momentum;
    (void)input_layer;
    for (unsigned int n = 0; n < this->features_size; n++) {
        for (unsigned int e = 0; e < (this->size() / this->features_size); e++) {
            this->biases[n] -= learning_rate * this->errors[n * this->features_size + e];
        }
    }
}

void
Conv2D::summarize() const
{
    printf("Conv2D | Size : %u. Kernel size : %u.\n", this->size(), this->kernel_size);
}

unsigned int
Conv2D::size() const
{
    return this->output_values.size();
}

void
Conv2D::reset_values()
{
}

void
Conv2D::reset_errors()
{
}

void
Conv2D::print_layer() const
{
    // TODO
}

Conv2D::~Conv2D() {}