#include "Reshape.hpp"

void
Reshape::init(vector<size_t> input_shape)
{
    size_t input_size = 1;
    for (auto& dim : input_shape) {
        input_size *= dim;
    }
    size_t output_size = 1;
    for (auto& dim : output_shape) {
        output_size *= dim;
    }
    if (output_size != input_size) {
        throw length_error("Cannot reshape because total sizes are different.");
    }

    this->input_shape = input_shape;

    this->output_values = Tensor<double>(output_shape);
    this->errors = Tensor<double>(output_shape);
}

void
Reshape::forward(const Tensor<double>& input)
{
    input.copy_data(this->output_values);
}

void
Reshape::backprop(Layer* input_layer, double alpha, double beta)
{
    (void)alpha;
    (void)beta;
    this->errors.copy_data(input_layer->errors);
}

void
Reshape::summarize() const
{
    printf("Reshape shape (");
    for (auto& dim : input_shape)
        printf("%zu,", dim);
    printf(") into shape (");
    for (auto& dim : output_shape)
        printf("%zu,", dim);
    printf(").\n");
}

size_t
Reshape::size() const
{
    return this->output_values.size();
}

void
Reshape::reset_values()
{
    memset(this->output_values.data(), 0, this->output_values.size() * sizeof(double));
}

void
Reshape::reset_errors()
{
    memset(this->errors.data(), 0, this->output_values.size() * sizeof(double));
}

// debug
// TODO
void
Reshape::print_layer() const
{
}