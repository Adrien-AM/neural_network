#include "Flatten.hpp"

void
Flatten::init(vector<size_t> input_shape)
{
    this->input_shape = input_shape;

    size_t size = 1;
    for (int dim = input_shape.size() - 1; dim >= 0; dim--) {
        size *= input_shape[dim];
    }

    this->output_values = Tensor<double>(size);
    this->errors = Tensor<double>(size);
}

void
Flatten::forward(const Tensor<double>& input)
{
    input.copy_data(this->output_values);
}

void
Flatten::backprop(Layer* input_layer, double alpha, double beta)
{
    (void)alpha;
    (void)beta;
    this->errors.copy_data(input_layer->errors);
}

void
Flatten::summarize() const
{
    printf("Flatten layer : output of size %zu.\n", this->output_values.size());
}

size_t
Flatten::size() const
{
    return this->output_values.size();
}

void
Flatten::reset_values()
{
    memset(this->output_values.data(), 0, this->output_values.size() * sizeof(double));
}

void
Flatten::reset_errors()
{
    memset(this->errors.data(), 0, this->output_values.size() * sizeof(double));
}

// debug
// TODO
void
Flatten::print_layer() const
{
}