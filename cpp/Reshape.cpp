#include "Reshape.hpp"

void
Reshape::init(vector<size_t> input_shape)
{
    size_t input_size = 1;
    for (auto& dim : input_shape) {
        input_size *= dim;
    }
    size_t output_size = 1;
    for (auto& dim : shape) {
        output_size *= dim;
    }
    if (output_size != input_size) {
        throw length_error("Cannot reshape because total sizes are different.");
    }

    this->input_shape = input_shape;
}

Tensor<double>
Reshape::forward(const Tensor<double>& input) const
{
    Tensor<double> result(shape);
    input.copy_data(result);
    return result;
}

void
Reshape::summarize() const
{
    printf("Reshape shape (");
    for (auto& dim : input_shape)
        printf("%zu,", dim);
    printf(") into shape (");
    for (auto& dim : shape)
        printf("%zu,", dim);
    printf(").\n");
}

Reshape* Reshape::clone() const
{
    Reshape* copy = new Reshape(shape);
    copy->input_shape = input_shape;
    return copy;
}

vector<size_t> Reshape::output_shape() const
{
    return this->shape;
}

// debug
// TODO
void
Reshape::print_layer() const
{
}