#include "Flatten.hpp"

void
Flatten::init(vector<size_t> input_shape)
{
    this->input_shape = input_shape;

    size_t size = 1;
    for (int dim = input_shape.size() - 1; dim >= 0; dim--) {
        size *= input_shape[dim];
    }

    this->output_size = size;
}

Tensor<double>
Flatten::forward(const Tensor<double>& input) const
{
    Tensor<double> result(this->output_size);
    input.copy_data(result);
    return result;
}

vector<size_t>
Flatten::output_shape() const
{
    return { this->output_size };
}

void
Flatten::summarize() const
{
    printf("Flatten layer : output of size %zu.\n", this->output_size);
}

Flatten*
Flatten::clone() const
{
    Flatten* copy = new Flatten();
    copy->init(this->input_shape);
    return copy;
}

// debug
// TODO
void
Flatten::print_layer() const
{
}