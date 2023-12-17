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
}

void
Flatten::forward(const Tensor<double>& input)
{
    input.copy_data(this->output_values);
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
    this->output_values.reset_data();
}

Flatten* Flatten::clone() const
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