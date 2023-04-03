#include "Input.hpp"

Input::Input(Tensor<double> values)
{
    this->output_values = values;
    this->errors = Tensor<double>(values.shape()); // useless but needed
}

void
Input::init(vector<size_t>)
{
}

void
Input::forward(const Tensor<double>& inputs)
{
    this->output_values = inputs;
}

void
Input::backprop(Layer*, double, double)
{
}

void
Input::summarize() const
{
    printf("Input | Size %zu\n", this->output_values.size());
}

size_t
Input::size() const
{
    return this->output_values.total_size();
}

void
Input::reset_values()
{
}

void
Input::reset_errors()
{
}

void
Input::print_layer() const
{
    printf("Input values :\n");
    print_vector(this->output_values);
}