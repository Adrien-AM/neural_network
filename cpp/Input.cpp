#include "Input.hpp"

Input::Input(std::vector<double> values)
{
    this->output_values = values;
    this->errors = std::vector<double>(values.size()); // useless but needed
}

void
Input::init(unsigned int)
{
}

void
Input::forward(const std::vector<double>& inputs)
{
    for (unsigned int i = 0; i < inputs.size(); i++) {
        this->output_values[i] = inputs[i];
    }
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

unsigned int
Input::size() const
{
    return this->output_values.size();
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