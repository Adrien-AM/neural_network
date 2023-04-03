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
    size_t size = this->output_values.size();
    vector<Tensor<double>> curr(input.shape().size() - 1);
    for (size_t dim = 0; dim < curr.size(); dim++) {
        curr[dim] = (dim == 0 ? input : curr[dim - 1]).at(0);
    }

    vector<size_t> indices(input.shape().size());
    for (size_t i = 0; i < size; i++) {
        size_t dim = indices.size() - 1;
        while (dim != 0 && indices[dim] == input_shape[dim]) {
            indices[dim] = 0;
            dim--;
            indices[dim]++;
        }
        for (size_t d = 0; d < curr.size(); d++) {
            curr[d] = (d == 0 ? input : curr[d - 1]).at(indices[d]);
        }
        this->output_values[i] = curr.back()[indices.back()];
        indices.back()++;
    }
}

void
Flatten::backprop(Layer*, double, double)
{

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
    for (size_t i = 0; i < this->output_values.size(); i++) {
        this->output_values[i] = 0;
    }
}

void
Flatten::reset_errors()
{
    for (size_t i = 0; i < this->errors.size(); i++) {
        this->errors[i] = 0;
    }
}

// debug
// TODO
void
Flatten::print_layer() const
{
}