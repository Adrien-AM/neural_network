#include "Layer.hpp"

Layer::Layer(const Activation& act, unsigned int layer_size, bool use_bias)
  : activation(act)
{
    this->values = std::vector<double>(layer_size);
    this->actv_values = std::vector<double>(layer_size);
    this->errors = std::vector<double>(layer_size);
    this->delta_errors = std::vector<double>(layer_size);

    this->weights = std::vector<std::vector<double>>(layer_size);
    if(use_bias)
        this->biases = std::vector<double>(layer_size);
    else
        this->biases = std::vector<double>(0);
};

void
Layer::print_layer() const
{
    std::cout << "--Layer--\n";
    for (auto& row : this->weights) {
        print_vector(row);
    }
    std::cout << "--------\n" << std::endl;
}

unsigned int
Layer::size()
{
    return this->values.size();
}

Layer::~Layer() {}