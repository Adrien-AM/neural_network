#include "Dropout.hpp"

Dropout::Dropout(double rate)
  : rate(rate)
{
}

void
Dropout::init(vector<size_t> input_shape)
{
    this->shape = input_shape;
}

Tensor<double>
Dropout::forward(const Tensor<double>& input) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    Tensor<double> result(input);
    for (auto& r : result) {
        if(dis(gen) < rate) {
            r = 0;
        }
    }
    return result;
}

vector<size_t> Dropout::output_shape() const
{
    return this->shape;
}

void
Dropout::summarize() const
{
    printf("Dropout with probability %f\n", rate);
}

// debug
void
Dropout::print_layer() const
{
}

Dropout::~Dropout() {}