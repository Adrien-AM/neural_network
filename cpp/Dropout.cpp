#include "Dropout.hpp"

Dropout::Dropout(double rate)
  : rate(rate)
{
}

void
Dropout::init(vector<size_t> input_shape)
{
    this->output_values = input_shape;
    this->errors = input_shape;
}

void
Dropout::forward(const Tensor<double>& input)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    input.copy_data(output_values);
    double* outputs = output_values.data();
    actives = vector<bool>(input.total_size());
    for (size_t i = 0; i < actives.size(); i++) {
        if(dis(gen) < rate) {
            actives[i] = false;
            outputs[i] = 0;
        } else {
            actives[i] = true;
        }
    }
}
void
Dropout::summarize() const
{
    printf("Dropout with probability %f\n", rate);
}

size_t
Dropout::size() const
{
    return this->output_values.size();
}

void
Dropout::reset_values()
{
    this->output_values.reset_data();
}

void
Dropout::reset_errors()
{
    this->errors.reset_data();
}

// debug
void
Dropout::print_layer() const
{
}

Dropout::~Dropout() {}