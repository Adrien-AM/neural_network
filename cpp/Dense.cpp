#include "Dense.hpp"

Dense::Dense(size_t layer_size, const Activation& act, bool use_bias)
  : activation(act)
{
    this->output_values = Tensor<double>(layer_size);

    if (use_bias)
        this->biases = Tensor<double>(layer_size);
    else
        this->biases = Tensor<double>();
};

void
Dense::print_layer() const
{
    cout << "-- Dense layer --\nWeights : ";
    this->weights.print();
    printf("Biases : ");
    this->biases.print();
    cout << "--------\n" << endl;
}

void
Dense::forward(const Tensor<double>& inputs)
{
    Tensor<double> values = inputs.mm(weights);
    if (!this->biases.empty())
        values += this->biases;

    // Then compute activation
    this->output_values = this->activation.compute(values);
}

size_t
Dense::size() const
{
    return output_values.size();
}
void

Dense::reset_values()
{
    output_values.reset_data();
}

void
Dense::init(vector<size_t> input_shape)
{
    size_t size = this->size();
    this->weights = Tensor<double>(vector<size_t>{ input_shape[0], size });

    random_device rd;
    mt19937 gen(rd()); // Mersenne Twister engine
    
    // normal_distribution<double> initializer(0, 0.3);
    double var = sqrt(6 / (double)(input_shape.size() + this->size()));
    uniform_real_distribution<double> initializer(-var, var);
    for (size_t n = 0; n < input_shape[0]; n++) {
        for (size_t p = 0; p < size; p++) {
            this->weights({n, p}) = initializer(gen);
        }
    }
    if (!this->biases.empty()) {
        for (size_t n = 0; n < size; n++) {
            this->biases(n) = initializer(gen);
        }
    }
}

void
Dense::summarize() const
{
}

Dense*
Dense::clone() const
{
    Dense* copy = new Dense(size(), activation, !biases.empty());
    copy->weights = this->weights;
    copy->biases = this->biases;
    return copy;
}

Dense::~Dense() {}