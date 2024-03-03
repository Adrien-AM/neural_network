#include "Dense.hpp"

Dense::Dense(size_t layer_size, bool use_bias) : size(layer_size)
{
    this->shape = { layer_size };
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

Tensor<double>
Dense::forward(const Tensor<double>& inputs) const
{
    Tensor<double> result = inputs.mm(weights);
    if (!this->biases.empty())
        result += this->biases;
    return result;
}

void
Dense::init(vector<size_t> input_shape)
{
    this->weights = Tensor<double>(vector<size_t>{ input_shape[0], size });

    random_device rd;
    mt19937 gen(rd()); // Mersenne Twister engine
    
    // normal_distribution<double> initializer(0, 1e-3);
    double var = sqrt(6 / (double)(input_shape.size() + size));
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

vector<size_t> Dense::output_shape() const
{
    return this->shape;
}

void
Dense::summarize() const
{
}


Dense*
Dense::clone() const
{
    Dense* copy = new Dense(size, !biases.empty());
    copy->weights = this->weights;
    copy->biases = this->biases;
    return copy;
}

Dense::~Dense() {}