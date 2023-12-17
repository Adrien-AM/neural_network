#include "Conv2D.hpp"

Conv2D::Conv2D(size_t filters,
               size_t kernel_size,
               size_t padding,
               const Activation& act,
               bool use_bias)
  : filters_size(filters)
  , kernel_size(kernel_size)
  , padding(padding)
  , activation(act)
{
    if (use_bias)
        this->biases = Tensor<double>(filters_size);
    else
        this->biases = Tensor<double>();
}

void
Conv2D::init(vector<size_t> input_shape)
{
    this->channels = input_shape[0];
    size_t output_width = input_shape[1] - kernel_size + 1 + 2 * padding;
    size_t output_height = input_shape[2] - kernel_size + 1 + 2 * padding;

    this->weights = vector<size_t>({ filters_size, channels, kernel_size, kernel_size });
    this->output_values = vector<size_t>({ filters_size, output_height, output_width });

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 0.5);
    for (size_t n = 0; n < this->weights.size(); n++) {
        for (size_t c = 0; c < channels; c++)
            for (size_t i = 0; i < kernel_size; i++)
                for (size_t j = 0; j < kernel_size; j++)
                    weights({n, c, i, j}) = normal(gen);
    }
}

void
Conv2D::forward(const Tensor<double>& inputs)
{
    // padded_input = vector<size_t>(
    //   { channels, inputs.shape()[1] + 2 * padding, inputs.shape()[2] + 2 * padding });

    // for (size_t c = 0; c < channels; c++) {
    //     padded_input[c] = add_padding_2d(inputs[c], padding);
    // }

    for (size_t f = 0; f < this->filters_size; f++) {
        double bias = biases.empty() ? 0 : biases(f);
        Tensor<double> conv = convolution_2d(inputs, this->weights[f], bias, 1);
        output_values[f] = this->activation.compute(conv);
    }
}

void
Conv2D::summarize() const
{
    printf("Conv2D | Size : %zu. Kernel size : %zu.\n", this->size(), this->kernel_size);
}

size_t
Conv2D::size() const
{
    return this->output_values.size();
}

void
Conv2D::reset_values()
{
    output_values.reset_data();
}

void
Conv2D::print_layer() const
{
    // TODO
}

Conv2D*
Conv2D::clone() const
{
    Conv2D* copy = new Conv2D(filters_size, kernel_size, padding, activation, !biases.empty());
    copy->channels = channels;
    copy->weights = weights;
    return copy;
}

Conv2D::~Conv2D() {}