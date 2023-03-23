#include "Conv2D.hpp"

Conv2D::Conv2D(unsigned int filters,
               unsigned int kernel_size,
               unsigned int input_width,
               unsigned int padding,
               const Activation& act,
               bool use_bias)
  : filters_size(filters)
  , kernel_size(kernel_size)
  , input_width(input_width)
  , padding(padding)
  , activation(act)
{
    this->weights = std::vector<std::vector<double>>(filters);
    this->updates = std::vector<std::vector<double>>(filters);
    for (unsigned int i = 0; i < filters; i++) {
        this->weights[i] = std::vector<double>(kernel_size * kernel_size);
        this->updates[i] = std::vector<double>(kernel_size * kernel_size);
    }
    if (use_bias)
        this->biases = std::vector<double>(filters);
    else
        this->biases = std::vector<double>();

    this->output_values = std::vector<double>(filters_size * input_width * input_width);
    this->values = std::vector<double>(filters_size * input_width * input_width);
    this->errors = std::vector<double>(filters_size * input_width * input_width);
}

void
Conv2D::init(unsigned int input_size)
{
    (void)input_size;
    bool use_bias = !this->biases.empty();

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 10);
    for (unsigned int n = 0; n < this->weights.size(); n++) {
        for (unsigned int p = 0; p < this->weights[n].size(); p++) {
            this->weights[n][p] = normal(gen);
        }
        if (use_bias) {
            this->biases[n] = normal(gen);
        }
    }
}

void
Conv2D::forward(const std::vector<double>& inputs)
{
    unsigned int width = sqrt(inputs.size());
    this->padded_input = add_padding(inputs, width, this->padding);
    for (unsigned int n = 0; n < this->filters_size; n++) {
        std::vector<double> conv = convolution_product(
          this->padded_input, this->weights[n], sqrt(this->padded_input.size()), 1);
        std::copy(conv.begin(), conv.end(), this->values.begin() + n * inputs.size());
        for (unsigned int i = 0; i < conv.size(); i++) {
            conv[i] += this->biases[n];
        }
    }
    this->output_values = this->activation.compute(this->values);
}

void
Conv2D::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    for (unsigned int f = 0; f < this->filters_size; f++) {
        for (unsigned int kw = 0; kw < this->kernel_size; kw++) {
            for (unsigned int kh = 0; kh < this->kernel_size; kh++) {
                double gradient = 0;
                for (unsigned int x = 0; x < this->input_width; x++) {
                    for (unsigned int y = 0; y < this->input_width; y++) {
                        unsigned int output_index = x * this->input_width + y;
                        unsigned int input_index = (x + kh) * this->input_width + (y + kw);
                        gradient += this->errors[output_index + this->filters_size * f] *
                                    this->padded_input[input_index];
                        this->biases[f] -=
                          learning_rate * errors[output_index + this->filters_size * f];
                    }
                }
                this->errors[f * this->input_width * this->input_width + kh * this->input_width +
                             kw] = gradient;
                double update = (momentum * this->updates[f][kh * kernel_size + kw]) +
                                (1 - momentum) * learning_rate * gradient;
                this->weights[f][kh * kernel_size + kw] -= update;
                this->updates[f][kh * kernel_size + kw] = update;
                // printf("Weight : %f, error %f, update %f, gradient %f\n",
                //        this->updates[f][kh * kernel_size + kw],
                //        this->errors[f * this->input_width * this->input_width +
                //                     kh * this->input_width + kw],
                //        update, gradient);
            }
        }
    }

    for (unsigned int x = 0; x < this->input_width; x++) {
        for (unsigned int y = 0; y < this->input_width; y++) {
            for (unsigned int i = 0; i < this->kernel_size; i++) {
                for (unsigned int j = 0; j < this->kernel_size; j++) {
                    unsigned int input_index = x * this->input_width + y;
                    unsigned int out_x = x - i;
                    unsigned int out_y = y - j;
                    if (x >= i && out_x < this->input_width && y >= j &&
                        out_y < this->input_width) {
                        for (unsigned int f = 0; f < this->filters_size; f++) {
                            input_layer->errors[input_index] +=
                              this->errors[input_layer->size() * f + input_index] *
                              (this->weights[f][j * this->kernel_size + i] +
                               this->updates[f][j * this->kernel_size + i]);
                        }
                    }
                }
            }
        }
    }
}

void
Conv2D::summarize() const
{
    printf("Conv2D | Size : %u. Kernel size : %u.\n", this->size(), this->kernel_size);
}

unsigned int
Conv2D::size() const
{
    return this->output_values.size();
}

void
Conv2D::reset_values()
{
}

void
Conv2D::reset_errors()
{
}

void
Conv2D::print_layer() const
{
    // TODO
}

Conv2D::~Conv2D() {}