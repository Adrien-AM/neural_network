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
    this->weights = vector<size_t>({ filters_size, kernel_size, kernel_size });
    this->updates = vector<size_t>({ filters_size, kernel_size, kernel_size });
    if (use_bias)
        this->biases = Tensor<double>(filters_size);
    else
        this->biases = Tensor<double>();
}

void
Conv2D::init(vector<size_t> input_shape)
{
    bool use_bias = !this->biases.empty();
    this->output_values = vector<size_t>({ filters_size, input_shape[0], input_shape[1] });
    this->values = vector<size_t>({ filters_size, input_shape[0], input_shape[1] });
    this->errors = vector<size_t>({ filters_size, input_shape[0], input_shape[1] });

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 10);
    for (size_t n = 0; n < this->weights.size(); n++) {
        for (size_t i = 0; i < kernel_size; i++)
            for (size_t j = 0; j < kernel_size; j++)
                this->weights.at(n).at(i)[j] = normal(gen);
        if (use_bias) {
            this->biases[n] = normal(gen);
        }
    }
}

void
Conv2D::forward(const Tensor<double>& inputs)
{
    this->padded_input = add_padding(inputs, this->padding);
    for (size_t n = 0; n < this->filters_size; n++) {
        Tensor<double> value_n = this->values.at(n);
        Tensor<double> output_n = this->output_values.at(n);
        value_n = convolution_product(this->padded_input, this->weights.at(n), 1);
        // memcpy(this->values.data(), conv.data(), n * inputs.size() * sizeof(double));
        if (!this->biases.empty()) {
            for (size_t i = 0; i < kernel_size; i++) {
                Tensor<double> row_i = value_n.at(i);
                for (size_t j = 0; j < kernel_size; j++)
                    row_i[j] += this->biases[n];
            }
        }
        for (size_t i = 0; i < this->values.shape()[1]; i++)
            // Activation only works for vector for now
            output_n.at(i) = this->activation.compute(value_n.at(i));
    }
}

void
Conv2D::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    vector<size_t> input_shape = input_layer->output_values.shape();
    // NEED TO USE DELTA_ERRORS INSTEAD OF ERRORS :))))
    
    for (size_t f = 0; f < this->filters_size; f++) {
        Tensor<double> error_f = this->errors.at(f);
        Tensor<double> weight_f = this->weights.at(f);
        Tensor<double> update_f = this->updates.at(f);
        for (size_t kh = 0; kh < this->kernel_size; kh++) {
            Tensor<double> error_kh = error_f.at(kh);
            Tensor<double> weight_kh = weight_f.at(kh);
            Tensor<double> update_kh = update_f.at(kh);
            for (size_t kw = 0; kw < this->kernel_size; kw++) {
                double gradient = 0;
                for (size_t x = 0; x < input_shape[0]; x++) {
                    Tensor<double> error_f_x = error_f.at(x);
                    for (size_t y = 0; y < input_shape[1]; y++) {
                        double error = error_f_x[y];
                        gradient += error * this->padded_input.at(x + kh)[y + kw];
                        this->biases[f] -= learning_rate * error;
                    }
                }
                error_kh[kw] = gradient;
                double update =
                  (momentum * update_kh[kw]) + (1 - momentum) * learning_rate * gradient;
                weight_kh[kw] -= update;
                update_kh[kw] = update;
            }
        }
    }

    for (size_t f = 0; f < this->filters_size; f++) {
        Tensor<double> error_f = this->errors.at(f);
        Tensor<double> weight_f = this->weights.at(f);
        Tensor<double> update_f = this->updates.at(f);
        for (size_t x = 0; x < input_shape[0]; x++) {
            Tensor<double> input_error_x = input_layer->errors.at(x);
            Tensor<double> error_x = error_f.at(x);
            for (size_t y = 0; y < input_shape[1]; y++) {
                for (size_t j = 0; j < this->kernel_size; j++) {
                    Tensor<double> weight_j = weight_f.at(j);
                    Tensor<double> update_j = update_f.at(j);
                    for (size_t i = 0; i < this->kernel_size; i++) {
                        // size_t input_index = x * input_width + y;
                        size_t out_x = x - i;
                        size_t out_y = y - j;
                        if (x >= i && out_x < input_shape[0] && y >= j && out_y < input_shape[1]) {
                            input_error_x[y] += error_x[y] * (weight_j[i] + update_j[i]);
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