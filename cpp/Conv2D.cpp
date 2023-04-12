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
    this->updates = vector<size_t>({ filters_size, channels, kernel_size, kernel_size });

    this->values = vector<size_t>({ filters_size, output_height, output_width });
    this->output_values = vector<size_t>({ filters_size, output_height, output_width });
    this->errors = vector<size_t>({ filters_size, output_height, output_width });

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<double> normal(0, 0.5);
    for (size_t n = 0; n < this->weights.size(); n++) {
        Tensor<double> weights_n = this->weights.at(n);
        for (size_t c = 0; c < channels; c++)
            for (size_t i = 0; i < kernel_size; i++)
                for (size_t j = 0; j < kernel_size; j++)
                    weights_n.at(c).at(i)[j] = normal(gen);
    }
}

void
Conv2D::forward(const Tensor<double>& inputs)
{
    padded_input = vector<size_t>(
      { channels, inputs.shape()[1] + 2 * padding, inputs.shape()[2] + 2 * padding });

    for (size_t c = 0; c < channels; c++) {
        padded_input.at(c) = add_padding_2d(inputs.at(c), padding);
    }

    for (size_t f = 0; f < this->filters_size; f++) {
        Tensor<double> value_f = this->values.at(f);
        Tensor<double> output_f = this->output_values.at(f);
        value_f = convolution_product(this->padded_input, this->weights.at(f), 1);
        if (!this->biases.empty()) {
            for (size_t i = 0; i < value_f.size(); i++) {
                Tensor<double> row_i = value_f.at(i);
                for (size_t j = 0; j < row_i.size(); j++)
                    row_i[j] += this->biases[f];
            }
        }
        for (size_t i = 0; i < value_f.size(); i++)
            // Activation only works for vector for now
            output_f.at(i) = this->activation.compute(value_f.at(i));
    }
}

/* Careful : only works when partial derivatives don't depend on other values ! (simple activation)*/
void
Conv2D::backprop(Layer* input_layer, double learning_rate, double momentum)
{
    vector<size_t> input_shape = input_layer->output_values.shape();

    // Compute delta errors
    for (size_t f = 0; f < filters_size; f++) {
        Tensor<double> error_f = this->errors.at(f);
        Tensor<double> output_f = this->output_values.at(f);
        for (size_t row = 0; row < error_f.size(); row++) {
            Tensor<double> error_row = error_f.at(row);
            Tensor<double> jacobian = this->activation.derivative(output_f.at(row)); // dav/dv
            for (size_t j = 0; j < jacobian.size(); j++) {
                error_row[j] *= jacobian.at(j)[j];
            }
        }
    }

    // Compute updates
    for (size_t f = 0; f < this->filters_size; f++) {
        Tensor<double> error_f = this->errors.at(f);
        Tensor<double> weight_f = this->weights.at(f);
        Tensor<double> update_f = this->updates.at(f);
        for (size_t c = 0; c < channels; c++) {
            Tensor<double> weights_c = weight_f.at(c);
            Tensor<double> updates_c = update_f.at(c);
            Tensor<double> padded_input_c = padded_input.at(c);
            for (size_t kh = 0; kh < this->kernel_size; kh++) {
                Tensor<double> weight_kh = weights_c.at(kh);
                Tensor<double> update_kh = updates_c.at(kh);
                for (size_t kw = 0; kw < this->kernel_size; kw++) {
                    double gradient = 0;
                    for (size_t x = 0; x < padded_input.shape()[1] - kernel_size; x++) {
                        Tensor<double> error_f_x = error_f.at(x);
                        Tensor<double> padded_input_kh = padded_input_c.at(x + kh);
                        for (size_t y = 0; y < padded_input.shape()[2] - kernel_size; y++) {
                            double error = error_f_x[y];
                            gradient += error * padded_input_kh[y + kw];
                            if(kw == 0 && kh == 0) // Do it once per kernel
                                biases[f] -= learning_rate * error;
                        }
                    }
                    double update =
                      (momentum * update_kh[kw]) + (1 - momentum) * learning_rate * gradient;
                    weight_kh[kw] -= update;
                    update_kh[kw] = update;
                }
            }
        }
    }

    // Backpropagate to inputs
    // Convolution between errors and flipped kernel
    for (size_t f = 0; f < filters_size; f++) {
        Tensor<double> error_f = add_padding_2d(errors.at(f), kernel_size - 1);

        // Flip kernel
        Tensor<double> flipped_kernel = weights.at(f);
        double* data = flipped_kernel.data();
        size_t total_size = flipped_kernel.total_size();
        for (size_t m = 0; m < total_size / 2; m++) {
            double tmp = data[m];
            data[m] = data[total_size - m - 1];
            data[total_size - m - 1] = tmp;
        }
        // Apply convolution
        for (size_t e = 0; e < channels; e++) {
            input_layer->errors.at(e) = convolution_2d(error_f, flipped_kernel.at(e), 1);
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
    output_values.reset_data();
    values.reset_data();
}

void
Conv2D::reset_errors()
{
    errors.reset_data();
}

void
Conv2D::print_layer() const
{
    // TODO
}

Conv2D::~Conv2D() {}