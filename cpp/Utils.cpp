#include "Utils.hpp"

void
print_vector(Tensor<double> vec)
{
    std::cout << "[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void
print_softmax_output(Tensor<double> vec)
{
    printf("Class : ");
    double max;
    do {
        max = 0;
        int argmax = 0;
        for (size_t x = 0; x < vec.size(); x++) {
            if (vec[x] > max) {
                max = vec[x];
                argmax = x;
            }
        }
        printf("%d with %.3f certainty ; ", argmax, max);
        vec[argmax] = 0;
    } while (max > MIN_SOFTMAX_OUTPUT);
    printf("Others too low.\n");
}

Tensor<double>
add_padding_2d(const Tensor<double>& image, size_t pad_size)
{
    if(pad_size == 0)
        return Tensor<double>(image);
    // Padded image dimensions
    size_t height = image.shape()[0];
    size_t width = image.shape()[1];
    size_t padded_width = width + 2 * pad_size;
    size_t padded_height = height + 2 * pad_size;

    // Create a new vector to hold the padded image
    Tensor<double> padded_image = vector<size_t>({ padded_height, padded_height });

    // Copy the values from the original image to the padded image
    for (size_t y = 0; y < padded_height; y++) {
        for (size_t x = 0; x < padded_width; x++) {
            if (x < pad_size || x >= width + pad_size || y < pad_size || y >= height + pad_size) {
                // This pixel is in the padding area, set it to 0
                padded_image.at(y)[x] = 0.0;
            } else {
                // This pixel is in the original image area, copy the value from the original
                // image
                int original_x = x - pad_size;
                int original_y = y - pad_size;
                padded_image.at(y)[x] = image.at(original_y)[original_x];
            }
        }
    }

    return padded_image;
}

Tensor<double>
convolution_2d(const Tensor<double>& input, const Tensor<double>& kernel, size_t stride)
{
    (void)stride; // TODO : Implement stride
    vector<size_t> input_shape = input.shape();
    vector<size_t> kernel_shape = kernel.shape();
    int half = kernel_shape[0] / 2;

    if(input_shape.size() != 2 || kernel_shape.size() != 2) {
        throw length_error("Convolution 2D can only be done on 2d image and kernel.");
    }

    vector<size_t> output_shape = { input_shape[0] - kernel_shape[0] + 1,
                                    input_shape[1] - kernel_shape[1] + 1 };

    Tensor<double> output(output_shape);

    for (size_t i = half; i < input_shape[0] - half; i++) {
        Tensor<double> output_row = output.at(i - half);
        for (size_t j = half; j < input_shape[1] - half; j++) {
            double value = 0;
            for (int kw = -half; kw <= half; kw++) {
                Tensor<double> input_row = input.at(i + kw);
                Tensor<double> kernel_row = kernel.at(kw + half);
                for (int kh = -half; kh <= half; kh++) {
                    value += input_row[j + kh] * kernel_row[kh + half];
                }
            }
            output_row[j - half] = value;
        }
    }
    return output;
}

Tensor<double>
convolution_product(const Tensor<double>& input, const Tensor<double>& filter, size_t stride)
{
    vector<size_t> input_shape = input.shape();
    vector<size_t> kernel_shape = filter.shape();

    if (kernel_shape[0] != input_shape[0]) {
        fprintf(stderr,
                "Channel mismatch in convolution operation : %zu & %zu.\n",
                kernel_shape[0],
                input_shape[0]);
        exit(0);
    }

    vector<size_t> output_shape(input_shape.begin() + 1, input_shape.end());
    for (size_t i = 0; i < output_shape.size(); i++) {
        output_shape[i] = input_shape[i + 1] - kernel_shape[i + 1] + 1;
    }

    Tensor<double> output(output_shape);

    for (size_t c = 0; c < input_shape[0]; c++) {
        Tensor<double> input_channel = input.at(c);
        Tensor<double> filter_channel = filter.at(c);
        for (size_t i = 0; i <= input_shape[1] - kernel_shape[1]; i += stride) {
            Tensor<double> output_i = output.at(i);
            for (size_t j = 0; j <= input_shape[2] - kernel_shape[2]; j += stride) {
                for (size_t kh = 0; kh < kernel_shape[1]; kh++) {
                    Tensor<double> input_row = input_channel.at(i + kh);
                    Tensor<double> filter_row = filter_channel.at(kh);
                    for (size_t kw = 0; kw < kernel_shape[2]; kw++) {
                        output_i[j] += input_row[j + kw] * filter_row[kw];
                    }
                }
            }
        }
    }

    return output;
}