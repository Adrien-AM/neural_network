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
add_padding(const Tensor<double>& image, size_t pad_size)
{
    // Padded image dimensions
    size_t height = image.shape()[0];
    size_t width = image.shape()[1];
    size_t padded_width = width + 2 * pad_size;
    size_t padded_height = height + 2 * pad_size;

    // Create a new vector to hold the padded image
    Tensor<double> padded_image = vector<size_t>({ padded_width, padded_height });

    // Copy the values from the original image to the padded image
    for (size_t y = 0; y < padded_height; y++) {
        for (size_t x = 0; x < padded_width; x++) {
            if (x < pad_size || x >= width || y < pad_size || y >= height) {
                // This pixel is in the padding area, set it to 0
                padded_image.at(y)[x] = 0.0;
            } else {
                // This pixel is in the original image area, copy the value from the original image
                int original_x = x - pad_size;
                int original_y = y - pad_size;
                padded_image.at(y)[x] = image.at(original_y)[original_x];
            }
        }
    }

    return padded_image;
}

Tensor<double>
convolution_product(const Tensor<double>& input,
                    const Tensor<double>& filter,
                    size_t stride)
{
    vector<size_t> input_shape = input.shape();
    size_t kernel_size = filter.shape()[0];
    vector<size_t> output_shape(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); i++) {
        output_shape[i] = input_shape[i] - kernel_size + 1;
    }
    Tensor<double> output(output_shape);

    for (size_t i = 0; i <= input.shape()[0] - kernel_size; i += stride) {
        Tensor<double> output_i = output.at(i);
        for (size_t j = 0; j <= input.shape()[1] - kernel_size; j += stride) {
            for (size_t kh = 0; kh < kernel_size; kh++) {
                for (size_t kw = 0; kw < kernel_size; kw++) {
                    output_i[j] += input.at(i + kh)[j + kw] * filter.at(kh)[kw];
                }
            }
        }
    }

    return output;
}