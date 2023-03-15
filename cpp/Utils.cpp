#include "Utils.hpp"

void
print_vector(std::vector<double> vec)
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
print_softmax_output(std::vector<double> vec)
{
    printf("Class : ");
    double max;
    do {
        max = 0;
        int argmax = 0;
        for (unsigned int x = 0; x < vec.size(); x++) {
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

std::vector<double>
add_padding(std::vector<double>& image, unsigned int width, unsigned int pad_size)
{
    // Padded image dimensions
    int height = image.size() / width;
    int padded_width = width + 2 * pad_size;
    int padded_height = height + 2 * pad_size;

    // Create a new vector to hold the padded image
    std::vector<double> padded_image(padded_width * padded_height);

    // Copy the values from the original image to the padded image
    for (int y = 0; y < padded_height; y++) {
        for (int x = 0; x < padded_width; x++) {
            if (x < pad_size || x >= (width + pad_size) || y < pad_size ||
                y >= (height + pad_size)) {
                // This pixel is in the padding area, set it to 0
                padded_image[y * padded_width + x] = 0.0;
            } else {
                // This pixel is in the original image area, copy the value from the original image
                int original_x = x - pad_size;
                int original_y = y - pad_size;
                padded_image[y * padded_width + x] =
                  image[original_y * width + original_x];
            }
        }
    }
}

std::vector<double>
convolution_product(const std::vector<double>& input,
                    const std::vector<double>& filter,
                    unsigned int width,
                    unsigned int stride)
{
    unsigned int size = input.size();
    unsigned int kernel_size = sqrt(filter.size());
    std::vector<double> output(size / stride);

    for (unsigned int i = 0; i < size; i += stride) {
        for (unsigned int kh = 0; kh < kernel_size; kh++) {
            for (unsigned int kw = 0; kw < kernel_size; kw++) {
                output[i] += input[i + kw + (width * kh)] * filter[kw + (kh * kernel_size)];
            }
        }
    }
}