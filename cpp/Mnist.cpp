#include <iostream>
#include <vector>

#include "MnistUtils.hpp"
#include "Utils.hpp"

#define NB_CLASSES 10

int
main()
{
    std::vector<std::vector<uint8_t>> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images");
    std::cout << "Number of test images: " << mnist_train_images.size() << std::endl;
    std::vector<std::vector<double>> train_images = uint_to_double_images(mnist_train_images);

    std::vector<uint8_t> mnist_train_labels = read_idx_labels_file("../data/mnist/train-labels");
    std::vector<std::vector<double>> train_labels = uint_to_one_hot_labels(mnist_train_labels, NB_CLASSES);

    std::vector<std::vector<uint8_t>> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images");
    std::cout << "Number of test images: " << mnist_test_images.size() << std::endl;
    std::vector<std::vector<double>> test_images = uint_to_double_images(mnist_test_images);

    std::vector<uint8_t> mnist_test_labels = read_idx_labels_file("../data/mnist/test-labels");
    std::vector<std::vector<double>> test_labels = uint_to_one_hot_labels(mnist_test_labels, NB_CLASSES);

    display_image(test_images[0], 1, 10);
    print_vector(test_labels[0]);

    // Read the MNIST test labels data
    // std::vector<uint8_t> mnist_test_labels = read_idx_file("../data/mnist/test-labels");
    // std::cout << "Number of test labels: " << mnist_test_labels.size() << std::endl;

    return 0;
}
