#include <iostream>
#include <time.h>
#include <vector>

#include "Dense.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include "Metric.hpp"

#define IMAGE_SIZE 28 * 28
#define DATA_SIZE 30000
#define TEST_SIZE 2000
#define NB_CLASSES 10

int
main()
{
    std::vector<std::vector<uint8_t>> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images", DATA_SIZE);
    std::cout << "Number of train images: " << DATA_SIZE << std::endl;
    std::vector<std::vector<double>> train_images = uint_to_double_images(mnist_train_images);

    std::vector<uint8_t> mnist_train_labels =
      read_idx_labels_file("../data/mnist/train-labels", DATA_SIZE);
    std::vector<std::vector<double>> train_labels =
      uint_to_one_hot_labels(mnist_train_labels, NB_CLASSES);

    std::vector<std::vector<uint8_t>> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images", TEST_SIZE);
    std::cout << "Number of test images: " << TEST_SIZE << std::endl;
    std::vector<std::vector<double>> test_images = uint_to_double_images(mnist_test_images);

    std::vector<uint8_t> mnist_test_labels =
      read_idx_labels_file("../data/mnist/test-labels", TEST_SIZE);
    std::vector<std::vector<double>> test_labels =
      uint_to_one_hot_labels(mnist_test_labels, NB_CLASSES);

    shuffle_images_labels(train_images, train_labels);
    train_images.resize(DATA_SIZE);
    train_labels.resize(DATA_SIZE);
    test_images.resize(TEST_SIZE);
    test_labels.resize(TEST_SIZE);
    // shuffle_images_labels(test_images, test_labels);

    normalize_pixels(train_images);
    normalize_pixels(test_images);

    Sigmoid activation;
    Softmax softmax;
    Loss cce = CategoricalCrossEntropy();
    double learning_rate = 5e-3;
    double momentum = 0.9;
    unsigned int epochs = 10;
    size_t batch_size = 128;

    NeuralNetwork nn(
      IMAGE_SIZE, { new Dense(32, activation), new Dense(NB_CLASSES, softmax) }, cce);

    nn.fit(train_images, train_labels, learning_rate, momentum, batch_size, epochs);

    Metric* accuracy = new Accuracy();
    printf("Loss on test set : %f\n", nn.evaluate(test_images, test_labels, cce, accuracy));
    // unsigned int random_image = 9000;
    printf("Accuracy on test set : %f\n", accuracy->get_result());
    // display_image(test_images[random_image], 2, 10);
    // printf("Label : %u\n", mnist_test_labels[random_image]);

    // std::vector<double> prediction = nn.predict(test_images[0]);
    // print_softmax_output(prediction);
    // printf("Loss : %f\n--\n", cce.evaluate(test_labels[random_image], prediction));

    return 0;
}
