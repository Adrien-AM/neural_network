#include <iostream>
#include <vector>

#include "Dense.hpp"
#include "Conv2D.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"

#define IMAGE_SIZE 28 * 28
#define DATA_SIZE 100
#define TEST_SIZE 20
#define NB_CLASSES 10

int
main()
{
    std::vector<std::vector<uint8_t>> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images", DATA_SIZE);
    std::cout << "Number of train images: " << mnist_train_images.size() << std::endl;
    std::vector<std::vector<double>> train_images = uint_to_double_images(mnist_train_images);

    std::vector<uint8_t> mnist_train_labels = read_idx_labels_file("../data/mnist/train-labels", DATA_SIZE);
    std::vector<std::vector<double>> train_labels =
      uint_to_one_hot_labels(mnist_train_labels, NB_CLASSES);

    std::vector<std::vector<uint8_t>> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images", TEST_SIZE);
    std::cout << "Number of test images: " << mnist_test_images.size() << std::endl;
    std::vector<std::vector<double>> test_images = uint_to_double_images(mnist_test_images);

    std::vector<uint8_t> mnist_test_labels = read_idx_labels_file("../data/mnist/test-labels", TEST_SIZE);
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
    display_image(add_padding(train_images[0], 32, 2), 5, 10);
    exit(0);


    Sigmoid activation;
    Softmax softmax;
    Loss cce = CategoricalCrossEntropy();
    double learning_rate = 1e-4;
    double momentum = 0.5;
    unsigned int epochs = 10;

    NeuralNetwork nn(IMAGE_SIZE, { new Conv2D(32, 2, 28, activation), new Dense(NB_CLASSES, softmax, true) }, cce);

    nn.fit(train_images, train_labels, learning_rate, momentum, epochs);

    printf("Loss on test set : %f\n", nn.evaluate(test_images, test_labels, cce));
    unsigned int random_image = 62;
    // display_image(test_images[random_image], 2, 10);
    printf("Label : %u\n", mnist_test_labels[random_image]);
    print_vector(test_labels[random_image]);

    std::vector<double> prediction = nn.predict(test_images[0]);
    print_vector(prediction);
    printf("Loss : %f\n--\n", cce.evaluate(test_labels[random_image], prediction));

    return 0;
}
