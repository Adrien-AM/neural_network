#include <iostream>
#include <vector>

#include "Conv2D.hpp"
#include "Dense.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"

#define IMAGE_SIZE 28 * 28
#define DATA_SIZE 60000
#define TEST_SIZE 10000
#define NB_CLASSES 10

int
main()
{
    vector<vector<uint8_t>> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images", DATA_SIZE);
    std::cout << "Number of train images: " << mnist_train_images.size() << std::endl;
    vector<vector<double>> train_images = uint_to_double_images(mnist_train_images);

    vector<uint8_t> mnist_train_labels =
      read_idx_labels_file("../data/mnist/train-labels", DATA_SIZE);
    vector<vector<double>> train_labels =
      uint_to_one_hot_labels(mnist_train_labels, NB_CLASSES);

    vector<vector<uint8_t>> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images", TEST_SIZE);
    std::cout << "Number of test images: " << mnist_test_images.size() << std::endl;
    vector<vector<double>> test_images = uint_to_double_images(mnist_test_images);

    vector<uint8_t> mnist_test_labels =
      read_idx_labels_file("../data/mnist/test-labels", TEST_SIZE);
    vector<vector<double>> test_labels =
      uint_to_one_hot_labels(mnist_test_labels, NB_CLASSES);

    shuffle_images_labels(train_images, train_labels);
    shuffle_images_labels(test_images, test_labels);
    train_images.resize(DATA_SIZE);
    train_labels.resize(DATA_SIZE);
    test_images.resize(TEST_SIZE);
    test_labels.resize(TEST_SIZE);

    normalize_pixels(train_images);
    normalize_pixels(test_images);

    Sigmoid activation;
    Softmax softmax;
    Loss cce = CategoricalCrossEntropy();
    double learning_rate = 5e-4;
    double momentum = 0.5;
    size_t batch_size = 1500;
    unsigned int epochs = 25;

    NeuralNetwork nn(IMAGE_SIZE,
                     { new Conv2D(32, 3, 1, activation),
                       new Dense(NB_CLASSES, softmax, true) },
                     cce);

    nn.fit(train_images, train_labels, learning_rate, momentum, batch_size, epochs);

    Metric* accuracy = new Accuracy();
    printf("Loss on test set : %f\n", nn.evaluate(test_images, test_labels, cce, accuracy));
    unsigned int random_image = 13;
    // display_image(test_images[random_image], 2, 10);
    printf("Label : %u\n", mnist_test_labels[random_image]);

    vector<double> prediction = nn.predict(test_images[0]);
    print_softmax_output(prediction);
    printf("Loss : %f\n--Accuracy : %f\n",
           cce.evaluate(test_labels[random_image], prediction),
           accuracy->get_result());

    return 0;
}
