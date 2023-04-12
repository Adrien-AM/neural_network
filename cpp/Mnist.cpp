#include <iostream>
#include <time.h>

#include "Dense.hpp"
#include "Flatten.hpp"
#include "Metric.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"

#define IMAGE_SHAPE                                                                                \
    {                                                                                              \
        1, 28, 28                                                                                  \
    }
#define DATA_SIZE 60000
#define TEST_SIZE 10000
#define NB_CLASSES 10

using namespace std;

int
main()
{
    Tensor<uint8_t> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images", DATA_SIZE);
    std::cout << "Number of train images: " << DATA_SIZE << std::endl;
    Tensor<double> train_images = uint_to_double_images(mnist_train_images);

    vector<uint8_t> mnist_train_labels =
      read_idx_labels_file("../data/mnist/train-labels", DATA_SIZE);
    Tensor<double> train_labels = uint_to_one_hot_labels(mnist_train_labels, NB_CLASSES);

    Tensor<uint8_t> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images", TEST_SIZE);
    std::cout << "Number of test images: " << TEST_SIZE << std::endl;
    Tensor<double> test_images = uint_to_double_images(mnist_test_images);

    vector<uint8_t> mnist_test_labels =
      read_idx_labels_file("../data/mnist/test-labels", TEST_SIZE);
    Tensor<double> test_labels = uint_to_one_hot_labels(mnist_test_labels, NB_CLASSES);

    shuffle_images_labels(train_images, train_labels);
    train_images.resize(DATA_SIZE);
    train_labels.resize(DATA_SIZE);
    test_images.resize(TEST_SIZE);
    test_labels.resize(TEST_SIZE);

    normalize_pixels(train_images);
    normalize_pixels(test_images);

    ReLU activation;
    Softmax softmax;
    Loss cce = CategoricalCrossEntropy();
    double learning_rate = 1e-3;
    double momentum = 0.1;
    size_t epochs = 100;
    size_t batch_size = 256;

    NeuralNetwork nn(IMAGE_SHAPE,
                     { new Flatten(),
                       new Dense(128, activation),
                       new Dense(64, activation),
                       new Dense(NB_CLASSES, softmax) },
                     cce);

    nn.fit(train_images, train_labels, learning_rate, momentum, batch_size, epochs);

    Accuracy accuracy;
    printf("Loss on test set : %f\n", nn.evaluate(test_images, test_labels, cce, &accuracy));
    printf("Accuracy on test set : %f\n", accuracy.get_result());

    return 0;
}
