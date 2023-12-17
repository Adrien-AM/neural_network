#include <iostream>
#include <time.h>

#include "Conv2D.hpp"
#include "Dataset.hpp"
#include "Dense.hpp"
#include "Dropout.hpp"
#include "Flatten.hpp"
#include "Metric.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"
#include "Optimizer.hpp"
#include "Utils.hpp"

#define IMAGE_SHAPE                                                                                \
    {                                                                                              \
        1, 28, 28                                                                                  \
    }
#define DATA_SIZE 100
#define TEST_SIZE 50
#define NB_CLASSES 10

using namespace std;

int
main()
{

    MnistData data_train(
      "../data/mnist/train-images", "../data/mnist/train-labels", DATA_SIZE);
    // Tensor<uint8_t> mnist_train_images =
    //   read_idx_images_file("../data/mnist/train-images", DATA_SIZE);
    // std::cout << "Number of train images: " << DATA_SIZE << std::endl;
    // Tensor<double> train_images = uint_to_double_images(mnist_train_images);

    // vector<uint8_t> mnist_train_labels =
    //   read_idx_labels_file("../data/mnist/train-labels", DATA_SIZE);
    // Tensor<double> train_labels = uint_to_one_hot_labels(mnist_train_labels, NB_CLASSES);

    Tensor<uint8_t> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images", TEST_SIZE);
    std::cout << "Number of test images: " << TEST_SIZE << std::endl;
    Tensor<double> test_images = uint_to_double_images(mnist_test_images);

    vector<uint8_t> mnist_test_labels =
      read_idx_labels_file("../data/mnist/test-labels", TEST_SIZE);
    Tensor<double> test_labels = uint_to_one_hot_labels(mnist_test_labels, NB_CLASSES);

    // shuffle_images_labels(train_images, train_labels);

    // train_images /= 255.0;
    // test_images /= 255.0;

    ReLU activation;
    Softmax softmax;
    CategoricalCrossEntropy cce;
    size_t epochs = 10;
    size_t batch_size = 1;

    SGD optimizer(1e-3);
    NeuralNetwork nn(IMAGE_SHAPE,
                     { new Flatten(),
                       new Dense(4, activation),
                       new Dense(NB_CLASSES, softmax) },
                     cce,
                     optimizer);

    nn.fit(data_train, batch_size, epochs);

    Accuracy accuracy;
    printf("Loss on test set : %f\n", nn.evaluate(test_images, test_labels, cce, &accuracy));
    printf("Accuracy on test set : %f\n", accuracy.get_result());

    return 0;
}
