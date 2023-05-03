#include <iostream>

#include "Conv2D.hpp"
#include "Dense.hpp"
#include "Flatten.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"
#include "Tensor.hpp"
#include "Utils.hpp"
#include "Optimizer.hpp"

#define IMAGE_SIZE 28
#define DATA_SIZE 60000
#define TEST_SIZE 10000
#define NB_CLASSES 10

int
main()
{
    Tensor<uint8_t> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images", DATA_SIZE);
    std::cout << "Number of train images: " << mnist_train_images.size() << std::endl;
    Tensor<double> train_images = uint_to_double_images(mnist_train_images);

    vector<uint8_t> mnist_train_labels =
      read_idx_labels_file("../data/mnist/train-labels", DATA_SIZE);
    Tensor<double> train_labels = uint_to_one_hot_labels(mnist_train_labels, NB_CLASSES);

    shuffle_images_labels(train_images, train_labels);
    // shuffle_images_labels(test_images, test_labels);
    train_images.resize(DATA_SIZE);
    train_labels.resize(DATA_SIZE);

    normalize_pixels(train_images);

    ReLU activation;
    Sigmoid sigm;
    Softmax softmax;
    CategoricalCrossEntropy cce;
    double learning_rate = 1e-3;
    // double momentum = 0.1;
    SGD optimizer(learning_rate);
    size_t batch_size = 64;
    size_t epochs = 25;

    NeuralNetwork nn({ 1, IMAGE_SIZE, IMAGE_SIZE },
                     { new Conv2D(16, 3, 0, activation),
                       new Conv2D(8, 3, 0, activation),
                       new Flatten(),
                       new Dense(16, sigm),
                       new Dense(NB_CLASSES, softmax) },
                     cce, optimizer);

    nn.fit(train_images, train_labels, batch_size, epochs);
    exit(0);

    Tensor<uint8_t> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images", TEST_SIZE);
    std::cout << "Number of test images: " << mnist_test_images.size() << std::endl;
    Tensor<double> test_images = uint_to_double_images(mnist_test_images);

    vector<uint8_t> mnist_test_labels =
      read_idx_labels_file("../data/mnist/test-labels", TEST_SIZE);
    Tensor<double> test_labels = uint_to_one_hot_labels(mnist_test_labels, NB_CLASSES);

    test_images.resize(TEST_SIZE);
    test_labels.resize(TEST_SIZE);
    normalize_pixels(test_images);

    Accuracy accuracy;
    printf("Loss on test set : %f\n", nn.evaluate(test_images, test_labels, cce, &accuracy));
    size_t random_image = 3;
    // display_image(test_images[random_image], 2, 10);
    printf("Label : %u\n", mnist_test_labels[random_image]);

    Tensor<double> prediction = nn.predict(test_images[0]);
    print_softmax_output(prediction);
    printf("Loss : %f\n--Accuracy : %f\n",
           cce.evaluate(test_labels[random_image], prediction),
           accuracy.get_result());

    return 0;
}
