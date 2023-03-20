#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "Activation.hpp"
#include "Dense.hpp"
#include "Loss.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"

#define SIZE 28 * 28
#define DATA_SIZE 60000
#define TEST_SIZE 1000
#define LATENT_SPACE 256

int
main(void)
{
    std::vector<std::vector<uint8_t>> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images");
    std::vector<std::vector<double>> train_images = uint_to_double_images(mnist_train_images);
    std::random_shuffle(train_images.begin(), train_images.end());
    train_images.resize(DATA_SIZE);
    normalize_pixels(train_images);

    std::vector<std::vector<uint8_t>> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images");
    std::vector<std::vector<double>> test_images = uint_to_double_images(mnist_test_images);
    test_images.resize(TEST_SIZE);
    normalize_pixels(test_images);

    Sigmoid activation;
    Sigmoid output_activation;
    Loss loss = MeanAbsoluteError();
    std::vector<Layer*> layers = {
        new Dense(1024, activation), new Dense(256, activation), new Dense(LATENT_SPACE, activation),
        new Dense(256, activation), new Dense(1024, activation), new Dense(SIZE, output_activation)
    };

    NeuralNetwork nn = NeuralNetwork(SIZE, layers, loss);

    double lr = 1e-4;
    double momentum = 0.1;
    size_t epochs = 50;
    size_t batch_size = 128;

    nn.fit(train_images, train_images, lr, momentum, batch_size, epochs);

    size_t idx_test = 23;
    display_image(test_images[idx_test], 1, 10);
    std::vector<double> restored = nn.predict(test_images[idx_test]);
    display_image(restored, 2, 10);

    printf("Loss : %f\n", loss.evaluate(test_images[idx_test], restored));
    exit(0);

    // Create a normal distribution with mean 0 and standard deviation 1
    std::normal_distribution<double> dist(0.0, 1.0);

    // Create a random number engine
    std::default_random_engine generator;

    // Generate 10 random values from the normal distribution and store them in a vector
    std::vector<double> noise(LATENT_SPACE);
    for (int i = 0; i < LATENT_SPACE; ++i) {
        noise[i] = dist(generator);
    }

    nn.reset_values();
    layers[1]->actv_values = noise;
    for (size_t i = 2; i < layers.size(); i++) {
        layers[i]->forward(layers[i - 1]->actv_values);
    }

    std::vector<double> result = layers.back()->actv_values;
    display_image(result, 10, 10);

    return 0;
}
