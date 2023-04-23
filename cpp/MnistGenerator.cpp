#include <algorithm>
#include <iostream>
#include <random>

#include "Activation.hpp"
#include "Dense.hpp"
#include "Flatten.hpp"
#include "Loss.hpp"
#include "MnistUtils.hpp"
#include "NeuralNetwork.hpp"
#include "Reshape.hpp"

#define SHAPE                                                                                      \
    {                                                                                              \
        1, 28, 28                                                                                  \
    }
#define DATA_SIZE 60000
#define TEST_SIZE 10000
#define LATENT_SPACE 32

using namespace std;

int
main(void)
{
    Tensor<uint8_t> mnist_train_images =
      read_idx_images_file("../data/mnist/train-images", DATA_SIZE);
    Tensor<double> train_images_2d = uint_to_double_images(mnist_train_images);
    normalize_pixels(train_images_2d);
    Tensor<double> train_images = vector<size_t>({ DATA_SIZE, 28 * 28 });
    for (size_t i = 0; i < DATA_SIZE; i++) {
        train_images.at(i) = train_images_2d.at(i).flatten();
    }

    Sigmoid activation;
    Sigmoid output_activation;
    MeanSquaredError loss;
    vector<Layer*> layers = { new Dense(64, activation),
                              new Dense(LATENT_SPACE, activation),
                              new Dense(64, activation),
                              new Dense(28 * 28, output_activation) };

    // TODO : SHUFFLE ON TENSOR
    NeuralNetwork nn = NeuralNetwork({ 28 * 28 }, layers, loss);

    double lr = 1e-2;
    double momentum = 0.9;
    size_t epochs = 30;
    size_t batch_size = 128;

    nn.fit(train_images, train_images, lr, momentum, batch_size, epochs);

    Tensor<uint8_t> mnist_test_images =
      read_idx_images_file("../data/mnist/test-images", TEST_SIZE);
    Tensor<double> test_images_2d = uint_to_double_images(mnist_test_images);
    normalize_pixels(test_images_2d);
    Tensor<double> test_images = vector<size_t>({ TEST_SIZE, 28 * 28 });
    for (size_t i = 0; i < TEST_SIZE; i++) {
        test_images.at(i) = test_images_2d.at(i).flatten();
    }

    size_t idx_test = 0;
    display_image(test_images_2d.at(idx_test), 1, 10);
    Tensor<double> restored = nn.predict(test_images.at(idx_test));
    Tensor<double> image = vector<size_t>({ 1, 28, 28 });
    restored.copy_data(image);
    display_image(image, 2, 10);

    printf("Loss : %f\n", loss.evaluate(test_images.at(idx_test), restored));
    exit(0);
    /*
        // Create a normal distribution with mean 0 and standard deviation 1
        std::normal_distribution<double> dist(0.0, 1.0);

        // Create a random number engine
        std::default_random_engine generator;

        // Generate 10 random values from the normal distribution and store them in a vector
        Tensor<double> noise(LATENT_SPACE);
        for (int i = 0; i < LATENT_SPACE; ++i) {
            noise[i] = dist(generator);
        }

        nn.reset_values();
        layers[1]->actv_values = noise;
        for (size_t i = 2; i < layers.size(); i++) {
            layers[i]->forward(layers[i - 1]->actv_values);
        }

        Tensor<double> result = layers.back()->actv_values;
        display_image(result, 10, 10);

        return 0;*/
}
