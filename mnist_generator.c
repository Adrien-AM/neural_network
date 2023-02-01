#include <stdio.h>
#include <stdlib.h>

#include "mnist_utils.h"
#include "neural_network.h"
#include "utils.h"
#include "data_utils.h"
#include "layer.h"

#define LATENT_SPACE 32

int main(void)
{
    srand(time(NULL));
    unsigned char** images_train_c = read_images("./data/mnist/train-images");
    unsigned char** images_test_c = read_images("./data/mnist/test-images");

    double* images_train[TRAIN_IMAGES];
    convert_images_uc_to_f(images_train, images_train_c, TRAIN_IMAGES, IMAGE_SIZE);
    double* images_test[TEST_IMAGES];
    convert_images_uc_to_f(images_test, images_test_c, TEST_IMAGES, IMAGE_SIZE);

    struct neural_network* nn = create_model(
        ssim, 1, 10, IMAGE_SIZE, 6,
        dense_layer(128, sigmoid),
        // dense_layer(96, sigmoid),
        dense_layer(64, sigmoid),
        // dense_layer(48, sigmoid),
        dense_layer(LATENT_SPACE, sigmoid),
        // dense_layer(48, sigmoid),
        dense_layer(64, sigmoid),
        // dense_layer(96, relu),
        dense_layer(128, sigmoid),
        dense_layer(IMAGE_SIZE, relu)
    );

    randomize_weights(nn, 0, 2);
    fit(nn, 5000, images_train, images_train, 25, 1, 1e-1, 0);
    evaluate(nn, 100, images_test, images_test, ssim, 1);

    // Generation
    /*
    size_t latent_pos = 4;
    reset_values(nn);
    struct layer* latent_layer = nn->layers[latent_pos];
    for (size_t i = 0; i < LATENT_SPACE; i++) {
        latent_layer->neurons[i]->actv_value = latent_layer->activation(rand() / RAND_MAX, 0);
    }

    for (size_t l = latent_pos + 1; l < nn->number_of_layers; l++) {
        struct layer *layer = nn->layers[l];
        layer->forward(layer, nn->layers[l - 1]);
    }

    struct layer* last_layer = nn->layers[nn->number_of_layers - 1];
    double* result = (double*)malloc(sizeof(double) * last_layer->size); // last layer values
    for (size_t n = 0; n < last_layer->size; n++) {
        result[n] = last_layer->neurons[n]->actv_value;
    }

    display_image(result, 10000);
    free(result);
    */

   // Reconstruction :

    size_t img = rand() % TEST_IMAGES;

    display_image(images_test[img], 2000);

    double* reconstructed = predict(nn, images_test[img], 1);

    display_image(reconstructed, 5000);
    free(reconstructed);

    save_nn(nn, "generator.nn");
    free_neural_network(nn);

    for (size_t im = 0; im < TRAIN_IMAGES; im++) {
        free(images_train[im]);
        free(images_train_c[im]);
    }
    free(images_train_c);

    for (size_t im = 0; im < TEST_IMAGES; im++) {
        free(images_test[im]);
        free(images_test_c[im]);
    }
    free(images_test_c);

    return 0;
}
