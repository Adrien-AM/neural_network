#include <arpa/inet.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "data_utils.h"
#include "neural_network.h"
#include "utils.h"

#define TRAIN_IMAGES 60000
#define TEST_IMAGES 10000

// void read_image(FILE *f, size_t nb_rows, size_t nb_cols, unsigned int **image)
// {
//     return;
// }

unsigned char**
read_images(char* filename)
{
    FILE* f = fopen(filename, "rb");

    uint32_t magic_number = 0;
    fread(&magic_number, sizeof(uint32_t), 1, f);
    magic_number = ntohl(magic_number);

    uint32_t nb_images = 0;
    fread(&nb_images, sizeof(uint32_t), 1, f);
    nb_images = ntohl(nb_images);

    uint32_t nb_rows = 0;
    fread(&nb_rows, sizeof(uint32_t), 1, f);
    nb_rows = ntohl(nb_rows);
    uint32_t nb_cols = 0;
    fread(&nb_cols, sizeof(uint32_t), 1, f);
    nb_cols = ntohl(nb_cols);

    unsigned char** images = malloc(sizeof(unsigned char*) * nb_images);
    for (size_t im = 0; im < nb_images; im++) {
        images[im] = malloc(sizeof(unsigned char) * nb_cols * nb_rows);
        fread(images[im], sizeof(unsigned char), nb_rows * nb_cols, f);
    }

    fclose(f);

    return images;
}

unsigned char*
read_labels(char* filename)
{
    FILE* f = fopen(filename, "rb");

    uint32_t magic_number = 0;
    fread(&magic_number, sizeof(uint32_t), 1, f);
    magic_number = ntohl(magic_number);

    uint32_t nb_labels = 0;
    fread(&nb_labels, sizeof(uint32_t), 1, f);
    nb_labels = ntohl(nb_labels);

    unsigned char* labels = malloc(sizeof(unsigned char) * nb_labels);
    fread(labels, sizeof(unsigned char), nb_labels, f);
    fclose(f);

    return labels;
}

int
main(void)
{

    unsigned char** images_train_c = read_images("./data/mnist/train-images");
    unsigned char** images_test_c = read_images("./data/mnist/test-images");
    unsigned char* labels_train_c = read_labels("./data/mnist/train-labels");
    unsigned char* labels_test_c = read_labels("./data/mnist/test-labels");

    float* images_train[TRAIN_IMAGES];
    convert_images_uc_to_f(images_train, images_train_c, TRAIN_IMAGES, 28*28);
    float* labels_train[TRAIN_IMAGES];
    convert_labels_uc_to_f(labels_train, labels_train_c, TRAIN_IMAGES);
    float* images_test[TEST_IMAGES];
    convert_images_uc_to_f(images_test, images_test_c, TEST_IMAGES, 28*28);
    float* labels_test[TEST_IMAGES];
    convert_labels_uc_to_f(labels_test, labels_test_c, TEST_IMAGES);

    struct norm normalization = get_norm_parameters(images_train, 28*28, TRAIN_IMAGES);
    normalize_inputs(images_train, 28 * 28, TRAIN_IMAGES, normalization);
    normalize_inputs(images_test, 28 * 28, TEST_IMAGES, normalization);

    size_t layers_size[] = { 16, 8, 1 };
    float (*activations[])(float, int) = { &relu, &relu, &linear };
    struct neural_network* nn = create_model(3, layers_size, 28 * 28, activations);
    randomize_weights(nn, 0, 0.2, 1);

    fit(nn, TRAIN_IMAGES, images_train, labels_train, 10, 1, 1e-6, 0.2);

    free_neural_network(nn);
    for (size_t im = 0; im < TRAIN_IMAGES; im++) {
        free(images_train[im]);
        free(images_train_c[im]);
        free(labels_train[im]);
    }
    free(images_train_c);
    free(labels_train_c);

    for (size_t im = 0; im < TEST_IMAGES; im++) {
        free(images_test[im]);
        free(images_test_c[im]);
        free(labels_test[im]);
    }
    free(images_test_c);
    free(labels_test_c);
    return 0;
}
