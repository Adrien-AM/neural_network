#include <arpa/inet.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "data_utils.h"
#include "neural_network.h"
#include "utils.h"
#include "layer.h"

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

    double* images_train[TRAIN_IMAGES];
    convert_images_uc_to_f(images_train, images_train_c, TRAIN_IMAGES, 28*28);
    double* labels_train[TRAIN_IMAGES];
    convert_labels_uc_to_f(labels_train, labels_train_c, TRAIN_IMAGES);
    double* images_test[TEST_IMAGES];
    convert_images_uc_to_f(images_test, images_test_c, TEST_IMAGES, 28*28);
    double* labels_test[TEST_IMAGES];
    convert_labels_uc_to_f(labels_test, labels_test_c, TEST_IMAGES);

    struct norm normalization = get_norm_parameters(images_train, 28*28, TRAIN_IMAGES);
    normalize_inputs(images_train, 28 * 28, TRAIN_IMAGES, normalization);
    normalize_inputs(images_test, 28 * 28, TEST_IMAGES, normalization);

    size_t nb_classes = 10;
    convert_labels_to_softmax(labels_train, nb_classes, TRAIN_IMAGES);
    convert_labels_to_softmax(labels_test, nb_classes, TEST_IMAGES);

    struct neural_network* nn = create_model(
        cross_entropy, 1, 0, 28 * 28, 3,
        dense_layer(96, &sigmoid),
        dense_layer(nb_classes, &sigmoid),
        softmax_layer(nb_classes)
    );
    randomize_weights(nn, 0, 1);

    fit(nn, TRAIN_IMAGES, images_train, labels_train, 20, 1, 5e-4, 0.2);

    evaluate(nn, TEST_IMAGES, images_test, labels_test, cross_entropy, 3);

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
