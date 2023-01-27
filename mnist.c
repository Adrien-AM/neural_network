#include "mnist_utils.h"
#include "data_utils.h"
#include "utils.h"
#include "neural_network.h"
#include "layer.h"

int
main(int argc, char **argv)
{
    srand(time(NULL));
    unsigned char** images_train_c = read_images("./data/mnist/train-images");
    unsigned char** images_test_c = read_images("./data/mnist/test-images");
    unsigned char* labels_train_c = read_labels("./data/mnist/train-labels");
    unsigned char* labels_test_c = read_labels("./data/mnist/test-labels");

    double* images_train[TRAIN_IMAGES];
    convert_images_uc_to_f(images_train, images_train_c, TRAIN_IMAGES, IMAGE_SIZE);
    double* labels_train[TRAIN_IMAGES];
    convert_labels_uc_to_f(labels_train, labels_train_c, TRAIN_IMAGES);
    double* images_test[TEST_IMAGES];
    convert_images_uc_to_f(images_test, images_test_c, TEST_IMAGES, IMAGE_SIZE);
    double* labels_test[TEST_IMAGES];
    convert_labels_uc_to_f(labels_test, labels_test_c, TEST_IMAGES);

    // struct norm normalization = get_norm_parameters(images_train, IMAGE_SIZE, TRAIN_IMAGES);
    // normalize_inputs(images_train, 28 * 28, TRAIN_IMAGES, normalization);
    // normalize_inputs(images_test, 28 * 28, TEST_IMAGES, normalization);

    size_t nb_classes = 10;
    convert_labels_to_softmax(labels_train, nb_classes, TRAIN_IMAGES);
    convert_labels_to_softmax(labels_test, nb_classes, TEST_IMAGES);

    struct neural_network* nn = create_model(
        cross_entropy, 1, 1, IMAGE_SIZE, 3,
        dense_layer(32, &sigmoid),
        dense_layer(nb_classes, &linear),
        softmax_layer(nb_classes)
    );
    randomize_weights(nn, 0, 1);

    size_t img = rand() % TRAIN_IMAGES;
    fit(nn, TRAIN_IMAGES, images_train, labels_train, 5, 1, 1e-4, 0.5);
    print_softmax(predict(nn, images_train[img], 1), 10, 0.3);
    display_image(images_train[img], 10000);

    // evaluate(nn, TEST_IMAGES, images_test, labels_test, cross_entropy, 3);

    if (argc > 1)
        save_nn(nn, argv[1]);

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
