#ifndef __MNIST_UTILS_H__
#define __MNIST_UTILS_H__

#include <arpa/inet.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <SDL2/SDL.h>

#define TRAIN_IMAGES 60000
#define TEST_IMAGES 10000
#define IMAGE_SIZE 28*28

#define SCALE 10

unsigned char**
read_images(char* filename);

unsigned char*
read_labels(char* filename);

void
plot_mnist_image(double data[28 * 28], SDL_Window* window);

void
display_image(double* image, size_t time);

#endif // __MNIST_UTILS_H__