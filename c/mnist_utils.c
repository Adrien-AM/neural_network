#include "mnist_utils.h"

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

// Assume that 'data' is a 28x28 array of uint8_t representing the MNIST image
// and 'window' is a pointer to an SDL_Window

// thanks chatgpt

void plot_mnist_image(double data[28 * 28], SDL_Window *window) {
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            uint8_t pixel = (uint8_t)data[y * 28 + x];
            SDL_SetRenderDrawColor(renderer, pixel, pixel, pixel, 255);
            for (int i = 0; i < SCALE; i++) {
                for (int j = 0; j < SCALE; j++) {
                    SDL_RenderDrawPoint(renderer, x*SCALE + i, y*SCALE + j);
                }
            }
        }
    }

    SDL_RenderPresent(renderer);
    SDL_DestroyRenderer(renderer);
}

void display_image(double *image, size_t time)
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("MNIST Image", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 28 * SCALE, 28 * SCALE, 0);
    plot_mnist_image(image, window);
    SDL_Delay(time); // Wait for 3 seconds
    SDL_DestroyWindow(window);
    SDL_Quit();
}
