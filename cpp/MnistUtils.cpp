#include "MnistUtils.hpp"

// Function to read IDX file format and return a vector of elements
// (mainly) DONE BY CHATGPT
std::vector<std::vector<uint8_t>>
read_idx_images_file(const std::string& filename)
{
    // Open the IDX file
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the magic number
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 0x00000803 && magic_number != 0x00000801) {
        throw std::runtime_error("Invalid magic number in file: " + filename);
    }

    // Read the dimensions
    int32_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = ntohl(num_images);

    int32_t side;
    file.read(reinterpret_cast<char*>(&side), sizeof(side));
    side = ntohl(side);
    if (side != 28) {
        throw std::runtime_error("Invalid number of rows");
    }
    file.read(reinterpret_cast<char*>(&side), sizeof(side));
    side = ntohl(side);
    if (side != 28) {
        throw std::runtime_error("Invalid number of columns");
    }

    // Read the data
    std::vector<std::vector<uint8_t>> data(num_images);
    for (int32_t i = 0; i < num_images; i++) {
        data[i] = std::vector<uint8_t>(side * side);
        file.read(reinterpret_cast<char*>(data[i].data()), sizeof(uint8_t) * side * side);
    }

    // Close the file
    file.close();

    return data;
}

std::vector<uint8_t>
read_idx_labels_file(const std::string& filename)
{
    // Open the IDX file
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the magic number
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 0x00000803 && magic_number != 0x00000801) {
        throw std::runtime_error("Invalid magic number in file: " + filename);
    }

    // Read the dimensions
    int32_t num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = ntohl(num_labels);

    std::vector<uint8_t> data(num_labels);
    for (int32_t i = 0; i < num_labels; i++) {
        file.read(reinterpret_cast<char*>(data.data()), sizeof(uint8_t));
    }

    // Close the file
    file.close();

    return data;
}

std::vector<std::vector<double>>
uint_to_double_images(std::vector<std::vector<uint8_t>> images)
{
    std::vector<std::vector<double>> new_images(images.size());
    for (unsigned int i = 0; i < images.size(); i++) {
        new_images[i] = std::vector<double>(images[i].size());
        for (unsigned int pixel = 0; pixel < images[i].size(); pixel++) {
            new_images[i][pixel] = (double)(images[i][pixel]);
        }
    }

    return new_images;
}

std::vector<std::vector<double>>
uint_to_one_hot_labels(std::vector<uint8_t> labels, unsigned int nb_classes)
{
    std::vector<std::vector<double>> result(labels.size());
    for (unsigned int i = 0; i < labels.size(); i++) {
        result[i] = std::vector<double>(nb_classes);
        result[i][labels[i] + 1] = 1;
    }

    return result;
}

void
display_image(std::vector<double> image, unsigned int time, int upscale)
{

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("MNIST Image",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          28 * upscale,
                                          28 * upscale,
                                          0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            uint8_t pixel = (uint8_t)image[y * 28 + x];
            SDL_SetRenderDrawColor(renderer, pixel, pixel, pixel, 255);
            for (int i = 0; i < upscale; i++) {
                for (int j = 0; j < upscale; j++) {
                    SDL_RenderDrawPoint(renderer, x * upscale + i, y * upscale + j);
                }
            }
        }
    }

    SDL_RenderPresent(renderer);
    SDL_DestroyRenderer(renderer);
    SDL_Delay(time * 1000);
    SDL_DestroyWindow(window);
    SDL_Quit();
}