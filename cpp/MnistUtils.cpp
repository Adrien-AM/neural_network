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

    std::vector<uint8_t> data;
    uint8_t byte;
    while (file.read(reinterpret_cast<char*>(&byte), sizeof(uint8_t))) {
        data.push_back(byte);
    }

    if(data.size() != (unsigned int)num_labels) {
        throw std::runtime_error("Data read is not the expected size.\n");
    }

    // Close the file
    file.close();

    return data;
}

std::vector<std::vector<double>>
uint_to_double_images(const std::vector<std::vector<uint8_t>>& images)
{
    std::vector<std::vector<double>> new_images(images.size());
    for (unsigned int i = 0; i < images.size(); i++) {
        new_images[i] = std::vector<double>(images[i].size());

        // Avoid calling []
        std::vector<double>& new_img = new_images[i];
        const std::vector<uint8_t>& old_img = images[i];
        for (unsigned int pixel = 0; pixel < images[i].size(); pixel++) {
            new_img[pixel] = (double)(old_img[pixel]);
        }
    }

    return new_images;
}

std::vector<std::vector<double>>
uint_to_one_hot_labels(const std::vector<uint8_t>& labels, unsigned int nb_classes)
{
    std::vector<std::vector<double>> result(labels.size());
    for (unsigned int i = 0; i < labels.size(); i++) {
        result[i] = std::vector<double>(nb_classes);
        result[i][labels[i]] = 1;
    }

    return result;
}

void
display_image(const std::vector<double>& image, unsigned int time, int upscale)
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
            // put pixel back in [0, 255] from [0, 1]
            uint8_t pixel = (uint8_t)(image[y * 28 + x] * 255);
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

void
shuffle_images_labels(std::vector<std::vector<double>>& images,
                      std::vector<std::vector<double>>& labels)
{
    // Create a vector of indices for train_images and train_labels
    std::vector<int> indices(images.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices vector using a random number generator
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Shuffle train_images and train_labels in place using the shuffled indices
    for (size_t i = 0; i < indices.size(); ++i) {
        std::swap(images[i], images[indices[i]]);
        std::swap(labels[i], labels[indices[i]]);
    }
}

void
normalize_pixels(std::vector<std::vector<double>>& images)
{
    for(auto& row : images) {
        for(double& pixel : row) {
            pixel /= 255;
        }
    }
}