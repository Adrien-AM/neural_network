#include "MnistUtils.hpp"

using namespace std;

// Function to read IDX file format and return a vector of elements
// (mainly) DONE BY CHATGPT
Tensor<uint8_t>
read_idx_images_file(const string& filename, int max_images)
{
    // Open the IDX file
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Failed to open file: " + filename);
    }

    // Read the magic number
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 0x00000803 && magic_number != 0x00000801) {
        throw runtime_error("Invalid magic number in file: " + filename);
    }

    // Read the dimensions
    int32_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = ntohl(num_images);
    num_images = num_images > max_images ? max_images : num_images;

    int32_t side;
    file.read(reinterpret_cast<char*>(&side), sizeof(side));
    side = ntohl(side);
    if (side != 28) {
        throw runtime_error("Invalid number of rows");
    }
    file.read(reinterpret_cast<char*>(&side), sizeof(side));
    side = ntohl(side);
    if (side != 28) {
        throw runtime_error("Invalid number of columns");
    }

    // Read the data
    Tensor<uint8_t> data(vector<size_t>({ (size_t)num_images, 1, (size_t)side, (size_t)side }));
    for (int32_t i = 0; i < num_images; i++) {
        Tensor<uint8_t> img = data[i][0];
        for (int32_t j = 0; j < side; j++) {
            vector<uint8_t> row(side);
            file.read(reinterpret_cast<char*>(&(*row.begin())), sizeof(uint8_t) * side);
            img[j] = row;
        }
    }

    // Close the file
    file.close();

    return data;
}

vector<uint8_t>
read_idx_labels_file(const string& filename, int max_labels)
{
    // Open the IDX file
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Failed to open file: " + filename);
    }

    // Read the magic number
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 0x00000803 && magic_number != 0x00000801) {
        throw runtime_error("Invalid magic number in file: " + filename);
    }

    // Read the dimensions
    int32_t num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = ntohl(num_labels);
    max_labels = num_labels > max_labels ? max_labels : num_labels;

    vector<uint8_t> data;
    uint8_t byte;
    while (file.read(reinterpret_cast<char*>(&byte), sizeof(uint8_t)) &&
           data.size() < (size_t)max_labels) {
        data.push_back(byte);
    }

    if (data.size() != (size_t)max_labels) {
        throw runtime_error("Data read is not the expected size.\n");
    }

    // Close the file
    file.close();

    return data;
}

Tensor<double>
uint_to_double_images(const Tensor<uint8_t>& images)
{
    Tensor<double> new_images(vector<size_t>(images.shape()));

    vector<size_t> shape = images.shape();
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t c = 0; c < shape[1]; c++) {
            for (size_t x = 0; x < shape[2]; x++) {
                for (size_t y = 0; y < shape[3]; y++) {
                    new_images({i, c, x, y}) = (double)(images({i, c, x, y}));
                }
            }
        }
    }

    return new_images;
}

Tensor<double>
uint_to_one_hot_labels(const vector<uint8_t>& labels, size_t nb_classes)
{
    Tensor<double> result(vector<size_t>({ labels.size(), nb_classes }));
    for (size_t i = 0; i < labels.size(); i++) {
        result({i, labels[i]}) = 1.0;
    }

    return result;
}

void
display_image(const Tensor<double>& image, size_t time, int upscale)
{
    Tensor<double> image_2d = image[0];
    size_t width = image_2d.shape()[1];
    size_t height = image_2d.shape()[0];

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("MNIST Image",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          width * upscale,
                                          height * upscale,
                                          0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // put pixel back in [0, 255] from [0, 1]
            uint8_t pixel = (uint8_t)(image_2d({y, x}) * 255);
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
shuffle_images_labels(Tensor<double>& images, Tensor<double>& labels)
{
    // Create a vector of indices for train_images and train_labels
    vector<int> indices(images.size());
    iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices vector using a random number generator
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    // Shuffle train_images and train_labels in place using the shuffled indices
    for (size_t i = 0; i < indices.size(); ++i) {
        Tensor<double> tmpImg = images[i];
        Tensor<double> tmpLabel = labels[i];
        images[i] = images[indices[i]];
        labels[i] = labels[indices[i]];
        images[indices[i]] = tmpImg;
        labels[indices[i]] = tmpLabel;
    }
}