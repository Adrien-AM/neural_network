#include "MnistUtils.hpp"

// Function to read IDX file format and return a vector of elements
// (mainly) DONE BY CHATGPT
Tensor<uint8_t>
read_idx_images_file(const std::string& filename, int max_images)
{
    // Open the IDX file
    ifstream file(filename, std::ios::binary);
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
    num_images = num_images > max_images ? max_images : num_images;

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
    Tensor<uint8_t> data(vector<size_t>({ (size_t)num_images, (size_t)side, (size_t)side }));
    for (int32_t i = 0; i < num_images; i++) {
        file.read(reinterpret_cast<char*>(data.at(i).data()), sizeof(uint8_t) * side * side);
    }

    // Close the file
    file.close();

    return data;
}

vector<uint8_t>
read_idx_labels_file(const std::string& filename, int max_labels)
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
    max_labels = num_labels > max_labels ? max_labels : num_labels;

    vector<uint8_t> data;
    uint8_t byte;
    while (file.read(reinterpret_cast<char*>(&byte), sizeof(uint8_t))) {
        data.push_back(byte);
    }

    if (data.size() != (size_t)num_labels) {
        throw std::runtime_error("Data read is not the expected size.\n");
    }

    // Close the file
    file.close();

    return data;
}

Tensor<double>
uint_to_double_images(const Tensor<uint8_t>& images)
{
    Tensor<double> new_images(vector<size_t>(images.shape()));

    for (size_t i = 0; i < images.size(); i++) {
        Tensor<uint8_t> old_img = images.at(i);
        Tensor<double> new_img = new_images.at(i);
        for (size_t x = 0; x < old_img.shape()[0]; x++) {
            Tensor<uint8_t> old_row = old_img.at(x);
            Tensor<double> new_row = new_img.at(x);
            for (size_t y = 0; y < old_img.shape()[1]; y++) {
                new_row[y] = (double)(old_row[y]);
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
        result.at(i)[labels[i]] = 1;
    }

    return result;
}

void
display_image(const Tensor<double>& image, size_t time, int upscale)
{
    size_t width = image.shape()[0];
    size_t height = image.shape()[1];

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
            uint8_t pixel = (uint8_t)(image.at(y)[x] * 255);
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
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices vector using a random number generator
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Shuffle train_images and train_labels in place using the shuffled indices
    for (size_t i = 0; i < indices.size(); ++i) {
        Tensor<double> tmpImg = images.at(i);
        Tensor<double> tmpLabel = labels.at(i);
        images.at(i) = images.at(indices[i]);
        labels.at(i) = labels.at(indices[i]);
        images.at(indices[i]) = tmpImg;
        labels.at(indices[i]) = tmpLabel;
    }
}

void
normalize_pixels(Tensor<double>& images)
{
    size_t nb_images = images.size();
    for (size_t i = 0; i < nb_images; i++) {
        Tensor<double> im = images.at(i);
        for (size_t x = 0; x < images.at(i).size(); x++) {
            Tensor<double> row = im.at(x);
            for (size_t y = 0; y < images.at(i).at(x).size(); y++)
                row[y] /= 255;
        }
    }
}