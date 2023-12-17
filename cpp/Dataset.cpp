#include "Dataset.hpp"

#include <arpa/inet.h>
#include <fstream>
#include <numeric>
#include <stdexcept>

size_t
MnistData::count_mnist_images(string path)
{
    // Open the IDX file
    ifstream file(path, ios::binary);
    if (!file) {
        throw runtime_error("Failed to open file: " + path);
    }

    // Read the magic number
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 0x00000803 && magic_number != 0x00000801) {
        throw runtime_error("Invalid magic number in file: " + path);
    }

    // Read the number of images
    int32_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = ntohl(num_images);

    // Close the file
    file.close();

    return num_images;
}

MnistData::MnistData(string path_img, string path_labels, size_t nb)
  : path_img(path_img)
  , path_labels(path_labels)
  , nb_images(nb)
{
    if (nb_images == 0) {
        nb_images = count_mnist_images(path_img);
    }
}

size_t
MnistData::size() const
{
    return this->nb_images;
}

vector<Tensor<double>>
MnistData::get_item(size_t index, size_t nb) const
{
    vector<Tensor<double>> result(2);
    // Open the IDX file
    ifstream file(path_img, ios::binary);
    ifstream file_labels(path_labels, ios::binary);

    // Seek to the starting position in the file
    file.seekg(16 + (index * H * W), ios::beg);
    file_labels.seekg(8 + (index), ios::beg);

    // Read the data
    Tensor<uint8_t> data(vector<size_t>({ nb, 1, H, W }));
    Tensor<uint8_t> data_labels(vector<size_t>({ nb }));
    for (size_t k = 0; k < nb; k++) {
        Tensor<uint8_t> img = data[k][0];
        uint8_t label;
        file_labels.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));
        for (size_t j = 0; j < H; j++) {
            vector<uint8_t> row(W);
            file.read(reinterpret_cast<char*>(&(*row.begin())), sizeof(uint8_t) * W);
            img[j] = row;
        }
    }

    Tensor<double> images(vector<size_t>({ nb, 1, H, W }));
    for (size_t i = 0; i < nb; i++) {
        for (size_t x = 0; x < H; x++) {
            for (size_t y = 0; y < W; y++) {
                images({ i, 0, x, y }) = (double)(data({ i, 0, x, y })) / 255;
            }
        }
    }

    Tensor<double> labels(vector<size_t>({ data_labels.size(), 10 }));
    for (size_t i = 0; i < data_labels.size(); i++) {
        labels({ i, data_labels[i] }) = 1.0;
    }

    // Close the file
    file.close();
    file_labels.close();

    result[0] = images;
    result[1] = labels;
    return result;
}
