#ifndef __SDLMNIST_HPP__
#define __SDLMNIST_HPP__

#include <SDL2/SDL.h>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <stdexcept>

#include <numeric>
#include <random>

#include "Tensor.hpp"

using namespace std;

Tensor<uint8_t>
read_idx_images_file(const std::string& filename, int max_images);

vector<uint8_t>
read_idx_labels_file(const std::string& filename, int max_labels);

Tensor<double>
uint_to_double_images(const Tensor<uint8_t>& images);

Tensor<double>
uint_to_one_hot_labels(const vector<uint8_t>& labels, size_t nb_classes);

void
display_image(const Tensor<double>& image, size_t time, int upscale);

void
shuffle_images_labels(Tensor<double>& images,
                      Tensor<double>& labels);

void
normalize_pixels(Tensor<double>& images);

#endif // __SDLMNIST_HPP__