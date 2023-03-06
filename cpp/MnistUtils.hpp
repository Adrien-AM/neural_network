#ifndef __SDLMNIST_HPP__
#define __SDLMNIST_HPP__

#include <SDL2/SDL.h>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <random>

std::vector<std::vector<uint8_t>>
read_idx_images_file(const std::string& filename, int max_images);

std::vector<uint8_t>
read_idx_labels_file(const std::string& filename, int max_labels);

std::vector<std::vector<double>>
uint_to_double_images(const std::vector<std::vector<uint8_t>>& images);

std::vector<std::vector<double>>
uint_to_one_hot_labels(const std::vector<uint8_t>& labels, unsigned int nb_classes);

void
display_image(const std::vector<double>& image, unsigned int time, int upscale);

void
shuffle_images_labels(std::vector<std::vector<double>>& images,
                      std::vector<std::vector<double>>& labels);

void
normalize_pixels(std::vector<std::vector<double>>& images);

#endif // __SDLMNIST_HPP__