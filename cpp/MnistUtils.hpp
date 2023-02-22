#ifndef __SDLMNIST_HPP__
#define __SDLMNIST_HPP__

#include <vector>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <stdexcept>
#include <SDL2/SDL.h>

std::vector<std::vector<uint8_t>>
read_idx_images_file(const std::string& filename);

std::vector<uint8_t>
read_idx_labels_file(const std::string& filename);

std::vector<std::vector<double>>
uint_to_double_images(std::vector<std::vector<uint8_t>> images);

std::vector<std::vector<double>>
uint_to_one_hot_labels(std::vector<uint8_t> labels, unsigned int nb_classes);

void
display_image(std::vector<double> image, unsigned int time, int upscale);

#endif // __SDLMNIST_HPP__