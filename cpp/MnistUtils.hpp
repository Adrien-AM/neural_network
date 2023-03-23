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

using namespace std;

vector<vector<uint8_t>>
read_idx_images_file(const std::string& filename, int max_images);

vector<uint8_t>
read_idx_labels_file(const std::string& filename, int max_labels);

vector<vector<double>>
uint_to_double_images(const vector<vector<uint8_t>>& images);

vector<vector<double>>
uint_to_one_hot_labels(const vector<uint8_t>& labels, unsigned int nb_classes);

void
display_image(const vector<double>& image, unsigned int time, int upscale);

void
shuffle_images_labels(vector<vector<double>>& images,
                      vector<vector<double>>& labels);

void
normalize_pixels(vector<vector<double>>& images);

#endif // __SDLMNIST_HPP__