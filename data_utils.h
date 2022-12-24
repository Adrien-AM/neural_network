#ifndef __DATA_UTILS_H__
#define __DATA_UTILS_H__

#define MAX_NB_COLUMNS 100
#define MAX_LINE_SIZE 2000
#define MAX_NB_LINES 30000

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct csv
{
    double** data;
    size_t nb_lines;
    char** columns;
    size_t nb_columns;
};

struct norm
{
    double mean;
    double stddev;
};

// Data functions
void
generate_data_inputs(size_t data_size,
                     size_t input_size,
                     double** inputs,
                     int range_start,
                     int range_end);
void
generate_data_outputs(size_t data_size,
                      size_t output_size,
                      double** inputs,
                      double** outputs,
                      double (*func)(double*));

struct norm
get_norm_parameters(double** inputs, size_t input_size, size_t nb_inputs);

void
normalize_inputs(double** inputs, size_t input_size, size_t nb_inputs, struct norm normalization);

// CSV utils

// Undefined behaviour when files doesnt have ending newline :)
struct csv*
read_csv(char* filename);
void
free_csv(struct csv*);

// Does NOT check if `target` exists in columns. May cause undefined behaviour
struct csv*
extract_target_from_data(char* target, struct csv* data);

// Image functions for MNIST

// Allocates and copy data from an unsigned char image to a double image (or label)
void
convert_images_uc_to_f(double** images, unsigned char** originals, size_t nb, size_t size);

void
convert_labels_uc_to_f(double** labels, unsigned char* originals, size_t nb);

#endif // __DATA_UTILS_H__