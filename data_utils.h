#ifndef __DATA_UTILS_H__
#define __DATA_UTILS_H__

#define MAX_NB_COLUMNS 100
#define MAX_LINE_SIZE 2000
#define MAX_NB_LINES 30000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct csv {
    float **data;
    size_t nb_lines;
    char **columns;
    size_t nb_columns;
};

// Data functions
void generate_data_inputs(size_t data_size, size_t input_size,
                          float **inputs, int range_start, int range_end);
void generate_data_outputs(size_t data_size, size_t output_size, float ** inputs,
                           float **outputs, float (*func)(float *));

// CSV utils

// Undefined behaviour when files doesnt have ending newline :)
struct csv * read_csv(char *filename);
void free_csv(struct csv *);

// Does NOT check if `target` exists in columns. May cause undefined behaviour
struct csv *extract_target_from_data(char *target, struct csv *data);

#endif // __DATA_UTILS_H__