#include "utils.h"

#define MAX_LINE_SIZE 1024

float relu(float x, int derivative)
{
    if (derivative)
        return x >= 0 ? 1 : 0;
    return x >= 0 ? x : 0;
}

float sigmoid(float x, int derivative)
{
    if (derivative)
    {
        float out = sigmoid(x, 0);
        return out * (1 - out);
    }

    return 1 / (1 + exp(-x));
}

float linear(float x, int derivative)
{
    if (derivative)
        return 1;
    return x;
}

// CREDITS https://www.tutorialspoint.com/generate-random-numbers-following-a-normal-distribution-in-c-cplusplus

float rand_gen()
{
    return ((float)rand() + 1) / ((float)RAND_MAX + 1);
}

float rand_normal(float mu, float sigma)
{
    float v1 = rand_gen();
    float v2 = rand_gen();

    float r1 = cos(2 * 3.14 * v2) * sqrt(-2 * log(v1));
    return r1 * sigma + mu;
}

float mean_squared_error(float *y_true, float *y_pred, size_t size)
{
    if (0 == size)
        return 0;
    float total = 0;
    for (size_t i = 0; i < size; i++)
    {
        float error = y_true[i] - y_pred[i];
        total += error * error;
    }

    return total / size;
}

void generate_data_inputs(size_t data_size, size_t input_size, float **inputs, int range_start, int range_end)
{
    for (size_t i = 0; i < data_size; i++)
    {
        inputs[i] = malloc(sizeof(float) * input_size);
        for (size_t j = 0; j < input_size; j++)
        {
            inputs[i][j] = (float)rand() / (float)(RAND_MAX / (range_end - range_start)) + range_start;
        }
    }
}

void generate_data_outputs(size_t data_size, size_t output_size, float **inputs,
                           float **outputs, float (*func)(float *))
{
    for (size_t i = 0; i < data_size; i++)
    {
        outputs[i] = malloc(sizeof(float) * output_size);
        *outputs[i] = func(inputs[i]);
    }
}

float **read_csv(char *filename, size_t *nb_lines, size_t *nb_columns, char **columns)
{
    (void)nb_lines;
    FILE *f = fopen(filename, "r");
    if (NULL == f)
    {
        fprintf(stderr, "Cannot open file %s\n", filename);
        perror(NULL);
        exit(0);
    }

    char line[MAX_LINE_SIZE];

    if (NULL == fgets(line, MAX_LINE_SIZE, f))
    {
        fprintf(stderr, "Cannot read first line of csv file\n");
        exit(0);
    }

    int offset = 0;
    int c = 0;
    *nb_columns = 1;
    while (line[c + offset] != '\n')
    {
        if (line[c + offset] == ',')
        {
            offset += c;
            c = 0;
            *nb_columns += 1;
            continue;
        }
        columns[*nb_columns - 1][c] = line[c + offset];
    }

    fclose(f);
    return NULL;
}