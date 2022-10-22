#include "utils.h"

#define MAX_LINE_SIZE 2000
#define MAX_NB_LINES 30000

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

void read_columns(FILE *f, size_t *nb_columns, char **columns)
{
    char line[MAX_LINE_SIZE];

    if (NULL == fgets(line, MAX_LINE_SIZE, f))
    {
        fprintf(stderr, "Cannot read first line of csv file\n");
        exit(0);
    }

    size_t offset = 0;
    size_t c = 0;
    *nb_columns = 1;
    char buffer[MAX_LINE_SIZE];
    while (1)
    {
        buffer[c] = line[c + offset];
        c++;
        if (line[c + offset] == ',' || line[c + offset] == '\n')
        {
            columns[*nb_columns - 1] = malloc(sizeof(char) * (c + 1));
            strncpy(columns[*nb_columns - 1], buffer, c);
            columns[*nb_columns - 1][c] = '\0';
            if (line[c + offset] == '\n')
            {
                break;
            }
            *nb_columns += 1;
            offset += c + 1;
            c = 0;
        }
    }
}

float **read_csv(char *filename, size_t *nb_lines, size_t *nb_columns, char **columns)
{
    FILE *f = fopen(filename, "r");
    if (NULL == f)
    {
        fprintf(stderr, "Cannot open file %s\n", filename);
        perror(NULL);
        exit(0);
    }

    read_columns(f, nb_columns, columns);

    char line[MAX_LINE_SIZE];

    size_t offset;
    size_t c;
    size_t col;

    float **data = malloc(sizeof(float *) * MAX_NB_LINES);
    *nb_lines = 0;
    char buffer[MAX_LINE_SIZE];
    while (fgets(line, MAX_LINE_SIZE, f) != NULL)
    {
        *nb_lines += 1;
        data[*nb_lines - 1] = malloc(sizeof(float) * *nb_columns);
        col = 0;
        offset = 0;
        c = 0;
        while (col < *nb_columns)
        {
            while (line[c + offset] != ',' && line[c + offset] != '\n' && line[c + offset] != EOF)
            {
                buffer[c] = line[c + offset];
                c++;
            }

            buffer[c] = '\0';
            data[*nb_lines - 1][col] = atof(buffer);
            col++;
            offset += c + 1;
            c = 0;
        }
    }

    data = realloc(data, sizeof(float *) * *nb_lines);

    fclose(f);
    return data;
}

void free_csv(size_t nb_lines, size_t nb_columns, float **data, char **columns)
{
    for (size_t i = 0; i < nb_columns; i++)
    {
        free(columns[i]);
    }
    for (size_t i = 0; i < nb_lines; i++)
    {
        free(data[i]);
    }
    free(data);
}