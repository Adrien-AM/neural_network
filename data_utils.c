#include "data_utils.h"

void
generate_data_inputs(size_t data_size,
                     size_t input_size,
                     float** inputs,
                     int range_start,
                     int range_end)
{
    for (size_t i = 0; i < data_size; i++) {
        inputs[i] = malloc(sizeof(float) * input_size);
        for (size_t j = 0; j < input_size; j++) {
            inputs[i][j] =
              (float)rand() / (float)(RAND_MAX / (range_end - range_start)) + range_start;
        }
    }
}

void
generate_data_outputs(size_t data_size,
                      size_t output_size,
                      float** inputs,
                      float** outputs,
                      float (*func)(float*))
{
    for (size_t i = 0; i < data_size; i++) {
        outputs[i] = malloc(sizeof(float) * output_size);
        *outputs[i] = func(inputs[i]);
    }
}

struct norm
get_norm_parameters(float** inputs, size_t input_size, size_t nb_inputs)
{
    struct norm norm;
    norm.mean = 0;
    norm.stddev = 0;
    for (size_t i = 0; i < nb_inputs; i++) {
        for (size_t j = 0; j < input_size; j++) {
            norm.mean += inputs[i][j];
        }
    }

    norm.mean /= (nb_inputs * input_size);

    for (size_t i = 0; i < nb_inputs; i++) {
        for (size_t j = 0; j < input_size; j++) {
            norm.stddev += (norm.mean - inputs[i][j]) * (norm.mean - inputs[i][j]);
        }
    }
    norm.stddev = sqrt(norm.stddev / (nb_inputs * input_size));

    return norm;
}

void
normalize_inputs(float** inputs, size_t input_size, size_t nb_inputs, struct norm norm)
{
    for (size_t i = 0; i < nb_inputs; i++) {
        for (size_t j = 0; j < input_size; j++) {
            inputs[i][j] = (inputs[i][j] - norm.mean) / norm.stddev;
        }
    }
}

void
read_columns(FILE* f, struct csv* csv)
{
    char line[MAX_LINE_SIZE];

    if (NULL == fgets(line, MAX_LINE_SIZE, f)) {
        fprintf(stderr, "Cannot read first line of csv file\n");
        exit(0);
    }

    size_t offset = 0;
    size_t c = 0;
    csv->nb_columns = 1;
    char buffer[MAX_LINE_SIZE];
    csv->columns = malloc(sizeof(char*) * MAX_NB_COLUMNS);
    while (1) {
        buffer[c] = line[c + offset];
        c++;
        if (line[c + offset] == ',' || line[c + offset] == '\n') {
            csv->columns[csv->nb_columns - 1] = malloc(sizeof(char) * (c + 1));
            strncpy(csv->columns[csv->nb_columns - 1], buffer, c);
            csv->columns[csv->nb_columns - 1][c] = '\0';
            if (line[c + offset] == '\n') {
                break;
            }
            csv->nb_columns += 1;
            offset += c + 1;
            c = 0;
        }
    }
    csv->columns = realloc(csv->columns, sizeof(char*) * csv->nb_columns);
}

struct csv*
read_csv(char* filename)
{
    struct csv* csv = malloc(sizeof(struct csv));
    FILE* f = fopen(filename, "r");
    if (NULL == f) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        perror(NULL);
        exit(0);
    }

    read_columns(f, csv);

    char line[MAX_LINE_SIZE];

    size_t offset;
    size_t c;
    size_t col;

    float** data = malloc(sizeof(float*) * MAX_NB_LINES);
    csv->nb_lines = 0;
    char buffer[MAX_LINE_SIZE];
    while (fgets(line, MAX_LINE_SIZE, f) != NULL) {
        csv->nb_lines += 1;
        data[csv->nb_lines - 1] = malloc(sizeof(float) * csv->nb_columns);
        col = 0;
        offset = 0;
        c = 0;
        while (col < csv->nb_columns) {
            while (line[c + offset] != ',' && line[c + offset] != '\n' && line[c + offset] != EOF) {
                buffer[c] = line[c + offset];
                c++;
            }

            buffer[c] = '\0';
            data[csv->nb_lines - 1][col] = atof(buffer);
            col++;
            offset += c + 1;
            c = 0;
        }
    }

    data = realloc(data, sizeof(float*) * csv->nb_lines);

    csv->data = data;

    fclose(f);
    return csv;
}

void
free_csv(struct csv* csv)
{
    for (size_t i = 0; i < csv->nb_columns; i++) {
        free(csv->columns[i]);
    }
    free(csv->columns);
    for (size_t i = 0; i < csv->nb_lines; i++) {
        free(csv->data[i]);
    }
    free(csv->data);
    free(csv);
}

struct csv*
extract_target_from_data(char* target, struct csv* data)
{
    size_t target_index = 0;
    while (strcmp(data->columns[target_index], target) != 0) {
        target_index++;
    }

    struct csv* target_data = malloc(sizeof(struct csv));
    target_data->nb_columns = 1;
    target_data->nb_lines = data->nb_lines;

    target_data->columns = malloc(sizeof(char*));
    target_data->columns[0] = malloc(sizeof(char) * strlen(target) + 1);
    strcpy(target_data->columns[0], target);

    target_data->data = malloc(sizeof(float*) * data->nb_lines);

    for (size_t i = 0; i < data->nb_lines; i++) {
        target_data->data[i] = malloc(sizeof(float));
        target_data->data[i][0] = data->data[i][target_index];

        data->data[i][target_index] =
          data->data[i][data->nb_columns - 1]; // replace target by last col
        data->data[i] = realloc(data->data[i], sizeof(float) * data->nb_columns - 1);
    }

    // reallocate memory for column name, then rename
    data->columns[target_index] = realloc(
      data->columns[target_index], sizeof(char) * strlen(data->columns[data->nb_columns - 1]) + 1);
    strcpy(data->columns[target_index], data->columns[data->nb_columns - 1]);
    free(data->columns[data->nb_columns - 1]);

    data->columns = realloc(data->columns, sizeof(char*) * data->nb_columns - 1);
    data->nb_columns -= 1;

    return target_data;
}

void
convert_images_uc_to_f(float** images, unsigned char** originals, size_t nb, size_t size)
{
    for (size_t i = 0; i < nb; i++) {
        images[i] = malloc(sizeof(float) * size);
        for (size_t j = 0; j < size; j++) {
            images[i][j] = (float)originals[i][j];
        }
    }
}

void
convert_labels_uc_to_f(float** labels, unsigned char* originals, size_t nb)
{
    for (size_t i = 0; i < nb; i++) {
        labels[i] = malloc(sizeof(float));
        labels[i][0] = (float)originals[i];
    }
}