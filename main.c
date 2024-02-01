#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jansson.h>

void execute(const char* path) {
    char scriptContent[200] = "#!/bin/bash\n python ";
    strcat(scriptContent, path);
    FILE *scriptFile = fopen("temp_script.sh", "w");
    fprintf(scriptFile, "%s", scriptContent);
    fclose(scriptFile);
    system("source temp_script.sh");
    remove("temp_script.sh");
}

void** read_weights(const char* path) {
    FILE *file = fopen(path, "r");
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *json_content = (char *)malloc(file_size + 1);
    fread(json_content, 1, file_size, file);
    fclose(file);
    json_content[file_size] = '\0';

    // Parse the JSON
    json_t *root;
    json_error_t error;
    root = json_loads(json_content, 0, &error);
    free(json_content);

    void** ans [10];


    // Retrieve values by key
    json_t *value = json_object_get(root, "conv1.weight");
    float conv1_weight[6][3][5][5];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 5; k++) {
                for (int r = 0; r < 5; r++) {
                    json_t *element = json_array_get(json_array_get(json_array_get(json_array_get(value, i), j), k), r);
                    conv1_weight[i][j][k][r] = json_real_value(element);
                }
            }
        }
    }
    ans[0]  = conv1_weight;

    value = json_object_get(root, "conv1.bias");
    float conv1_bias[6];
    for (int i = 0; i < 6; i++) {
        json_t *element = json_array_get(value, i);
        conv1_bias[i] = json_real_value(element);
    }
    ans[1] = conv1_bias;

    value = json_object_get(root, "conv2.weight");
    float conv2_weight[16][6][5][5];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 5; k++) {
                for (int r = 0; r < 5; r++) {
                    json_t *element = json_array_get(json_array_get(json_array_get(json_array_get(value, i), j), k), r);
                    conv2_weight[i][j][k][r] = json_real_value(element);
                }
            }
        }
    }
    ans[2] = conv2_weight;

    value = json_object_get(root, "conv2.bias");
    float conv2_bias[16];
    for (int i = 0; i < 16; i++) {
        json_t *element = json_array_get(value, i);
        conv2_bias[i] = json_real_value(element);
    }
    ans[3] = conv2_bias;

    value = json_object_get(root, "fc1.weight");
    float fc1_weight[120][400];
    for (int i = 0; i < 120; i++) {
        for (int j = 0; j < 400; j++) {
            json_t *element = json_array_get(json_array_get(value, i), j);
            fc1_weight[i][j] = json_real_value(element);
        }
    }
    ans[4] = fc1_weight;

    value = json_object_get(root, "fc1.bias");
    float fc1_bias[120];
    for (int i = 0; i < 120; i++) {
        json_t *element = json_array_get(value, i);
        fc1_bias[i] = json_real_value(element);
    }
    ans[5] = fc1_bias;

    value = json_object_get(root, "fc2.weight");
    float fc2_weight[84][120];
    for (int i = 0; i < 84; i++) {
        for (int j = 0; j < 120; j++) {
            json_t *element = json_array_get(json_array_get(value, i), j);
            fc2_weight[i][j] = json_real_value(element);
        }
    }
    ans[6] = fc2_weight;

    value = json_object_get(root, "fc2.bias");
    float fc2_bias[84];
    for (int i = 0; i < 84; i++) {
        json_t *element = json_array_get(value, i);
        fc2_bias[i] = json_real_value(element);
    }
    ans[7] = fc2_bias;

    value = json_object_get(root, "fc3.weight");
    float fc3_weight[10][84];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 84; j++) {
            json_t *element = json_array_get(json_array_get(value, i), j);
            fc3_weight[i][j] = json_real_value(element);
        }
    }
    ans[8] = fc3_weight;

    value = json_object_get(root, "fc3.bias");
    float fc3_bias[10];
    for (int i = 0; i < 10; i++) {
        json_t *element = json_array_get(value, i);
        fc3_bias[i] = json_real_value(element);
    }
    ans[9] = fc3_bias;
    return ans;
}

void** average_weights() {
    void*** aggregated_weights[10];
    for (int i = 0; i < 10; i++) {
        char* path[100];
        path = "model_weights/weights"
        sprintf(path, "%s%d", path, i);
        aggregated_weights[i] = read_weights(path);
    }
    json_t *root = json_object();

    float conv1_weight[6][3][5][5];
    json_t* first_layer = json_array();
    for (int i = 0; i < 6; i++) {
        json_t* second_layer = json_array();
        for (int j = 0; j < 3; j++) {
            json_t third_layer = json_array();
            for (int k = 0; k < 5; k++) {
                json_t forth_layer = json_array()
                for (int r = 0; r < 5; r++) {
                    conv1_weight[i][j][k][r] = 0;
                    for (int p = 0; p < 10; p++) {
                        conv1_weight[i][j][k][r] += aggregated_weights[p][0][i][j][k][r];
                    }
                    conv1_weight[i][j][k][r] /= 10;
                    json_array_append(forth_layer, json_real(conv1_weight[i][j][k][r]));
                }
                json_array_append(third_layer, forth_layer);
            }
            json_array_append(second_layer, third_layer);
        }
        json_array_append(first_layer, second_layer);
    }
    result[0] = conv1_weight;
    json_object_set(root, "conv1.weight", first_layer);

    float conv1_bias[6];
    json_t* conv1_bias_first_layer = json_array();
    for (int i = 0; i < 6; i++) {
        conv1_bias[i] = 0;
        for (int p = 0; p < 10; p++) {
            conv1_bias[i] += aggregated_weights[p][1][i];
        }
        conv1_bias[i] /= 10
        json_array_append(conv1_bias_first_layer, json_real(conv1_bias[i]));
    }

    result[1] = conv1_bias;
    json_object_set(root, "conv1.bias", conv1_bias_first_layer);

    float conv2_weight[16][6][5][5];
    json_t* conv2_weight_first_layer = json_array();
    for (int i = 0; i < 16; i++) {
        json_t* conv2_weight_second_layer = json_array();
        for (int j = 0; j < 6; j++) {
            json_t* conv2_weight_third_layer = json_array();
            for (int k = 0; k < 5; k++) {
                json_t* conv2_weight_forth_layer = json_array();
                for (int r = 0; r < 5; r++) {
                    conv2_weight[i][j][k][r] = 0;
                    for (int p = 0; p < 10; p++) {
                        conv2_weight[i][j][k][r] += aggregated_weights[p][2][i][j][k][r];
                    }
                    conv2_weight[i][j][k][r] /= 10;
                    json_array_append(conv2_weight_forth_layer, json_real(conv2_weight[i][j][k][r]));
                }
                json_array_append(conv2_weight_third_layer, conv2_weight_forth_layer);

            }
            json_array_append(conv2_weight_second_layer, conv2_weight_third_layer);

        }
        json_array_append(conv2_weight_first_layer, conv2_weight_second_layer);

    }
    result[2] = conv2_weight;
    json_object_set(root, "conv2.weight", conv2_weight_first_layer);


    float conv2_bias[16];
    json_t* conv2_bias_first_layer = json_array();
    for (int i = 0; i < 16; i++) {
        conv2_bias[i] = 0;
        for (int p = 0; p < 10; p++) {
            conv2_bias[i] += aggregated_weights[p][3][i];
        }
        conv2_bias[i] /= 10;
        json_array_append(conv2_bias_first_layer, json_real(conv2_bias[i]));
    }
    result[3] = conv2_bias;
    json_object_set(root, "conv2.bias", conv2_bias_first_layer);


    float fc1_weight[120][400];
    json_t* fc1_weight_first_layer = json_array();
    for (int i = 0; i < 120; i++) {
        json_t* fc1_weight_second_layer = json_array();
        for (int j = 0; j < 400; j++) {
            fc1_weight[i][j] = 0;
            for (int p = 0; p < 10; p++) {
                fc1_weight[i][j] += aggregated_weights[p][4][i][j];
            }
            fc1_weight[i][j] /= 10;
            json_array_append(fc1_weight_second_layer, json_real(fc1_weight[i][j]));
        }
        json_array_append(fc1_weight_first_layer, fc1_weight_second_layer)
    }
    result[4] = fc1_weight;
    json_object_set(root, "fc1.weight", fc1_weight_first_layer);


    float fc1_bias[120];
    json_t* fc1_bias_first_layer = json_array();

    for (int i = 0; i < 120; i++) {
        fc1_bias[i] = 0;
        for (int p = 0; p < 10; p++) {
            fc1_bias[i] += aggregated_weights[p][5][i];
        }
        fc1_bias /= 10;
        json_array_append(fc1_bias_first_layer, json_real(fc1_bias[i]));
    }
    result[5] = fc1_bias;
    json_object_set(root, "fc1.bias", fc1_bias_first_layer);

    float fc2_weight[84][120];
    json_t* fc2_weight_first_layer = json_array();

    for (int i = 0; i < 84; i++) {
        json_t* fc2_weight_second_layer = json_array();
        for (int j = 0; j < 120; j++) {
            fc2_weight[i][j] = 0;
            for (int p = 0; p < 10; p++) {
                fc2_weight[i][j] += aggregated_weights[p][6][i][j];
            }
            fc2_weight[i][j] /= 10;
            json_array_append(fc2_weight_second_layer, json_real(fc2_weight[i][j]));
        }
        json_array_append(fc2_weight_first_layer, fc2_weight_second_layer);
    }
    result[6] = fc2_weight;
    json_object_set(root, "fc2.weight", fc2_weight_first_layer);


    float fc2_bias[84];
    json_t* fc2_bias_first_layer = json_array();

    for (int i = 0; i < 84; i++) {
        fc2_bias[i] = 0;
        for (int p = 0; p < 10; p++) {
            fc2_bias[i] += aggregated_weights[p][7][i];
        }
        fc2_bias[i] /= 10;
        json_array_append(fc2_bias_first_layer, json_real(fc2_bias[i]));
    }
    result[7] = fc2_bias;
    json_object_set(root, "fc2.bias", fc2_bias_first_layer);

    float fc3_weight[10][84];
    json_t* fc3_weight_first_layer = json_array();

    for (int i = 0; i < 10; i++) {
        json_t* fc3_weight_second_layer = json_array();
        for (int j = 0; j < 84; j++) {
            fc3_weight[i][j] = 0;
            for (int p = 0; p < 10; p++) {
                fc3_weight[i][j] += aggregated_weights[p][8][i][j];
            }
            fc3_weight[i][j] /= 10;
            json_array_append(fc3_weight_second_layer, json_real(fc3_weight[i][j]));
        }
        json_array_append(fc3_weight_first_layer, fc3_weight_second_layer);
    }
    result[8] = fc3_weight;
    json_object_set(root, "fc3.weight", fc3_weight_first_layer);

    float fc3_bias[10];
    json_t* fc3_bias_first_layer = json_array();
    for (int i = 0; i < 10; i++) {
        for (int p = 0; p < 10; p++) {
            fc3_bias[i] += aggregated_weights[p][9][i];
        }
        json_array_append(fc3_bias_first_layer, json_real(fc3_bias[i]));
    }
    result[9] = fc3_bias;
    json_object_set(root, "fc3.bias", fc3_bias_first_layer);

}

int main() {
    void** result = read_weights("model_weights/shared_weights");
    float* fc3_bias = (float*)result[9];
    for (int i = 0; i < 10; i++) {
        printf("%f\n", fc3_bias[i]);
    }
    return 0;
}
