#ifndef __LENET_H__
#define __LENET_H__

#include "data.h"
#include "layer.h"

const bool S2C3[16][6] = {
    {1, 1, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0},
    {0, 0, 0, 1, 1, 1}, {1, 0, 0, 0, 1, 1}, {1, 1, 0, 0, 0, 1},
    {1, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 1},
    {1, 0, 0, 1, 1, 1}, {1, 1, 0, 0, 1, 1}, {1, 1, 1, 0, 0, 1},
    {1, 1, 0, 1, 1, 0}, {0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1},
    {1, 1, 1, 1, 1, 1}};

class Lenet {

public:
    Lenet();
    Lenet(const char *filepath);
    void forward_prop(float in[32][32]);
    void backward_prop(Data &data);

    void train_batch(Dataset &dataset, int l, int n, float eta);

    int predict(float in[32][32]);
    float error_rate(Dataset &dataset);
    void error_and_loss(Dataset &dataset, float *error, float *loss);

    void save_file(const char *save_path, float train_err, float cross_err);

    void toggle_test_mode(bool t);

public:
    Conv_layer<32, 5> C1[6];     // 117600, 150 args
    Subsamp_layer<28, 2> S2[6];  // 4704
    Conv_layer<14, 5> C3[16];    // 40000, 400 args
    Subsamp_layer<10, 2> S4[16]; // 1600
    Dense_layer<400, 120> F5;    // 48000, 48000 args
    Dense_layer<120, 84> F6;     // 10080, 10080 args
    Dense_layer<84, 10> F7;      // 840, 840 args

    float S4_out[400], S4_d_out[400];
};

#endif