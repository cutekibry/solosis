#include "layer.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

Layer::Layer(int _in_n, int _n, ACTIVATION _activate)
    : in_n(_in_n), n(_n), activate(_activate) {
    W = new float *[in_n];
    for (int i = 0; i < in_n; i++)
        W[i] = new float[n];
    b = new float[n];
    y = new float[n];
    dy = new float[n];

    for (int i = 0; i < in_n; i++)
        for (int j = 0; j < n; j++)
            W[i][j] = 0.5 - (float)rand() / RAND_MAX;
    for (int i = 0; i < n; i++)
        b[i] = 0.5 - (float)rand() / RAND_MAX;
}

Layer::~Layer() {
    for (int i = 0; i < in_n + 1; i++)
        delete[] W[i];
    delete[] W;
    delete[] b;
    delete[] y;
    delete[] dy;
}

void Layer::forward_propagation(float *in) {
    memset(y, 0, n * sizeof(float));
    for (int i = 0; i < in_n; i++)
        for (int j = 0; j < n; j++)
            y[j] += in[i] * W[i][j];
    for (int i = 0; i < n; i++)
        y[i] += b[i];
    for (int i = 0; i < n; i++)
        y[i] = f(y[i], activate);
}

void Layer::backward_propagation(float *prev_y, float *prev_dy, float eta) {
    for (int i = 0; i < n; i++)
        dy[i] = dy[i] * gradient(y[i], activate);
    for (int i = 0; i < in_n; i++)
        for (int j = 0; j < n; j++)
            W[i][j] -= eta * (prev_y[i] * dy[j] + REG_LAMBDA * W[i][j]);
    for (int i = 0; i < n; i++)
        b[i] -= eta * (dy[i] + REG_LAMBDA * b[i]);

    if (prev_dy != NULL) {
        memset(prev_dy, 0, in_n * sizeof(float));
        for (int i = 0; i < in_n; i++)
            for (int j = 0; j < n; j++)
                prev_dy[i] += dy[j] * W[i][j];
    }
}

inline float Layer::f(float x, ACTIVATION activate) {
    switch (activate) {
    case RELU:
        return (x >= 0) ? x : 0;
    case SIGMOID:
        return 1 / (1 + expf(-x));
    }
    exit(-1);
}
inline float Layer::gradient(float x, ACTIVATION activate) {
    switch (activate) {
    case RELU:
        return x > 0;
    case SIGMOID:
        return x * (1 - x);
    }
    exit(-1);
}