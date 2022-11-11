#include "network.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

Network::Network() {}
Network::Network(int _in_n, int _max_layer_n, LOSSTYPE _losstype, float _eta)
    : in_n(_in_n), max_layer_n(_max_layer_n), losstype(_losstype), eta(_eta) {
    layer_n = 0;
    layers = new Layer *[max_layer_n];
}
Network::Network(const char *file) {
    FILE *f = fopen(file, "rb");

    if (f == NULL) {
        layer_n = -1;
        return;
    }

    int t;

    fscanf(f, "%d %d %d %d %f", &t, &in_n, &layer_n, &max_layer_n, &eta);
    losstype = LOSSTYPE(t);
    layers = new Layer *[max_layer_n];
    for (int i = 0; i < layer_n; i++) {
        int lact, lin_n, ln;
        fscanf(f, "%d %d %d\n", &lact, &lin_n, &ln);
        layers[i] = new Layer(lin_n, ln, ACTIVATION(lact));
        for (int j = 0; j < layers[i]->in_n; j++)
            for (int k = 0; k < layers[i]->n; k++)
                fscanf(f, "%f", &(layers[i]->W[j][k]));
        for (int j = 0; j < layers[i]->n; j++)
            fscanf(f, "%f", &(layers[i]->b[j]));
    }
}
Network::~Network() {
    for (int i = 0; i < layer_n; i++)
        delete layers[i];
    delete[] layers;
}

void Network::add_layer(int n, ACTIVATION activation) {
    assert(layer_n < max_layer_n);
    layers[layer_n] =
        new Layer(layer_n ? layers[layer_n - 1]->n : in_n, n, activation);
    layer_n++;
}

void Network::forward_propagation(float *in) {
    for (int i = 0; i < layer_n; i++)
        layers[i]->forward_propagation(i ? layers[i - 1]->y : in);
}
void Network::back_propagation(float *in, int out, float eta) {
    calc_dy(layers[layer_n - 1], out);
    for (int i = layer_n - 1; i >= 0; i--)
        layers[i]->backward_propagation(i ? layers[i - 1]->y : in,
                                        i ? layers[i - 1]->dy : NULL, eta);
}
void Network::train(Data data, float eta) {
    forward_propagation(data.in);
    if (std::max_element(get_out(), get_out() + get_last_layer()->n) -
            get_out() !=
        data.out)
        back_propagation(data.in, data.out, eta * STRENGTHEN_LAMBDA);
    else
        back_propagation(data.in, data.out, eta);
}
void Network::get_error_and_loss(Dataset &dataset, float *error, float *ls) {
    *error = *ls = 0;
    for (int i = 0; i < dataset.n; i++) {
        forward_propagation(dataset[i].in);
        *ls += calc_loss(layers[layer_n - 1], losstype);
        if (dataset[i].out !=
            std::max_element(layers[layer_n - 1]->y,
                             layers[layer_n - 1]->y + layers[layer_n - 1]->n) -
                layers[layer_n - 1]->y)
            (*error)++;
    }
    *error /= dataset.n;
    *ls /= dataset.n;

    for (int i = 0; i < layer_n; i++) {
        for (int j = 0; j < layers[i]->in_n; j++)
            for (int k = 0; k < layers[i]->n; k++)
                *ls += layers[i]->W[j][k] * layers[i]->W[j][k] * REG_LAMBDA;
        for (int j = 0; j < layers[i]->n; j++)
            *ls += layers[i]->b[j] * layers[i]->b[j] * REG_LAMBDA;
    }
}

float *Network::get_out() { return layers[layer_n - 1]->y; }
Layer *Network::get_last_layer() { return layers[layer_n - 1]; }

int Network::predict(float *in) {
    forward_propagation(in);
    return std::max_element(layers[layer_n - 1]->y,
                            layers[layer_n - 1]->y + layers[layer_n - 1]->n) -
           layers[layer_n - 1]->y;
}
void Network::predict_p(float *in) {
    float expsum = 0;

    forward_propagation(in);
    for (int i = 0; i < layers[layer_n - 1]->n; i++) {
        layers[layer_n - 1]->y[i] = expf(layers[layer_n - 1]->y[i]);
        expsum += layers[layer_n - 1]->y[i];
    }
    for (int i = 0; i < layers[layer_n - 1]->n; i++)
        layers[layer_n - 1]->y[i] /= expsum;
}

void Network::write_file(const char *file, float train_error, float cross_error,
                         float test_error) {
    create_parents_for_file(file);
    FILE *f = fopen(file, "wb");

    fprintf(f, "%d %d %d %d %g\n", (int)losstype, in_n, layer_n, max_layer_n,
            eta);
    for (int i = 0; i < layer_n; i++) {
        fprintf(f, "%d %d %d\n", (int)layers[i]->activate, layers[i]->in_n,
                layers[i]->n);
        for (int j = 0; j < layers[i]->in_n; j++)
            for (int k = 0; k < layers[i]->n; k++)
                fprintf(f, "%g ", layers[i]->W[j][k]);
        fputc('\n', f);
        for (int j = 0; j < layers[i]->n; j++)
            fprintf(f, "%g ", layers[i]->b[j]);
        fputc('\n', f);
    }
    fprintf(f, "train_err = %.4f%%, cross_err = %.4f%%, test_err = %.4f%%",
            train_error * 100, cross_error * 100, test_error * 100);
}

void Network::calc_dy(Layer *layer, int out) {
    memset(layer->dy, 0, layer->n * sizeof(float));
    switch (losstype) {
    case SOFTMAX: {
        float expsum = 0;
        for (int i = 0; i < layer->n; i++) {
            layer->dy[i] = expf(layer->y[i]);
            expsum += layer->dy[i];
        }
        for (int i = 0; i < layer->n; i++)
            layer->dy[i] /= expsum;
        layer->dy[out]--;
        break;
    }
    }
}
float Network::calc_loss(Layer *layer, int out) {
    float result = 0;
    switch (losstype) {
    case SOFTMAX: {
        for (int i = 0; i < layer->n; i++)
            result += expf(layer->y[i]);
        result = logf(result) - layer->y[out];
        break;
    }
    }
    return result;
}