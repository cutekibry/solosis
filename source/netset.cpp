#include "netset.h"
#include <algorithm>
#include <cassert>

Netset::Netset(int _cap, int _out_n) : cap(_cap), out_n(_out_n) {
    n = 0;
    networks = new Network *[cap];
    out = new float[out_n];
}
Netset::~Netset() {
    for (int i = 0; i < n; i++)
        delete networks[i];
    delete[] networks;
}

bool Netset::add_network(const char *file) {
    assert(n < cap);
    networks[n] = new Network(file);
    if (networks[n]->layer_n == -1 or
        networks[n]->get_last_layer()->n != out_n) {
        delete networks[n];
        return false;
    } else {
        n++;
        return true;
    }
}

void Netset::predict_p(float *in) {
    for (int i = 0; i < out_n; i++)
        out[i] = 0;
    for (int i = 0; i < n; i++) {
        networks[i]->predict_p(in);
        for (int j = 0; j < out_n; j++)
            out[j] += networks[i]->get_out()[j];
    }
    for (int i = 0; i < out_n; i++)
        out[i] /= n;
}
int Netset::predict(float *in) {
    predict_p(in);
    return std::max_element(out, out + out_n) - out;
}

float Netset::error_rate(Dataset &dataset) {
    int cnt = 0;
    for (int i = 0; i < dataset.n; i++)
        if (predict(dataset[i].in) != dataset[i].out)
            cnt++;
    return float(cnt) / dataset.n;
}